"""Coupled forward filtration PINNs with a time-dependent inlet concentration.

Adapted from the project's feasibility pilot. All hydraulic formulations use
the same state map, physical problem, points and loss scaling. Coefficients
specify dimensionless forward cases; no experimental parameter fitting occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from itertools import chain
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
import torch
from torch.nn.functional import softplus

from i060_coupled_pilot import grad, mlp
from i060_material import GroutMaterial


STATE_MAP = "cumulative-hazard-v2"
FEED_STATE_MAP = "feed-flux-cumulative-square-rate-v3"


@dataclass(frozen=True)
class Problem:
    phi0: float = .4
    cin: float = .1
    capacity: float = .2
    dispersion: float = .01
    retention: float = 4.
    end: float = 1.
    inlet: str = "steady"
    dilution_fraction: float = .2
    material: GroutMaterial | None = None
    boundary: str = "resident"
    switch_time: float = .55
    switch_width: float = .05
    feed_transitions: list[dict] | None = None
    initial_recipe: tuple[float, float, float] = (1., 0., 0.)

    def __post_init__(self):
        # Recorded model inputs are JSON objects when a checkpoint is resumed.
        if isinstance(self.material, dict):
            object.__setattr__(self, "material", GroutMaterial(**self.material))
        recipe = tuple(float(value) for value in self.initial_recipe)
        if (len(recipe) != 3 or not all(np.isfinite(value) and value >= 0 for value in recipe)
                or not np.isclose(sum(recipe), 1., rtol=0, atol=1e-12)):
            raise ValueError("The initial recipe requires three nonnegative volume fractions summing to one.")
        object.__setattr__(self, "initial_recipe", recipe)

    @property
    def concentration_ceiling(self):
        return self.material.concentrations[-1] if self.material else 1.

    def fraction(self, t):
        if self.boundary == "feed_flux":
            recipe = self.recipe(t)
            return recipe[:, :1]+self.dilution_fraction*recipe[:, 1:2]
        if self.inlet == "steady":
            return torch.ones_like(t)
        return self.dilution_fraction+(1-self.dilution_fraction)/2*(1-torch.tanh((t-.55)/.025))

    @property
    def transitions(self):
        if self.feed_transitions is not None:
            return self.feed_transitions
        if self.inlet == "steady":
            return []
        return [{"start": self.switch_time-self.switch_width/2,
                 "width": self.switch_width, "tail": self.inlet}]

    def recipe(self, t):
        """Stock fractions on the original clock, including interrupted blends."""
        anchor = t.new_tensor([self.initial_recipe])
        value = anchor.expand(len(t), -1)
        previous = None
        for transition in self.transitions:
            if previous is not None:
                u = min(1., (transition["start"]-previous["start"])/previous["width"])
                anchor = anchor+u*u*(3-2*u)*(target-anchor)
            target = t.new_tensor([[0., float(transition["tail"] == "dilution"),
                                   float(transition["tail"] == "water")]])
            u = ((t-transition["start"])/transition["width"]).clamp(0, 1)
            after = anchor+u*u*(3-2*u)*(target-anchor)
            value = torch.where(t >= transition["start"], after, value)
            previous = transition
        return value

    def feed_weights(self, t):
        """Volume fractions drawn from first stock, second stock and water."""
        g = self.startup(t)
        recipe = self.recipe(t)
        first, second = g*recipe[:, :1], g*recipe[:, 1:2]
        return torch.cat((first, second, 1-first-second), dim=-1)

    def startup(self, t):
        return 1-torch.exp(-(t/.06)**2)

    def capture_clock(self, t):
        # Integrated inlet startup: an O(t^3) growth envelope enforcing
        # zero initial deposition rate.
        return t-(torch.pi**.5*.06/2)*torch.erf(t/.06)

    def inlet_c(self, t):
        return self.cin*self.startup(t)*self.fraction(t)


def continued_problem(problem, time, tail):
    """Keep the realized feed prefix and replace its future by one tail blend.

    The new PINN solves the complete history, including the original startup
    and common hydration age. No snapshot is reset to an empty column.
    """
    if problem.boundary != "feed_flux" or tail not in ("water", "dilution"):
        raise ValueError("Continuation requires a feed-flux water or dilution tail.")
    if not 0 <= time <= problem.end-problem.switch_width:
        raise ValueError("The tail transition must fit within the original time horizon.")
    prefix = [dict(stage) for stage in problem.transitions if stage["start"] < time]
    return replace(problem, feed_transitions=prefix+[
        {"start": time, "width": problem.switch_width, "tail": tail}])


def material_problem(config_path, *, inlet="steady", retention=None,
                     switch_time=None, initial_recipe=(1., 0., 0.)):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    material = GroutMaterial.from_config(config)
    first = material.solid_fraction(config["feed"]["first_water_cement_mass_ratio"])
    second = material.solid_fraction(config["feed"]["second_water_cement_mass_ratio"])
    settings = dict(config["model"])
    if retention is not None:
        settings["retention"] = retention
    settings.update(config["feed"]["control"])
    if switch_time is not None:
        settings["switch_time"] = switch_time
    return Problem(**settings, inlet=inlet, cin=first,
                   dilution_fraction=second/first, material=material,
                   initial_recipe=initial_recipe)


class Fields(torch.nn.Module):
    def __init__(self, problem, width=32, capture_nodes=65):
        super().__init__()
        self.problem = problem
        self.state = mlp(2, 3, width)
        self.hazard = mlp(1, capture_nodes, width)
        self.register_buffer("capture_grid", problem.capture_clock(
            torch.linspace(0, problem.end, capture_nodes)))
        self.pressure = mlp(2, 1, width)
        self.flow = mlp(1, 1, width)

    def optimization_parameters(self, method):
        # R reconstructs the hydraulic fields from the particle states. Keep the common
        # checkpoint layout, but omit inactive hydraulic weights from L-BFGS.
        return chain(self.state.parameters(), self.hazard.parameters()) if method == "resistance" else self.parameters()

    def capture_rates(self, x):
        raw = self.hazard(2*x-1)
        # Water tails can cease capturing once the suspension has passed.
        # A square represents that zero rate without an infinite raw output.
        return raw.square() if self.problem.material else softplus(raw)

    def cumulative_hazard(self, xt):
        """Integrate nonnegative, piecewise-linear rates in the startup clock.

        Both the cumulative value and its autograd derivative use this same
        interpolant. Search selects the interval only; its endpoint values and
        partial integral retain all spatial and parameter gradients.
        """
        x, t = xt[:, :1], xt[:, 1:]
        rates = self.capture_rates(x)
        clock = self.problem.capture_clock(t)
        grid = self.capture_grid
        widths = grid.diff()
        cumulative = torch.cat((torch.zeros_like(x), torch.cumsum(
            .5*(rates[:, :-1]+rates[:, 1:])*widths, dim=1)), dim=1)
        index = (torch.searchsorted(grid, clock.contiguous(), right=True)-1).clamp(0, len(grid)-2)
        left = rates.gather(1, index)
        right = rates.gather(1, index+1)
        elapsed = clock-grid[index]
        return cumulative.gather(1, index)+elapsed*(left+.5*(right-left)*elapsed/widths[index])

    def deposition_grid(self, x, times):
        """Evaluate deposition and its exact time derivative on a tensor grid.

        Nodal rates depend only on x, so each spatial network evaluation is
        shared across all requested times instead of repeated for every pair.
        """
        rates = self.capture_rates(x[:, None])
        clock = self.problem.capture_clock(times.flatten())
        grid = self.capture_grid
        widths = grid.diff()
        cumulative = torch.cat((torch.zeros_like(x[:, None]), torch.cumsum(
            .5*(rates[:, :-1]+rates[:, 1:])*widths, dim=1)), dim=1)
        index = (torch.searchsorted(grid, clock, right=True)-1).clamp(0, len(grid)-2)
        left, right = rates[:, index], rates[:, index+1]
        elapsed = clock-grid[index]
        hazard = cumulative[:, index]+elapsed*(left+.5*(right-left)*elapsed/widths[index])
        rate = self.problem.startup(times.flatten())*(left+(right-left)*elapsed/widths[index])
        sigma = -self.problem.capacity*torch.expm1(-hazard)
        return sigma, (self.problem.capacity-sigma)*rate

    def transport_components(self, xt, flow=None):
        """Concentration and flux, with an optional feed-boundary lift."""
        p = self.problem
        x, t = xt[:, :1], xt[:, 1:]
        normalized_t = 2*t/p.end-1
        raw = self.state(torch.cat((2*x-1, normalized_t), 1))
        outlet_offset = self.state(torch.cat((torch.ones_like(t), normalized_t), 1))[:, 2:3]
        ceiling = p.concentration_ceiling
        # Separate the outlet value from the interior shape. Both polynomial
        # factors have zero derivative at x=1, without folding the raw fields
        # about the outlet. The interior factor has unit maximum on [0,1].
        if p.boundary == "feed_flux":
            inlet_fraction = torch.full_like(t, p.cin)
            logit_offset = outlet_offset+(1-x)**2*raw[:, :1]
        else:
            inlet_fraction = p.cin*p.fraction(t)
            logit_offset = x*(2-x)*outlet_offset + (27/4)*x*(1-x)**2*raw[:, :1]
        inlet_logit = torch.log(inlet_fraction/(ceiling-inlet_fraction))
        c = ceiling*p.startup(t)*torch.sigmoid(inlet_logit+logit_offset)
        diff_flux = p.cin*p.dispersion**.5*(1-x)*p.startup(t)*raw[:, 1:2]
        if p.boundary == "feed_flux":
            diff_flux = 4*x*diff_flux
            if flow is not None:
                face_c, _ = self.transport_components(torch.cat((torch.zeros_like(t), t), 1))
                diff_flux = diff_flux+(1-x)*flow*(p.inlet_c(t)-face_c)
        return c, diff_flux

    def forward(self, xt):
        c, diff_flux = self.transport_components(xt)
        sigma = -self.problem.capacity*torch.expm1(-self.cumulative_hazard(xt))
        return c, sigma, diff_flux

    def transport(self, xt, flow):
        c, diff_flux = self.transport_components(xt, flow)
        sigma = -self.problem.capacity*torch.expm1(-self.cumulative_hazard(xt))
        return c, sigma, diff_flux

    def p(self, xt):
        x, t = xt[:, :1], xt[:, 1:]
        normalized = torch.cat((2*x-1, 2*t/self.problem.end-1), 1)
        envelope = self.problem.startup(t) if self.problem.material else self.problem.capture_clock(t)
        return 1-x+envelope*x*(1-x)*self.pressure(normalized)

    def q(self, t):
        if self.problem.material:
            # Local viscosity and density start changing with c=O(t^2), and
            # water initially flows faster than the reference fresh grout.
            return self.problem.material.initial_flow*torch.exp(
                self.problem.startup(t)*self.flow(2*t/self.problem.end-1))
        return torch.exp(-self.problem.capture_clock(t)*softplus(self.flow(2*t/self.problem.end-1)))


def permeability(problem, sigma):
    return ((problem.phi0-sigma)/problem.phi0)**3


def quadrature(order):
    z, w = leggauss(order)
    return torch.as_tensor((z+1)/2), torch.as_tensor(w/2)


def hydraulic_coefficients(problem, c, sigma, times):
    """Local m/k resistance density and dimensionless opposing gravity."""
    if problem.material is None:
        return 1/permeability(problem, sigma), torch.zeros_like(c)
    return (problem.material.relative_viscosity(c, times)/permeability(problem, sigma),
            problem.material.gravity(c))


def resistance_q(net, times, rule):
    x, w = rule
    xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
    c, _ = net.transport_components(torch.stack((xx.flatten(), tt.flatten()), 1))
    sigma = net.deposition_grid(x, times)[0].reshape(-1, 1)
    density, gravity = hydraulic_coefficients(net.problem, c, sigma, tt.reshape(-1, 1))
    resistance = (w[:, None]*density.reshape(len(x), -1)).sum(0)
    drive = 1-(w[:, None]*gravity.reshape(len(x), -1)).sum(0)
    return (drive/resistance)[:, None]


def residuals(net, x, times, method, rule):
    p = net.problem
    xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
    xt = torch.stack((xx.flatten(), tt.flatten()), 1).requires_grad_(True)
    qt = resistance_q(net, times, rule) if method == "resistance" else net.q(times)
    q = qt.T.expand(len(x), -1).reshape(-1, 1)
    c, df = net.transport_components(xt, q)
    sigma, ds = (v.reshape(-1, 1) for v in net.deposition_grid(x, times))
    phi = p.phi0-sigma
    dc = grad(c, xt)
    jx = q*dc[:, :1]+grad(df, xt)[:, :1]
    capture = p.retention*q*c*(1-sigma/p.capacity)
    rows = {
        "mass": (phi*dc[:, 1:]-c*ds+jx+capture)/p.cin,
        "exchange": (ds-capture)/(p.retention*p.cin),
        "constitutive": (df+phi*p.dispersion*dc[:, :1])/(p.cin*p.dispersion**.5),
    }
    if method != "resistance":
        density, gravity = hydraulic_coefficients(p, c, sigma, xt[:, 1:])
        rows["darcy"] = q+(grad(net.p(xt), xt)[:, :1]+gravity)/density
    if method == "augmented":
        rows["hydraulic_integral"] = qt/resistance_q(net, times, rule)-1
    return rows


def objective(net, x, times, method, rule):
    return sum(r.square().mean() for r in residuals(net, x, times, method, rule).values())


def segment_inventory_residual(net, times, edges, mean_scale, method, rule,
                               *, include_capture=False):
    """Time-integrated segment balances normalized by a solid-fraction scale."""
    q = resistance_q(net, times, rule) if method == "resistance" else net.q(times)
    xx, tt = torch.meshgrid(edges, times.flatten(), indexing="ij")
    xt = torch.stack((xx.flatten(), tt.flatten()), 1)
    qe = q.T.expand(len(edges), -1)
    c, jd = net.transport_components(xt, qe.reshape(-1, 1))
    flux = (qe.reshape(-1, 1)*c+jd).reshape(len(edges), -1)
    inward = flux[:-1]-flux[1:]
    dt = times.flatten().diff()
    def integrate_time(values):
        return torch.cat((torch.zeros(len(edges)-1, 1),
                          torch.cumsum((values[:, 1:]+values[:, :-1])*dt/2, dim=1)), dim=1)

    accumulated = integrate_time(inward)
    nodes, weights = quadrature(8)
    widths = edges.diff()
    points = edges[:-1, None]+widths[:, None]*nodes
    xx, tt = torch.meshgrid(points.flatten(), times.flatten(), indexing="ij")
    c, _ = net.transport_components(torch.stack((xx.flatten(), tt.flatten()), 1))
    sigma = net.deposition_grid(points.flatten(), times)[0].reshape(-1, 1)
    storage = ((net.problem.phi0-sigma)*c+sigma).reshape(len(widths), len(nodes), -1)
    def integrate_space(values):
        values = values.reshape(len(widths), len(nodes), -1)
        return (values*weights[None, :, None]).sum(1)*widths[:, None]

    inventory = integrate_space(storage)
    scale = mean_scale*widths[:, None]
    total_balance = (inventory-inventory[:, :1]-accumulated)/scale
    if not include_capture:
        return total_balance
    deposited = integrate_space(sigma)
    capture = (net.problem.retention*q.T[None, :, :]
               *c.reshape(len(widths), len(nodes), -1)
               *(1-sigma.reshape(len(widths), len(nodes), -1)/net.problem.capacity))
    captured = integrate_time(integrate_space(capture))
    exchange_balance = (deposited-deposited[:, :1]-captured)/scale
    return torch.stack((total_balance, exchange_balance))


def evaluate(net, method, rule, *, regional_residuals=False,
             spatial_points=301, temporal_points=201):
    problem = net.problem
    x = torch.linspace(0, 1, spatial_points)
    times = torch.linspace(0, problem.end, temporal_points)
    if problem.boundary == "feed_flux" and problem.transitions:
        # Off-grid switch boundaries must be explicit integration knots. In
        # particular, interpolation must not leak cement past pure-water start.
        events = [stage["start"]+fraction*stage["width"]
                  for stage in problem.transitions for fraction in (0., .5, 1.)]
        times = torch.unique(torch.cat((times, times.new_tensor(events))))
    times = times[:, None]
    xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
    xt = torch.stack((xx.flatten(), tt.flatten()), 1)
    with torch.no_grad():
        qr = resistance_q(net, times, quadrature(96))
        q = resistance_q(net, times, rule) if method == "resistance" else net.q(times)
        flow = q.T.expand(len(x), -1).reshape(-1, 1)
        c, df = (v.reshape(len(x), len(times)) for v in net.transport_components(xt, flow))
        sigma, _ = net.deposition_grid(x, times)
        j = c*q.T+df
        mobile = torch.trapezoid((problem.phi0-sigma)*c, x, dim=0)
        deposited = torch.trapezoid(sigma, x, dim=0)
        dt = times[1:, 0]-times[:-1, 0]
        net_in = j[0]-j[-1]
        influx = torch.cat((torch.zeros(1), torch.cumsum((net_in[1:]+net_in[:-1])*dt/2, 0)))
        balance = mobile+deposited-influx
        if method == "resistance":
            # Display pressure reconstructed from the same resistance law.
            density, gravity = hydraulic_coefficients(problem, c, sigma, tt)
            cum_r = torch.cat((torch.zeros(1, len(times)), torch.cumsum(
                (density[1:]+density[:-1])*(x[1]-x[0])/2, dim=0)), dim=0)
            cum_g = torch.cat((torch.zeros(1, len(times)), torch.cumsum(
                (gravity[1:]+gravity[:-1])*(x[1]-x[0])/2, dim=0)), dim=0)
            pressure = 1-cum_g-cum_r*q.T
        else:
            pressure = net.p(xt).reshape(len(x), len(times))
    xv = (torch.arange(96)+.5)/96
    tv = ((torch.arange(100)+.5)*problem.end/100)[:, None]
    rv = residuals(net, xv, tv, method, quadrature(64))
    metrics = {k+"_rms": float(v.detach().square().mean().sqrt()) for k, v in rv.items()}
    if regional_residuals:
        xxv, ttv = torch.meshgrid(xv, tv.flatten(), indexing="ij")
        supports = {
            "whole_domain": torch.ones_like(xxv, dtype=torch.bool),
            "startup": ttv <= .15,
            "inlet_third": xxv < 1/3,
            "outlet_third": xxv > 2/3,
            "late_time": ttv >= .7,
        }
        if problem.inlet == "dilution":
            supports["dilution_transition"] = (ttv >= .48) & (ttv <= .62)
        scales = {"mass": problem.cin, "exchange": problem.retention*problem.cin,
                  "constitutive": problem.cin*problem.dispersion**.5}
        metrics["particle_residual_rms_unscaled"] = {
            support: {name: float((rv[name].detach().flatten()[mask.flatten()]*scale)
                                 .square().mean().sqrt())
                      for name, scale in scales.items()}
            for support, mask in supports.items()
        }
    metrics.update(
        q_final=float(q[-1]), outlet_c_final=float(c[-1, -1]),
        sigma_max=float(sigma.max()), concentration_min=float(c.min()),
        concentration_max=float(c.max()), deposited_final=float(deposited[-1]),
        mobile_final=float(mobile[-1]), max_mass_defect=float(balance.abs().max()),
        max_pressure_flow_defect=float((q-qr).abs().max()),
        max_interior_above_current_inlet=float((c[1:]-problem.inlet_c(times).T).max()),
        cumulative_inlet_solid=float(torch.trapezoid(j[0], times.flatten())),
        cumulative_outlet_solid=float(torch.trapezoid(j[-1], times.flatten())),
    )
    fields = {name: value.cpu().numpy() for name, value in {
        "x": x, "times": times.flatten(), "c": c, "sigma": sigma,
        "q": q, "j": j, "pressure": pressure, "mass_defect": balance,
        "mobile": mobile, "deposited": deposited, "inlet_c": c[0],
    }.items()}
    if problem.boundary == "feed_flux":
        metrics["max_interior_above_feed"] = metrics.pop("max_interior_above_current_inlet")
        fields["feed_c"] = problem.inlet_c(times).flatten().cpu().numpy()
        fields["feed_weights"] = problem.feed_weights(times).cpu().numpy()
        metrics["max_feed_flux_defect"] = float(
            (j[0]-q.flatten()*problem.inlet_c(times).flatten()).abs().max())
    return metrics, fields


def initialize_capture_from_transport(net, source):
    """Initialize new rates from the previous c and q, never its deposition map.

    This is an initial guess for the subsequent coupled physics training.
    Integrating the retention-law hazard preserves the new state semantics.
    """
    x = torch.linspace(0, 1, 129)
    times = torch.linspace(0, net.problem.end, len(net.capture_grid)).clamp_min(1e-5)
    xx, tt = torch.meshgrid(x, times, indexing="ij")
    with torch.no_grad():
        c, _ = net.transport_components(torch.stack((xx.flatten(), tt.flatten()), 1))
        with np.load(source/"fields.npz") as fields:
            q = torch.as_tensor(np.interp(times.cpu().numpy(), fields["times"], fields["q"].reshape(-1)))
        target = (net.problem.retention/net.problem.capacity
                  *c.reshape(len(x), -1)*q/net.problem.startup(times))
    optimizer = torch.optim.Adam(net.hazard.parameters(), lr=.01)
    for _ in range(1000):
        optimizer.zero_grad(set_to_none=True)
        raw = net.hazard(2*x[:, None]-1)
        if net.problem.material:
            loss = (raw-torch.sqrt(target)).square().mean()
        else:
            loss = (torch.log1p(softplus(raw))-torch.log1p(target)).square().mean()
        loss.backward()
        optimizer.step()
    print(json.dumps({"initialization": "retention-law hazard from previous transport",
                      "loss": float(loss.detach())}), flush=True)


def train(args):
    torch.manual_seed(args.seed)
    problem = (material_problem(args.material_config, inlet=args.inlet, retention=args.retention,
                                switch_time=args.switch_time, initial_recipe=args.initial_recipe)
               if args.material_config else Problem(inlet=args.inlet, retention=args.retention))
    if args.history:
        source = json.loads((args.history/"result.json").read_text(encoding="utf-8"))
        if source["config"]["state_map"] != FEED_STATE_MAP:
            raise ValueError("Continuation requires the current cumulative-capture history.")
        problem = continued_problem(Problem(**source["model"]), args.query_time, args.tail)
    if problem.inlet == "water" and problem.boundary != "feed_flux":
        raise ValueError("Pure-water switching requires the feed-flux state map.")
    if args.capture_nodes is None:
        args.capture_nodes = 257 if problem.material else 65
    net = Fields(problem, args.width, args.capture_nodes)
    state_map = FEED_STATE_MAP if problem.boundary == "feed_flux" else STATE_MAP
    previous_stages = []
    if args.initialize_transport:
        original = json.loads((args.initialize_transport/"result.json").read_text(encoding="utf-8"))
        previous_map = original["config"]["state_map"]
        if previous_map not in ("integrated-startup-v1", "feed-flux-startup-v1", STATE_MAP,
                                 "feed-flux-cumulative-hazard-v2", FEED_STATE_MAP):
            raise ValueError("Transport initialization requires a supported particle map.")
        old = torch.load(args.initialize_transport/"model.pt", weights_only=True, map_location=args.device)
        transport = {key.removeprefix("state."): value for key, value in old.items() if key.startswith("state.")}
        if previous_map in ("integrated-startup-v1", "feed-flux-startup-v1"):
            for key in ("6.weight", "6.bias"):
                transport[key] = transport[key][[0, 2, 3]]
        net.state.load_state_dict(transport)
        for name in ("pressure", "flow"):
            getattr(net, name).load_state_dict({
                key.removeprefix(name+"."): value for key, value in old.items()
                if key.startswith(name+".")})
        initialize_capture_from_transport(net, args.initialize_transport)
    if args.resume:
        original = json.loads((args.resume/"result.json").read_text(encoding="utf-8"))
        if original["config"]["state_map"] != state_map:
            raise ValueError("Warm start must use the same particle state map.")
        checkpoint = torch.load(args.resume/"model.pt", weights_only=True, map_location=args.device)
        grid_changed = checkpoint["capture_grid"].shape != net.capture_grid.shape
        if grid_changed:
            initial = net.state_dict()
            for key in ("capture_grid", "hazard.6.weight", "hazard.6.bias"):
                checkpoint[key] = initial[key]
        net.load_state_dict(checkpoint)
        if grid_changed:
            initialize_capture_from_transport(net, args.resume)
        previous_stages = original.get("training_stages", [{
            "config": original["config"], "model": original["model"],
            "seconds": original["seconds"],
        }])
    gen = torch.Generator(device=args.device).manual_seed(args.seed+1000)
    rule = quadrature(args.order)
    focus_times = (torch.linspace(*args.focus_interval, 81)[:, None]
                   if args.focus_interval else None)
    focus_x = (torch.arange(96)+.5)/96
    if args.inventory_loss:
        supply = json.loads(args.material_config.read_text(encoding="utf-8"))
        balance = supply["inventory_balance"]
        edges = torch.tensor(balance["segment_edges"])
        mean_scale = balance["solid_fraction_scale"]
        balance_times = [torch.linspace(0, problem.end, 201), torch.linspace(0, .15, 101)]
        for stage in problem.transitions:
            balance_times.append(torch.linspace(stage["start"]-stage["width"]/2,
                                                 stage["start"]+1.5*stage["width"], 65).clamp(0, problem.end))
        if focus_times is not None:
            balance_times.append(focus_times.flatten())
        balance_times = torch.unique(torch.cat(balance_times))[:, None]
        focus_mask = ((balance_times.flatten() >= args.focus_interval[0])
                      & (balance_times.flatten() <= args.focus_interval[1])) if focus_times is not None else None

    def training_loss(x, t):
        loss = objective(net, x, t, args.method, rule)
        if focus_times is not None:
            loss = loss+objective(net, focus_x, focus_times, args.method, rule)
        if args.inventory_loss:
            balance = segment_inventory_residual(net, balance_times, edges, mean_scale,
                                                 args.method, rule,
                                                 include_capture=args.capture_inventory_loss)
            loss = loss+args.inventory_weight*balance.square().mean(dim=(-2, -1)).sum()
            if focus_mask is not None:
                loss = loss+args.inventory_weight*balance[..., focus_mask].square().mean(dim=(-2, -1)).sum()
        return loss

    def samples():
        nx, nt = args.spatial_samples, args.temporal_samples
        x = torch.cat(((torch.arange(nx)+torch.rand(nx, generator=gen))/nx,
                       torch.rand(8, generator=gen)*.15))
        if args.outlet_samples:
            x = torch.cat((x, torch.linspace(.8, 1., args.outlet_samples)))
        t = torch.cat(((torch.arange(nt)[:, None]+torch.rand(nt, 1, generator=gen))/nt,
                       torch.rand(16, 1, generator=gen)*.15,
                       torch.tensor([[0.], [1.]])))*problem.end
        if problem.inlet != "steady" and problem.boundary == "resident":
            t = torch.cat((t, .48+.14*torch.rand(16, 1, generator=gen)))
        elif problem.boundary == "feed_flux":
            for stage in problem.transitions:
                t = torch.cat((t, stage["start"]-stage["width"]
                               +3*stage["width"]*torch.rand(24, 1, generator=gen)))
        return x, t.clamp(0, problem.end)

    start = time.perf_counter()
    parameters = tuple(net.optimization_parameters(args.method))
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    x, t = samples()
    history = []
    for step in range(args.adam):
        if step % 200 == 0:
            x, t = samples()
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(x, t)
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == args.adam-1:
            row = {"step": step+1, "loss": float(loss.detach()), "seconds": time.perf_counter()-start}
            history.append(row)
            print(json.dumps(row), flush=True)
    evaluations = 0
    optimizer = torch.optim.LBFGS(parameters, max_iter=args.lbfgs, history_size=50,
                                  line_search_fn="strong_wolfe",
                                  tolerance_change=1e-12 if problem.boundary == "feed_flux" else 1e-9)

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(x, t)
        loss.backward()
        evaluations += 1
        if evaluations % 200 == 0:
            print(json.dumps({"evaluation": evaluations, "loss": float(loss.detach()),
                              "seconds": time.perf_counter()-start}), flush=True)
        return loss

    if args.lbfgs:
        optimizer.step(closure)
    metrics, fields = evaluate(net, args.method, rule,
                               temporal_points=1001 if problem.boundary == "feed_flux" else 201)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"fields.npz", **fields)
    torch.save(net.state_dict(), args.output/"model.pt")
    config = vars(args).copy()
    config["output"] = str(args.output)
    if args.resume:
        config["resume"] = str(args.resume)
    if args.initialize_transport:
        config["initialize_transport"] = str(args.initialize_transport)
    if args.material_config:
        config["material_config"] = str(args.material_config)
        config["hydraulic_map"] = "local-viscosity-gravity-v1"
    if args.history:
        config["history"] = str(args.history)
    config["state_map"] = state_map
    result = {"config": config, "model": asdict(problem), "metrics": metrics,
              "seconds": time.perf_counter()-start, "lbfgs_evaluations": evaluations,
              "history": history, "runtime": {"torch": torch.__version__, "dtype": "float64", "device": args.device}}
    result["training_stages"] = previous_stages+[
        {"config": config, "model": asdict(problem), "seconds": result["seconds"]}]
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"method": args.method, "inlet": problem.inlet, "seed": args.seed,
                      "metrics": metrics, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["differential", "resistance", "augmented"], default="resistance")
    parser.add_argument("--inlet", choices=["steady", "dilution", "water"], default="steady")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--capture-nodes", type=int)
    parser.add_argument("--order", type=int, default=32)
    parser.add_argument("--retention", type=float, default=4.)
    parser.add_argument("--material-config", type=Path)
    parser.add_argument("--switch-time", type=float)
    parser.add_argument("--initial-recipe", type=float, nargs=3, default=(1., 0., 0.),
                        metavar=("RICH", "DILUTE", "WATER"))
    parser.add_argument("--history", type=Path)
    parser.add_argument("--query-time", type=float)
    parser.add_argument("--tail", choices=["water", "dilution"])
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--initialize-transport", type=Path)
    parser.add_argument("--spatial-samples", type=int, default=32)
    parser.add_argument("--temporal-samples", type=int, default=64)
    parser.add_argument("--outlet-samples", type=int, default=0)
    parser.add_argument("--inventory-loss", action="store_true")
    parser.add_argument("--capture-inventory-loss", action="store_true")
    parser.add_argument("--inventory-weight", type=float, default=1.)
    parser.add_argument("--focus-interval", type=float, nargs=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--adam", type=int, default=4000)
    parser.add_argument("--lbfgs", type=int, default=600)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value is not None for value in (args.history, args.query_time, args.tail)):
        if args.history is None or args.query_time is None or args.tail is None or args.material_config is None:
            parser.error("Continuation requires --history, --query-time, --tail and --material-config")
    if args.inventory_loss and not args.material_config:
        parser.error("--inventory-loss requires --material-config with a delivery task")
    if args.capture_inventory_loss and not args.inventory_loss:
        parser.error("--capture-inventory-loss requires --inventory-loss")
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(args.device)
    torch.set_num_threads(args.threads)
    train(args)

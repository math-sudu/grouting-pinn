"""Solve old/new particle labels on the two frozen material PINN states.

Both labels solve the linear transport/capture equations independently.
The parent flow, pore volume, its time derivative and saturation are fixed;
neither label is a complement or a renormalized fraction of the parent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from numpy import trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_i060_supply_response import CASES
from i060_coupled import Fields, Problem, quadrature, resistance_q
from i060_coupled_pilot import grad, mlp

START = .525
EDGES = np.linspace(0., 1., 7)


def cumulative(values, times, axis=0):
    values = np.moveaxis(values, axis, 0)
    dt = np.diff(times).reshape((-1,)+(1,)*(values.ndim-1))
    integral = np.concatenate((np.zeros_like(values[:1]),
                               np.cumsum(.5*(values[1:]+values[:-1])*dt, axis=0)), axis=0)
    return np.moveaxis(integral, 0, axis)


class Labels(torch.nn.Module):
    def __init__(self, width, cin, dispersion, end, *, positive=True):
        super().__init__()
        self.state = mlp(3, 4, width)
        self.cin, self.dispersion, self.end = cin, dispersion, end
        self.positive = positive

    def forward(self, xt, q, feed):
        x, u = xt[:, :1], xt[:, 1:]
        raw = self.state(torch.cat((2*x-1, 2*u-1, 2*u.sqrt()-1), 1))
        c = self.concentrations(raw[:, :2], u)
        face_raw = self.state(torch.cat((-torch.ones_like(x), 2*u-1, 2*u.sqrt()-1), 1))
        face_c = self.concentrations(face_raw[:, :2], u)
        incoming = torch.cat((torch.zeros_like(feed), q*feed), 1)
        j = (q*c+(1-x)*(incoming-q*face_c)
             +x*(1-x)*self.cin*self.dispersion**.5*raw[:, 2:])
        return c, j

    def concentrations(self, raw, u):
        if not self.positive:
            return self.cin*raw
        positive = .5*(raw+torch.sqrt(raw.square()+1e-6))
        startup = torch.cat((torch.ones_like(u), -torch.expm1(-u/.002)), 1)
        return self.cin*startup*positive


class Parent:
    def __init__(self, name):
        self.source = ROOT / CASES[name]
        result = json.loads((self.source / "result.json").read_text())
        self.problem = Problem(**result["model"])
        config = result["config"]
        self.net = Fields(self.problem, config["width"], config["capture_nodes"])
        self.net.load_state_dict(torch.load(self.source / "model.pt", weights_only=True,
                                            map_location=torch.get_default_device()))
        self.net.requires_grad_(False)
        self.rule = quadrature(config["order"])
        geometry = json.loads((ROOT / "config/i060_material_delivery.json").read_text())["column"]
        self.scale = (np.pi*geometry["diameter_m"]**2/4*self.problem.material.length_m
                      *self.problem.material.cement_density_kg_m3*1000)

    @torch.no_grad()
    def data(self, x, times):
        p = self.problem
        xx, tt = torch.meshgrid(x, times, indexing="ij")
        physical_xt = torch.stack((xx.flatten(), tt.flatten()), 1)
        sigma, ds = self.net.deposition_grid(x, times)
        c, _ = self.net.transport_components(physical_xt)
        q = resistance_q(self.net, times[:, None], self.rule).T.expand(len(x), -1)
        return {
            "xt": torch.stack((xx.flatten(), (tt.flatten()-START)/(p.end-START)), 1),
            "q": q.reshape(-1, 1), "phi": (p.phi0-sigma).reshape(-1, 1),
            "ds": ds.reshape(-1, 1), "sigma": sigma,
            "capture": (p.retention*q*(1-sigma/p.capacity)).reshape(-1, 1),
            "feed": p.inlet_c(physical_xt[:, 1:]), "c": c,
        }


def residual(net, data):
    xt = data["xt"].detach().requires_grad_(True)
    c, j = net(xt, data["q"], data["feed"])
    mass, constitutive = [], []
    for k in range(2):
        dc, dj = grad(c[:, k:k+1], xt), grad(j[:, k:k+1], xt)
        mass.append((data["phi"]*dc[:, 1:]/(net.end-START)
                     -data["ds"]*c[:, k:k+1]+dj[:, :1]
                     +data["capture"]*c[:, k:k+1])/net.cin)
        constitutive.append((j[:, k:k+1]-data["q"]*c[:, k:k+1]
                             +data["phi"]*net.dispersion*dc[:, :1])
                            /(net.cin*net.dispersion**.5))
    return torch.cat(mass, 1), torch.cat(constitutive, 1)


def inventory_data(parent, times):
    nodes, weights = quadrature(10)
    edges = torch.as_tensor(EDGES)
    points = (edges[:-1, None]+edges.diff()[:, None]*nodes).flatten()
    data = parent.data(points, times)
    face = parent.data(edges, times)
    return data, face, weights/6


def inventory_residual(net, data, face, weights, times):
    nt = len(times)
    c, _ = net(data["xt"], data["q"], data["feed"])
    _, j = net(face["xt"], face["q"], face["feed"])
    def integrate_x(v):
        return (v.reshape(6, len(weights), nt, 2)*weights[None, :, None, None]).sum(1)
    def integrate_t(v):
        return torch.cat((torch.zeros_like(v[:, :1]), torch.cumsum(
            .5*(v[:, 1:]+v[:, :-1])*times.diff()[None, :, None], dim=1)), dim=1)
    mobile = integrate_x(data["phi"]*c)
    capture = integrate_t(integrate_x(data["capture"]*c))
    j = j.reshape(7, nt, 2)
    supplied = integrate_t(j[:-1]-j[1:])
    parent_initial = (data["phi"]*data["c"]).reshape(6, len(weights), nt)
    old_initial = (parent_initial[:, :, :1]*weights[None, :, None]).sum(1)
    initial = torch.stack((old_initial, torch.zeros_like(old_initial)), -1)
    return (mobile-initial+capture-supplied)/(.01/6)


def nodal_weights(points):
    widths = points.diff()
    return torch.cat((widths[:1]/2, (widths[:-1]+widths[1:])/2, widths[-1:]/2))


def train(parent, args, destination):
    torch.manual_seed(args.seed)
    p = parent.problem
    net = Labels(args.width, p.cin, p.dispersion, p.end, positive=not args.initialize)
    if args.resume:
        net.load_state_dict(torch.load(destination / "model.pt", weights_only=True,
                                      map_location=torch.get_default_device()))
    # A square-root time coordinate resolves the label boundary jump at START.
    x = torch.unique(torch.cat(((torch.arange(args.nx)+.5)/args.nx,
                                torch.linspace(0, .08, 17), torch.tensor([1.]))))
    u = torch.unique(torch.cat(((torch.arange(args.nt)+.5)/args.nt,
                                torch.logspace(-4, -1, 32), torch.tensor([1.]))))
    data = parent.data(x, START+(p.end-START)*u)
    measure = (nodal_weights(x)[:, None]*nodal_weights(u)[None, :]).reshape(-1, 1)
    xi = torch.linspace(0, 1, 301)
    initial = parent.data(xi, torch.tensor([START]))
    initial_c = torch.cat((initial["c"], torch.zeros_like(initial["c"])), 1)
    times = START+(p.end-START)*torch.unique(torch.cat((torch.linspace(0, 1, 257),
                                                      torch.logspace(-6, -1, 65))))
    inventory, face, weights = inventory_data(parent, times)
    start = time.perf_counter()
    def loss():
        mass, constitutive = residual(net, data)
        c0, _ = net(initial["xt"], initial["q"], initial["feed"])
        balance = inventory_residual(net, inventory, face, weights, times)
        value = ((measure*(mass.square()+constitutive.square())).sum()/2
                 +100*((c0-initial_c)/p.cin).square().mean()+balance.square().mean())
        if args.initialize:
            c, _ = net(data["xt"], data["q"], data["feed"])
            value = value+10*(measure*(torch.relu(-c)/p.cin).square()).sum()
        return value
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    for step in range(args.adam):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        optimizer.step()
        if (step+1) % 200 == 0:
            print(json.dumps({"step": step+1, "loss": value.item(),
                              "seconds": time.perf_counter()-start}), flush=True)
    optimizer = torch.optim.LBFGS(net.parameters(), max_iter=args.lbfgs, history_size=75,
                                  line_search_fn="strong_wolfe", tolerance_change=1e-13,
                                  tolerance_grad=1e-10)
    evaluations = 0
    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        evaluations += 1
        if evaluations % 200 == 0:
            print(json.dumps({"evaluation": evaluations, "loss": value.item(),
                              "seconds": time.perf_counter()-start}), flush=True)
        return value
    if args.lbfgs:
        optimizer.step(closure)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), destination / "model.pt")
    result = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "source": str(parent.source.relative_to(ROOT)).replace("\\", "/"),
              "start": START, "seconds": time.perf_counter()-start,
              "loss": loss().item(), "lbfgs_evaluations": evaluations}
    result["concentration_map"] = ("linear-initialization-v1" if args.initialize
                                   else "smooth-positive-old-zero-initial-new-v1")
    if args.resume:
        previous = json.loads((destination / "result.json").read_text())
        result["refinement_reference_g"] = {name: previous["result"]["origins"][name]["downstream_captured_g"]
                                             for name in ("old", "new")}
        result["training_stages"] = previous.get("training_stages", [previous["config"]])+[result["config"]]
    return net, result


@torch.no_grad()
def evaluate(net, parent, destination):
    x = torch.linspace(0, 1, 601)
    times = START+(parent.problem.end-START)*torch.unique(torch.cat((
        torch.linspace(0, 1, 1001), torch.logspace(-7, -1, 101))))
    chunks = []
    parent_c, parent_sigma, parent_phi, coefficients = [], [], [], []
    for sub_times in times.split(80):
        data = parent.data(x, sub_times)
        c, j = net(data["xt"], data["q"], data["feed"])
        chunks.append((c.reshape(len(x), -1, 2).cpu().numpy(),
                       j.reshape(len(x), -1, 2).cpu().numpy()))
        parent_c.append(data["c"].reshape(len(x), -1).cpu().numpy())
        parent_sigma.append(data["sigma"].cpu().numpy())
        parent_phi.append(data["phi"].reshape(len(x), -1).cpu().numpy())
        coefficients.append(data["capture"].reshape(len(x), -1).cpu().numpy())
    c = np.concatenate([v[0] for v in chunks], axis=1)
    j = np.concatenate([v[1] for v in chunks], axis=1)
    total_c, sigma, phi, k = [np.concatenate(v, axis=1)
                             for v in (parent_c, parent_sigma, parent_phi, coefficients)]
    x, times = x.cpu().numpy(), times.cpu().numpy()
    deposit = cumulative(k[:, :, None]*c, times, axis=1)
    scale = parent.scale
    mobile = trapezoid(phi[:, :, None]*c, x, axis=0)*scale
    captured = trapezoid(deposit, x, axis=0)*scale
    outlet = cumulative(j[-1], times)*scale
    inlet = cumulative(j[0], times)*scale
    downstream = x >= 1/3-1e-12
    downstream_captured = trapezoid(deposit[downstream], x[downstream], axis=0)*scale
    parent_initial_mobile = trapezoid(phi[:, 0]*total_c[:, 0], x)*scale
    initial_deposit = trapezoid(sigma[downstream, 0], x[downstream])*scale
    final_deposit = trapezoid(sigma[downstream, -1], x[downstream])*scale
    parent_gain = sigma-sigma[:, :1]
    label_gain = deposit.sum(axis=2)
    parent_mobile = trapezoid(phi*total_c, x, axis=0)*scale
    balance = np.array([parent_initial_mobile, 0.])+inlet-mobile-captured-outlet
    report = {
        "parent_initial_mobile_g": float(parent_initial_mobile),
        "initial_downstream_deposit_g": float(initial_deposit),
        "final_downstream_deposit_g": float(final_deposit),
        "origins": {name: {
            "initial_mobile_g": float(mobile[0, n]),
            "inlet_g": float(inlet[-1, n]), "captured_g": float(captured[-1, n]),
            "downstream_captured_g": float(downstream_captured[-1, n]),
            "outlet_g": float(outlet[-1, n]), "remaining_mobile_g": float(mobile[-1, n]),
            "final_balance_defect_g": float(balance[-1, n]),
            "max_balance_defect_g": float(np.max(np.abs(balance[:, n]))),
            "min_concentration": float(c[:, :, n].min()),
        } for n, name in enumerate(("old", "new"))},
        "compatibility": {
            "concentration_sum_relative_l2": float(np.sqrt(
                trapezoid(trapezoid((c.sum(2)-total_c)**2, x, axis=0), times)
                /trapezoid(trapezoid(total_c**2, x, axis=0), times))),
            "max_mobile_sum_defect_g": float(np.max(np.abs(mobile.sum(1)-parent_mobile))),
            "final_deposit_sum_defect_g": float(trapezoid(label_gain[:, -1]-parent_gain[:, -1], x)*scale),
            "downstream_deposit_sum_defect_g": float(downstream_captured[-1].sum()-(final_deposit-initial_deposit)),
            "max_downstream_deposit_sum_defect_g": float(np.max(np.abs(
                downstream_captured.sum(1)-trapezoid(parent_gain[downstream], x[downstream], axis=0)*scale))),
            "initial_mobile_error_g": float(mobile[0, 0]-parent_initial_mobile),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["dilution", "mixed"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/i060_origins")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--nt", type=int, default=160)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--adam", type=int, default=800)
    parser.add_argument("--lbfgs", type=int, default=2400)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--initialize", action="store_true",
                           help="Fit the linear-output warm start before nonnegative refinement")
    operation.add_argument("--resume", action="store_true")
    operation.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(args.device)
    torch.set_num_threads(2)
    parent = Parent(args.case)
    destination = args.output / args.case
    if args.evaluate_only:
        result = json.loads((destination / "result.json").read_text())
        maps = {"linear-initialization-v1": False, "smooth-positive-old-zero-initial-new-v1": True}
        net = Labels(result["config"]["width"], parent.problem.cin,
                     parent.problem.dispersion, parent.problem.end,
                     positive=maps[result["concentration_map"]])
        net.load_state_dict(torch.load(destination / "model.pt", weights_only=True,
                                      map_location=torch.get_default_device()))
    else:
        net, result = train(parent, args, destination)
    result["result"] = evaluate(net, parent, destination)
    if "previous_result" in result:
        previous = result.pop("previous_result")
        result["refinement_reference_g"] = {name: previous["origins"][name]["downstream_captured_g"]
                                             for name in ("old", "new")}
    if "refinement_reference_g" in result:
        result["downstream_change_g"] = {name: result["result"]["origins"][name]["downstream_captured_g"]
            -result["refinement_reference_g"][name] for name in ("old", "new")}
    (destination / "result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()

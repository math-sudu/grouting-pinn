"""Coupled forward filtration PINNs with a time-dependent inlet concentration.

Adapted from the project's feasibility pilot. All hydraulic formulations use
the same state map, physical problem, points and loss scaling. Coefficients
specify dimensionless forward cases; no experimental parameter fitting occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
import torch
from torch.nn.functional import softplus

from i060_coupled_pilot import grad, mlp


@dataclass(frozen=True)
class Problem:
    phi0: float = .4
    cin: float = .1
    capacity: float = .2
    dispersion: float = .01
    retention: float = 4.
    end: float = 1.
    inlet: str = "steady"

    def fraction(self, t):
        if self.inlet == "steady":
            return torch.ones_like(t)
        return .2+.4*(1-torch.tanh((t-.55)/.025))

    def startup(self, t):
        return 1-torch.exp(-(t/.06)**2)

    def inlet_c(self, t):
        return self.cin*self.startup(t)*self.fraction(t)


class Fields(torch.nn.Module):
    def __init__(self, problem, width=32):
        super().__init__()
        self.problem = problem
        self.state = mlp(2, 3, width)
        self.pressure = mlp(2, 1, width)
        self.flow = mlp(1, 1, width)

    def forward(self, xt):
        p = self.problem
        x, t = xt[:, :1], xt[:, 1:]
        y = x*(2-x)
        raw = self.state(torch.cat((2*y-1, 2*t/p.end-1), 1))
        inlet_fraction = p.cin*p.fraction(t)
        inlet_logit = torch.log(inlet_fraction/(1-inlet_fraction))
        # At x=0 this is exactly the prescribed inlet. Interior concentration
        # can exceed the current inlet after dilution, while remaining in [0,1].
        c = p.startup(t)*torch.sigmoid(inlet_logit+y*raw[:, :1])
        sigma = p.capacity*(1-torch.exp(-t*softplus(raw[:, 1:2])))
        diff_flux = p.cin*p.dispersion**.5*(1-x)*p.startup(t)*raw[:, 2:3]
        return c, sigma, diff_flux

    def p(self, xt):
        x, t = xt[:, :1], xt[:, 1:]
        normalized = torch.cat((2*x-1, 2*t/self.problem.end-1), 1)
        return 1-x+t*x*(1-x)*self.pressure(normalized)

    def q(self, t):
        return torch.exp(-t*softplus(self.flow(2*t/self.problem.end-1)))


def permeability(problem, sigma):
    return ((problem.phi0-sigma)/problem.phi0)**3


def quadrature(order):
    z, w = leggauss(order)
    return torch.as_tensor((z+1)/2), torch.as_tensor(w/2)


def resistance_q(net, times, rule):
    x, w = rule
    xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
    _, sigma, _ = net(torch.stack((xx.flatten(), tt.flatten()), 1))
    resistance = (w[:, None]/permeability(net.problem, sigma).reshape(len(x), -1)).sum(0)
    return 1/resistance[:, None]


def residuals(net, x, times, method, rule):
    p = net.problem
    xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
    xt = torch.stack((xx.flatten(), tt.flatten()), 1).requires_grad_(True)
    c, sigma, df = net(xt)
    phi = p.phi0-sigma
    qt = resistance_q(net, times, rule) if method == "resistance" else net.q(times)
    q = qt.T.expand(len(x), -1).reshape(-1, 1)
    dc, ds = grad(c, xt), grad(sigma, xt)
    jx = q*dc[:, :1]+grad(df, xt)[:, :1]
    capture = p.retention*q*c*(1-sigma/p.capacity)
    rows = {
        "mass": (phi*dc[:, 1:]-c*ds[:, 1:]+jx+capture)/p.cin,
        "exchange": (ds[:, 1:]-capture)/(p.retention*p.cin),
        "constitutive": (df+phi*p.dispersion*dc[:, :1])/(p.cin*p.dispersion**.5),
    }
    if method != "resistance":
        rows["darcy"] = q+permeability(p, sigma)*grad(net.p(xt), xt)[:, :1]
    if method == "augmented":
        rows["hydraulic_integral"] = qt/resistance_q(net, times, rule)-1
    return rows


def objective(net, x, times, method, rule):
    return sum(r.square().mean() for r in residuals(net, x, times, method, rule).values())


def evaluate(net, method, rule):
    problem = net.problem
    x = torch.linspace(0, 1, 301)
    times = torch.linspace(0, problem.end, 201)[:, None]
    xx, tt = torch.meshgrid(x, times.flatten(), indexing="ij")
    xt = torch.stack((xx.flatten(), tt.flatten()), 1)
    with torch.no_grad():
        c, sigma, df = (v.reshape(len(x), len(times)) for v in net(xt))
        qr = resistance_q(net, times, quadrature(96))
        q = resistance_q(net, times, rule) if method == "resistance" else net.q(times)
        j = c*q.T+df
        mobile = torch.trapezoid((problem.phi0-sigma)*c, x, dim=0)
        deposited = torch.trapezoid(sigma, x, dim=0)
        dt = times[1, 0]-times[0, 0]
        net_in = j[0]-j[-1]
        influx = torch.cat((torch.zeros(1), torch.cumsum((net_in[1:]+net_in[:-1])*dt/2, 0)))
        balance = mobile+deposited-influx
        if method == "resistance":
            # Display pressure reconstructed from the same resistance law.
            inv_k = 1/permeability(problem, sigma)
            cum_r = torch.cat((torch.zeros(1, len(times)), torch.cumsum(
                (inv_k[1:]+inv_k[:-1])*(x[1]-x[0])/2, dim=0)), dim=0)
            pressure = 1-cum_r*q.T
        else:
            pressure = net.p(xt).reshape(len(x), len(times))
    xv = (torch.arange(96)+.5)/96
    tv = ((torch.arange(100)+.5)*problem.end/100)[:, None]
    rv = residuals(net, xv, tv, method, quadrature(64))
    metrics = {k+"_rms": float(v.detach().square().mean().sqrt()) for k, v in rv.items()}
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
    fields = {"x": x.numpy(), "times": times.flatten().numpy(), "c": c.numpy(),
              "sigma": sigma.numpy(), "q": q.numpy(), "j": j.numpy(),
              "pressure": pressure.numpy(), "mass_defect": balance.numpy(),
              "mobile": mobile.numpy(), "deposited": deposited.numpy(),
              "inlet_c": problem.inlet_c(times).flatten().numpy()}
    return metrics, fields


def train(args):
    torch.manual_seed(args.seed)
    problem = Problem(inlet=args.inlet)
    net = Fields(problem, args.width)
    gen = torch.Generator().manual_seed(args.seed+1000)
    rule = quadrature(args.order)

    def samples():
        x = torch.cat(((torch.arange(32)+torch.rand(32, generator=gen))/32,
                       torch.rand(8, generator=gen)*.15))
        t = torch.cat(((torch.arange(64)[:, None]+torch.rand(64, 1, generator=gen))/64,
                       torch.rand(16, 1, generator=gen)*.15,
                       torch.tensor([[0.], [1.]])))
        if problem.inlet == "dilution":
            t = torch.cat((t, .48+.14*torch.rand(16, 1, generator=gen)))
        return x, t

    start = time.perf_counter()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    x, t = samples()
    history = []
    for step in range(args.adam):
        if step % 200 == 0:
            x, t = samples()
        optimizer.zero_grad(set_to_none=True)
        loss = objective(net, x, t, args.method, rule)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0 or step == args.adam-1:
            row = {"step": step+1, "loss": float(loss.detach()), "seconds": time.perf_counter()-start}
            history.append(row)
            print(json.dumps(row), flush=True)
    optimizer = torch.optim.LBFGS(net.parameters(), max_iter=args.lbfgs,
                                  history_size=50, line_search_fn="strong_wolfe")
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = objective(net, x, t, args.method, rule)
        loss.backward()
        evaluations += 1
        return loss

    if args.lbfgs:
        optimizer.step(closure)
    metrics, fields = evaluate(net, args.method, rule)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"fields.npz", **fields)
    torch.save(net.state_dict(), args.output/"model.pt")
    config = vars(args).copy()
    config["output"] = str(args.output)
    result = {"config": config, "model": asdict(problem), "metrics": metrics,
              "seconds": time.perf_counter()-start, "lbfgs_evaluations": evaluations,
              "history": history, "runtime": {"torch": torch.__version__, "dtype": "float64", "device": "cpu"}}
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"method": args.method, "inlet": args.inlet, "seed": args.seed,
                      "metrics": metrics, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["differential", "resistance", "augmented"], default="resistance")
    parser.add_argument("--inlet", choices=["steady", "dilution"], default="steady")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--order", type=int, default=32)
    parser.add_argument("--adam", type=int, default=4000)
    parser.add_argument("--lbfgs", type=int, default=600)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    train(args)

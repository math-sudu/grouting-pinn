"""Resolve missed local Darcy residuals without adding an integral loss.

Warm-start the recorded PINN and adapt spatial samples to the time-marginal
Darcy residual. The sampling distribution follows the RAD k=1,c=1 idea of
Wu et al. (2023), https://doi.org/10.1016/j.cma.2022.115671, Section 2.3.2.
Keep uniform spatial coverage and finish on a denser fixed point set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from i060_coupled import Fields, Problem, evaluate, objective, quadrature, residuals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adam", type=int, default=2000)
    parser.add_argument("--lbfgs", type=int, default=600)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    initial = json.loads((args.input/"result.json").read_text(encoding="utf-8"))
    problem = Problem(**initial["model"])
    net = Fields(problem, initial["config"]["width"])
    net.load_state_dict(torch.load(args.input/"model.pt", weights_only=True))
    gen = torch.Generator().manual_seed(initial["config"]["seed"]+3000)
    rule = quadrature(32)
    candidates = (torch.arange(1024)+.5)/1024
    probe_t = torch.linspace(0., 1., 33)[:, None]

    def samples(dense=False):
        rv = residuals(net, candidates, probe_t, "differential", rule)["darcy"]
        score = rv.detach().reshape(len(candidates), -1).square().mean(1).sqrt()
        weights = score/score.mean()+1
        selected = torch.multinomial(weights, 128 if dense else 64,
                                     replacement=False, generator=gen)
        n = 256 if dense else 64
        uniform = (torch.arange(n)+(.5 if dense else torch.rand(n, generator=gen)))/n
        x = torch.cat((uniform, candidates[selected], torch.rand(8, generator=gen)*.15))
        t = torch.cat(((torch.arange(64)[:, None]+torch.rand(64, 1, generator=gen))/64,
                       torch.rand(16, 1, generator=gen)*.15, torch.tensor([[0.], [1.]])))
        if problem.inlet == "dilution":
            t = torch.cat((t, .48+.14*torch.rand(16, 1, generator=gen)))
        return x, t

    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    history = []
    for step in range(args.adam):
        if step % 100 == 0:
            x, t = samples()
        optimizer.zero_grad(set_to_none=True)
        loss = objective(net, x, t, "differential", rule)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == args.adam-1:
            row = {"step": step+1, "loss": float(loss.detach()), "seconds": time.perf_counter()-start}
            print(json.dumps(row), flush=True)
            history.append(row)
    x, t = samples(dense=True)
    optimizer = torch.optim.LBFGS(net.parameters(), max_iter=args.lbfgs,
                                  history_size=50, line_search_fn="strong_wolfe")
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = objective(net, x, t, "differential", rule)
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    metrics, fields = evaluate(net, "differential", rule)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"fields.npz", **fields)
    torch.save(net.state_dict(), args.output/"model.pt")
    np.savez_compressed(args.output/"final_collocation.npz", x=x.numpy(), times=t.numpy())
    result = {
        "config": {**initial["config"], "output": str(args.output),
                   "sampling": "Uniform coverage plus residual-adaptive spatial sampling",
                   "repair_adam": args.adam, "repair_lbfgs": args.lbfgs},
        "model": initial["model"], "metrics": metrics, "history": history,
        "input_ref": str(args.input), "seconds": time.perf_counter()-start,
        "lbfgs_evaluations": evaluations,
        "execution": {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "exit_code": 0},
        "scientific_role": "Diagnose and repair the local-constraint sampling failure; added computation, same physics and loss, no integral hydraulic penalty.",
    }
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()

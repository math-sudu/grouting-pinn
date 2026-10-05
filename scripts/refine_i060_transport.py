"""Continue the recorded PINNs on denser transport samples.

The complete R fields and histories supply the manuscript's physical-response
figures and table. Physics, network maps and residual weights are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from i060_coupled import Fields, Problem, evaluate, objective, quadrature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lbfgs", type=int, default=1200)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    original = json.loads((args.input / "result.json").read_text(encoding="utf-8"))
    problem = Problem(**original["model"])
    method = original["config"]["method"]
    net = Fields(problem, original["config"]["width"])
    net.load_state_dict(torch.load(args.input / "model.pt", weights_only=True))
    # Resolve transport across the column and the outlet, and the short inlet
    # transition. Keep the entire injection history in the objective.
    x = torch.unique(torch.cat((torch.linspace(0., 1., 97),
                                torch.linspace(.8, 1., 33))))
    times = [torch.linspace(0., 1., 129), torch.linspace(0., .15, 25)]
    if problem.inlet == "dilution":
        times.append(torch.linspace(.48, .62, 41))
    t = torch.unique(torch.cat(times))[:, None]
    rule = quadrature(64)
    optimizer = torch.optim.LBFGS(net.parameters(), max_iter=args.lbfgs,
                                  history_size=50, line_search_fn="strong_wolfe",
                                  tolerance_change=1e-12)
    evaluations = 0
    start = time.perf_counter()

    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = objective(net, x, t, method, rule)
        loss.backward()
        evaluations += 1
        if evaluations % 200 == 0:
            print(json.dumps({"evaluation": evaluations, "loss": float(loss.detach()),
                              "seconds": time.perf_counter()-start}), flush=True)
        return loss

    optimizer.step(closure)
    metrics, fields = evaluate(net, method, rule)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "fields.npz", **fields)
    torch.save(net.state_dict(), args.output / "model.pt")
    result = {
        "config": {**original["config"], "output": str(args.output), "order": 64,
                   "refinement_lbfgs": args.lbfgs,
                   "spatial_samples": len(x), "temporal_samples": len(t)},
        "model": original["model"], "metrics": metrics,
        "input_ref": str(args.input), "seconds": time.perf_counter()-start,
        "lbfgs_evaluations": evaluations,
        "original_metrics": original["metrics"],
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

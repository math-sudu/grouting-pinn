"""Reconstruct numerical fields from the study's saved PINN checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from i060_coupled import FEED_STATE_MAP, STATE_MAP, Fields, Problem, evaluate, quadrature


def evaluate_checkpoint(directory):
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    config = result["config"]
    problem = Problem(**result["model"])
    expected = FEED_STATE_MAP if problem.boundary == "feed_flux" else STATE_MAP
    if config["state_map"] != expected:
        raise ValueError(f"{directory}: checkpoint state map is {config['state_map']}, expected {expected}")
    net = Fields(problem, config["width"], config["capture_nodes"])
    net.load_state_dict(torch.load(directory / "model.pt", weights_only=True, map_location="cpu"))
    net.eval()
    return evaluate(net, config["method"], quadrature(config["order"]),
                    temporal_points=1001 if problem.boundary == "feed_flux" else 201)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+",
                        help="Case directories or their parent directories, searched recursively")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    checkpoints = sorted({checkpoint for directory in args.directories
                          for checkpoint in directory.rglob("model.pt")})
    if not checkpoints:
        raise FileNotFoundError("No model.pt checkpoints found in the supplied directories")
    for checkpoint in checkpoints:
        metrics, fields = evaluate_checkpoint(checkpoint.parent)
        np.savez_compressed(checkpoint.with_name("fields.npz"), **fields)
        print(json.dumps({"case": checkpoint.parent.as_posix(), "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()

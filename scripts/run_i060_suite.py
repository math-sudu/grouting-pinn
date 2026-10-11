"""Train and continue the six current manuscript cases."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def plan(config):
    tasks = []
    for case in config["cases"]:
        suffix = "" if case["retention"] == 4 else f"_retention{case['retention']:g}"
        for method in config["methods"]:
            tasks.append({**case, "method": method,
                          "name": f"{case['inlet']}{suffix}_{method}_{config['seed']}"})
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config/i060_forward.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    def run(task):
        base = ROOT/"results/i060_forward"/task["name"]
        refined = ROOT/"results/i060_transport_refinement"/task["name"]
        commands = [
            (base, [sys.executable, "-X", "utf8", "-B", "scripts/i060_coupled.py",
                    "--inlet", task["inlet"], "--method", task["method"],
                    "--retention", str(task["retention"]),
                    "--seed", str(config["seed"]), "--width", str(config["width"]),
                    "--order", str(config["order"]), "--adam", str(config["adam"]),
                    "--lbfgs", str(config["lbfgs"]),
                    "--threads", str(config["threads_per_run"]), "--output", str(base)]),
            (refined, [sys.executable, "-X", "utf8", "-B", "scripts/refine_i060_transport.py",
                       "--input", str(base), "--output", str(refined),
                       "--lbfgs", str(config["refinement_lbfgs"]),
                       "--threads", str(config["threads_per_run"])]),
        ]
        for directory, command in commands:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory/"train.log").open("w", encoding="utf-8") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return task["name"]

    with ThreadPoolExecutor(max_workers=config["parallel_runs"]) as pool:
        for future in as_completed([pool.submit(run, task) for task in plan(config)]):
            print(f"Completed {future.result()}", flush=True)


if __name__ == "__main__":
    main()

"""Execute the declared forward cases and preserve their actual run traces."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from i060_coupled import Problem

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"results/i060_forward"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan(config):
    return [
        {"name": f"{inlet}_{method}_{config['seed']}", "inlet": inlet, "method": method}
        for inlet in config["inlets"] for method in config["methods"]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/i060_forward.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    tasks = plan(config)
    with (OUTPUT/"suite.log").open("w", encoding="utf-8") as transcript:
        def announce(value):
            line = json.dumps(value)
            print(line, flush=True)
            transcript.write(line+"\n")
            transcript.flush()

        def run(task):
            directory = OUTPUT/task["name"]
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-X", "utf8", "-B", "scripts/i060_coupled.py",
                       "--inlet", task["inlet"], "--method", task["method"],
                       "--seed", str(config["seed"]), "--width", str(config["width"]),
                       "--order", str(config["order"]), "--adam", str(config["adam"]),
                       "--lbfgs", str(config["lbfgs"]), "--threads", str(config["threads_per_run"]),
                       "--output", str(directory)]
            began = datetime.now(timezone.utc).isoformat()
            with (directory/"train.log").open("w", encoding="utf-8") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            return {**task, "command": command, "started_at": began,
                    "finished_at": datetime.now(timezone.utc).isoformat(), "exit_code": 0,
                    "result_ref": (directory/"result.json").relative_to(ROOT).as_posix()}

        announce({"started_at": started, "tasks": tasks, "config": config})
        finished = []
        with ThreadPoolExecutor(max_workers=config["parallel_runs"]) as pool:
            futures = [pool.submit(run, task) for task in tasks]
            for future in as_completed(futures):
                completed = future.result()
                finished.append(completed)
                announce({"completed": completed})
        report = {
            "command": f"python -X utf8 -B scripts/run_i060_suite.py --config {args.config.as_posix()}",
            "config_hash": fingerprint(config),
            "data_fingerprint": fingerprint([asdict(Problem(inlet=x)) for x in config["inlets"]]),
            "data_role": "Specified dimensionless forward problems and seeded collocation; no experimental fit.",
            "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
            "exit_code": 0, "transcript_ref": "results/i060_forward/suite.log",
            "runs": sorted(finished, key=lambda item: item["name"]),
            "run_output_paths": ["results/i060_forward/suite.json"] + [
                f"results/i060_forward/{task['name']}/{name}"
                for task in tasks for name in ("result.json", "model.pt", "fields.npz", "train.log")],
        }
        (OUTPUT/"suite.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        announce({"finished_at": report["finished_at"], "exit_code": 0, "runs": len(finished)})


if __name__ == "__main__":
    main()

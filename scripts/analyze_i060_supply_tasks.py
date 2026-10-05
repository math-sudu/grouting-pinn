"""Compare inlet histories when a common deposited inventory is first reached."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from i060_observation_operators import interval_integral

ROOT = Path(__file__).resolve().parents[1]


def first_attainment(times, inventory, target):
    """Locate the first crossing on the saved piecewise-linear trajectory."""
    if target <= inventory[0]:
        raise ValueError("The task must require more deposits than the initial inventory")
    crossings = np.flatnonzero(inventory >= target)
    if not len(crossings):
        raise ValueError(f"Deposited target {target:g} is not reached in the saved trajectory")
    right = int(crossings[0])
    left = right - 1
    fraction = float((target - inventory[left]) / (inventory[right] - inventory[left]))
    time = float(times[left] + fraction * (times[right] - times[left]))
    return time, left, right, fraction


def evaluate_task(fields, target):
    x, times = fields["x"], fields["times"]
    time, left, right, fraction = first_attainment(times, fields["deposited"], target)
    sigma = (1 - fraction) * fields["sigma"][:, left] + fraction * fields["sigma"][:, right]
    cumulative_deposit = np.r_[0., np.cumsum(np.diff(x) * (sigma[1:] + sigma[:-1]) / 2)]
    incoming = interval_integral(times, fields["j"][0], times[0], time)
    flow = fields["q"].reshape(-1)
    return {
        "time": time,
        "deposited_inventory": float(np.trapezoid(sigma, x)),
        "cumulative_inlet_solids": incoming,
        "cumulative_outlet_solids": interval_integral(times, fields["j"][-1], times[0], time),
        "cumulative_mixture_volume": interval_integral(times, flow, times[0], time),
        "mobile_inventory": float(np.interp(time, times, fields["mobile"])),
        "flow": float(np.interp(time, times, flow)),
        "half_inventory_position": float(np.interp(cumulative_deposit[-1] / 2, cumulative_deposit, x)),
        "inlet_third_deposited_fraction": interval_integral(x, sigma, 0., 1 / 3) / cumulative_deposit[-1],
        "deposited_fraction_of_incoming": float(cumulative_deposit[-1] / incoming),
    }


def compare_histories(input_directory, target):
    comparisons = {}
    for method in ("resistance", "augmented"):
        cases = {}
        for inlet in ("steady", "dilution"):
            source = input_directory / f"{inlet}_{method}_29" / "fields.npz"
            with np.load(source) as fields:
                cases[inlet] = evaluate_task(fields, target)
            cases[inlet]["source"] = source.as_posix()
        comparisons[method] = {
            "cases": cases,
            "dilution_relative_change": {
                quantity: cases["dilution"][quantity] / cases["steady"][quantity] - 1
                for quantity in ("time", "cumulative_inlet_solids", "flow")
            },
        }
    return {
        "target_deposited_inventory": target,
        "units": "Dimensionless; solid volumes are normalized by column volume.",
        "task": "First attainment of the same total deposited inventory during injection.",
        "evaluation": "Saved fields with linear crossing/profile interpolation and boundary-flux integration up to each crossing.",
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/i060_transport_refinement")
    parser.add_argument("--target-deposited", type=float, default=0.046)
    parser.add_argument("--output", type=Path, default=ROOT / "results/i060_forward/supply_tasks.json")
    args = parser.parse_args()
    analysis = compare_histories(args.input, args.target_deposited)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    for method, comparison in analysis["comparisons"].items():
        for inlet, case in comparison["cases"].items():
            print(f"{method}/{inlet}: t={case['time']:.6f}, input={case['cumulative_inlet_solids']:.6f}, "
                  f"q={case['flow']:.6f}, x50={case['half_inventory_position']:.6f}")


if __name__ == "__main__":
    main()

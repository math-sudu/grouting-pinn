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


def regional_deposits(fields, region):
    """Deposits in a specified interval, normalized by whole-column volume."""
    return np.array([interval_integral(fields["x"], profile, *region)
                     for profile in fields["sigma"].T])


def regional_capture(fields, model, region, start, stop):
    """Integrate the local retention law over an actual treatment interval."""
    x, times = fields["x"], fields["times"]
    rate = (model["retention"]*fields["q"].reshape(1, -1)*fields["c"]
            *(1-fields["sigma"]/model["capacity"]))
    regional_rate = np.array([interval_integral(x, profile, *region)
                              for profile in rate.T])
    deposited = regional_deposits(fields, region)
    return {
        "interval": [start, stop], "region": list(region),
        "deposited_gain": float(np.interp(stop, times, deposited)
                                - np.interp(start, times, deposited)),
        "integrated_capture": interval_integral(times, regional_rate, start, stop),
    }


def evaluate_task(fields, target, region=(0., 1.)):
    x, times = fields["x"], fields["times"]
    inventory = (fields["deposited"] if tuple(region) == (0., 1.)
                 else regional_deposits(fields, region))
    time, left, right, fraction = first_attainment(times, inventory, target)
    sigma = (1 - fraction) * fields["sigma"][:, left] + fraction * fields["sigma"][:, right]
    cumulative_deposit = np.r_[0., np.cumsum(np.diff(x) * (sigma[1:] + sigma[:-1]) / 2)]
    incoming = interval_integral(times, fields["j"][0], times[0], time)
    flow = fields["q"].reshape(-1)
    deposited = float(np.trapezoid(sigma, x))
    regional = interval_integral(x, sigma, *region)
    mobile = float(np.interp(time, times, fields["mobile"]))
    outgoing = interval_integral(times, fields["j"][-1], times[0], time)
    return {
        "time": time,
        "deposited_inventory": deposited,
        "regional_deposited_inventory": regional,
        "outside_deposited_inventory": deposited-regional,
        "cumulative_inlet_solids": incoming,
        "cumulative_outlet_solids": outgoing,
        "cumulative_mixture_volume": interval_integral(times, flow, times[0], time),
        "mobile_inventory": mobile,
        "mass_balance_defect": incoming-deposited-mobile-outgoing,
        "flow": float(np.interp(time, times, flow)),
        "half_inventory_position": float(np.interp(cumulative_deposit[-1] / 2, cumulative_deposit, x)),
        "inlet_third_deposited_fraction": interval_integral(x, sigma, 0., 1 / 3) / cumulative_deposit[-1],
        "deposited_fraction_of_incoming": float(cumulative_deposit[-1] / incoming),
    }


def transport_history(fields, model, start, stop):
    """Read capture and regional transfer over the post-dilution interval."""
    x, times = fields["x"], fields["times"]
    c, sigma, flux = fields["c"], fields["sigma"], fields["j"]
    flow = fields["q"].reshape(-1)
    exposure = np.array([
        interval_integral(times, flow * profile, times[0], stop) for profile in c
    ])
    capacity = model["capacity"]
    captured = capacity * (-np.expm1(-model["retention"] * exposure / capacity))
    total = (model["phi0"] - sigma) * c + sigma
    regions = {}
    for name, left, right in (("inlet_third", 0., 1 / 3),
                              ("middle_third", 1 / 3, 2 / 3),
                              ("outlet_third", 2 / 3, 1.)):
        inventory = np.array([
            interval_integral(x, profile, left, right) for profile in total.T
        ])
        boundary_difference = np.array([
            np.interp(left, x, profile) - np.interp(right, x, profile)
            for profile in flux.T
        ])
        regions[name] = {
            "total_inventory_change": float(
                np.interp(stop, times, inventory) - np.interp(start, times, inventory)),
            "integrated_net_transfer": interval_integral(
                times, boundary_difference, start, stop),
        }
    return {
        "interval": [start, stop],
        "deposited_gain": float(np.interp(stop, times, fields["deposited"])
                                - np.interp(start, times, fields["deposited"])),
        "mobile_change": float(np.interp(stop, times, fields["mobile"])
                               - np.interp(start, times, fields["mobile"])),
        "net_inlet_solids": interval_integral(times, flux[0], start, stop),
        "outlet_solids": interval_integral(times, flux[-1], start, stop),
        "deposited_at_stop": float(np.interp(stop, times, fields["deposited"])),
        "capture_history_deposited_at_stop": float(np.trapezoid(captured, x)),
        "regions": regions,
    }


def compare_histories(input_directory, target):
    comparisons = {}
    for method in ("resistance", "augmented"):
        cases = {}
        for inlet in ("steady", "dilution"):
            source = input_directory / f"{inlet}_{method}_29" / "fields.npz"
            with np.load(source) as fields:
                cases[inlet] = evaluate_task(fields, target)
                if inlet == "dilution":
                    model = json.loads(source.with_name("result.json").read_text(
                        encoding="utf-8"))["model"]
                    peak_time = float(fields["times"][np.argmax(fields["j"][-1])])
                    cases[inlet]["post_dilution_history"] = {
                        "to_outlet_peak": transport_history(fields, model, .55, peak_time),
                        "to_attainment": transport_history(
                            fields, model, .55, cases[inlet]["time"]),
                    }
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
        "input_definition": "Signed net solid volume crossing the injection face, including advection and dispersion.",
        "history_evaluation": "Integrate q*c from the initially particle-free state using the retention law; compare regional inventory changes with integrated boundary transfers after the dilution midpoint.",
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

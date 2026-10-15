"""Read stage accounts and common-deposit events from the saved feed histories."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_i060_supply_response import ROOT, load_case, state_at


NAMES = ("mixed", "dilution", "rich", "water", "dilute", "very_dilute")


def first_attainment(case, target):
    deposits = case["deposit_g"][2:].sum(axis=0)
    times = case["fields"]["times"]
    crossings = np.flatnonzero(deposits >= target)
    if not len(crossings):
        return {"reached_target": False, "state": None}
    index = int(crossings[0])
    time = float(times[index]) if index == 0 else float(np.interp(
        target, deposits[index - 1:index + 1], times[index - 1:index + 1]))
    return {"reached_target": True, "time_dimensionless": time,
            "state": state_at(case, time)}


def main():
    torch.set_num_threads(2)
    cases = {name: load_case(name) for name in NAMES}
    # Use the same array reduction for the target and its crossing comparison.
    target = float(cases["mixed"]["deposit_g"][2:].sum(axis=0)[-1])
    blend_start = (cases["dilution"]["problem"].switch_time
                   - cases["dilution"]["problem"].switch_width / 2)
    results = {}
    for name, case in cases.items():
        results[name] = {
            "source": case["source"],
            "at_end": state_at(case, case["fields"]["times"][-1]),
            "at_target": first_attainment(case, target),
        }
        if name in ("mixed", "dilution"):
            before = state_at(case, blend_start)
            after = {
                key: results[name]["at_end"][key] - before[key]
                for key in ("cement_kg", "water_kg", "outlet_g")
            }
            results[name]["stage_accounts"] = {
                "at_blend_start": before, "after_blend_inputs_and_outlet": after}

    origin_root = ROOT / "results/i060_origins"
    origins = {}
    for name in ("mixed", "dilution"):
        source = origin_root / name / "result.json"
        record = json.loads(source.read_text(encoding="utf-8"))["result"]
        available = record["parent_initial_mobile_g"]
        captured = record["origins"]["old"]["downstream_captured_g"]
        origins[name] = {"source": source.relative_to(ROOT).as_posix(),
                         "available_g": available, "captured_g": captured,
                         "capture_fraction": captured / available}
    b, s = origins["mixed"], origins["dilution"]
    inventory_term = ((b["capture_fraction"] + s["capture_fraction"]) / 2
                      * (s["available_g"] - b["available_g"]))
    fraction_term = ((b["available_g"] + s["available_g"]) / 2
                     * (s["capture_fraction"] - b["capture_fraction"]))
    baseline = results["mixed"]["at_target"]["state"]
    staged = results["dilution"]["at_target"]["state"]
    reductions = {key: 100 * (1 - staged[key] / baseline[key])
                  for key in ("time_min", "cement_kg", "water_kg", "outlet_g")}
    before_water = (results["mixed"]["stage_accounts"]["at_blend_start"]["water_kg"]
                    - results["dilution"]["stage_accounts"]["at_blend_start"]["water_kg"])
    after_water = (
        results["dilution"]["stage_accounts"]["after_blend_inputs_and_outlet"]["water_kg"]
        - results["mixed"]["stage_accounts"]["after_blend_inputs_and_outlet"]["water_kg"])
    output = {
        "target": {"baseline": "mixed", "region": "segments 3-6; x in [1/3, 1]",
                   "downstream_deposit_g": target,
                   "definition": "Final downstream deposit of the saved constant-mixture history",
                   "sampling": "First crossing with linear interpolation on the original time grid"},
        "blend_start_dimensionless": blend_start,
        "cases": results,
        "main_comparison": {"at_target_reduction_percent": reductions,
                            "before_blend_water_saving_kg": before_water,
                            "after_blend_extra_water_kg": after_water},
        "old_particle_decomposition": {
            "formula": "delta(A*p) = mean(p)*delta(A) + mean(A)*delta(p)",
            "cases": origins, "inventory_term_g": inventory_term,
            "fraction_term_g": fraction_term,
            "total_difference_g": s["captured_g"] - b["captured_g"]},
    }
    path = ROOT / "results/i060_forward/staging_analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target_deposit_g": target, "main_comparison": output["main_comparison"],
                      "old_inventory_term_g": inventory_term,
                      "old_fraction_term_g": fraction_term}, indent=2))


if __name__ == "__main__":
    main()


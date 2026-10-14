"""Close the supply-to-column material account for solved feed histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_i060_segment_tasks import evaluate_segment_task
from i060_coupled import Problem
from i060_observation_operators import interval_integral


ROOT = Path(__file__).resolve().parents[1]


def prepared_material(fields, problem, supply, time):
    """Count drawn stocks, fresh preparation, water and unused wet material.

    Stocks are prepared for this plan before the common wetting clock starts.
    Any configured reserve is charged to preparation, with no recovery credit.
    All volumes here are dimensional, unlike the underlying particle fields.
    """
    if problem.boundary != "feed_flux":
        raise ValueError("Fresh supply accounting requires a feed-flux boundary.")
    material = problem.material
    column = supply["column"]
    volume = np.pi*column["diameter_m"]**2/4*material.length_m
    times, flow = fields["times"], fields["q"].reshape(-1)
    drawn = np.array([
        interval_integral(times, flow*weight, times[0], time)*volume
        for weight in fields["feed_weights"].T
    ])
    concentrations = np.array([problem.cin, problem.cin*problem.dilution_fraction, 0.])
    feed_solid = float(drawn@concentrations)
    reserves = np.asarray(supply["feed"]["unused_stock_reserve_m3"], dtype=float)
    if reserves.shape != (2,) or np.any(reserves < 0):
        raise ValueError("Specify two nonnegative unused stock reserves.")
    # A reservoir is loaded only if that stock is used by the selected plan.
    unused = np.where(drawn[:2] > 0, reserves, 0.)
    prepared = drawn[:2]+unused
    unused_solid = float(unused@concentrations[:2])
    fresh_solid = feed_solid+unused_solid
    feed_water = float(drawn.sum()-feed_solid)
    unused_water = float(unused.sum()-unused_solid)
    return {
        "time_min": time*material.reference_time_s/60,
        "mixture_pumped_litre": float(drawn.sum()*1000),
        "drawn_stock_litre": (drawn[:2]*1000).tolist(),
        "separate_water_litre": float(drawn[2]*1000),
        "prepared_stock_litre": (prepared*1000).tolist(),
        "unused_stock_litre": (unused*1000).tolist(),
        "fed_cement_kg": feed_solid*material.cement_density_kg_m3,
        "fresh_cement_kg": fresh_solid*material.cement_density_kg_m3,
        "unused_cement_kg": unused_solid*material.cement_density_kg_m3,
        "fed_water_kg": feed_water*material.water_density_kg_m3,
        "fresh_water_kg": (feed_water+unused_water)*material.water_density_kg_m3,
        "stock_water_kg": float((prepared*(1-concentrations[:2])).sum()*material.water_density_kg_m3),
        "separate_water_kg": float(drawn[2]*material.water_density_kg_m3),
        "unused_water_kg": unused_water*material.water_density_kg_m3,
    }


def material_account(fields, problem, supply, stop, edges):
    """Add column and outlet destinations to preparation at the same stop."""
    time = stop["time"]
    account = prepared_material(fields, problem, supply, time)
    material = problem.material
    volume = np.pi*supply["column"]["diameter_m"]**2/4*material.length_m
    times, flow = fields["times"], fields["q"].reshape(-1)
    x, sigma = fields["x"], fields["sigma"]
    partition = stop["material_partition"]
    outlet = partition["outlet_solid_volume"]*volume
    segment_total = np.asarray(stop["segment_total"])*volume
    capture = (problem.retention*flow[None, :]*fields["c"]
               *(1-sigma/problem.capacity))
    captured = []
    initial_deposited = []
    for left, right in zip(edges[:-1], edges[1:]):
        rates = np.array([interval_integral(x, profile, left, right)
                          for profile in capture.T])
        captured.append(interval_integral(times, rates, times[0], time))
        initial_deposited.append(interval_integral(x, sigma[:, 0], left, right))
    capture_defect = (np.asarray(initial_deposited)+captured-np.asarray(stop["segment_deposited"]))
    return {**account,
        "outlet_cement_kg": outlet*material.cement_density_kg_m3,
        "outlet_water_kg": (account["mixture_pumped_litre"]/1000-outlet)*material.water_density_kg_m3,
        "initial_column_water_kg": volume*problem.phi0*material.water_density_kg_m3,
        "column_water_at_stop_kg": (volume*problem.phi0-segment_total.sum())*material.water_density_kg_m3,
        "segment_cement_kg": (segment_total*material.cement_density_kg_m3).tolist(),
        "segment_deposited_cement_kg": (np.asarray(stop["segment_deposited"])*volume*material.cement_density_kg_m3).tolist(),
        "segment_mobile_cement_kg": (np.asarray(stop["segment_mobile"])*volume*material.cement_density_kg_m3).tolist(),
        "segment_balance_defect_kg": (np.asarray(stop["segment_balance_defect"])*volume*material.cement_density_kg_m3).tolist(),
        "segment_capture_defect_kg": (capture_defect*volume*material.cement_density_kg_m3).tolist(),
        "feed_to_face_defect_kg": account["fed_cement_kg"]-partition["net_inlet_solid_volume"]*volume*material.cement_density_kg_m3,
        "cement_balance_defect_kg": account["fed_cement_kg"]-(outlet+segment_total.sum())*material.cement_density_kg_m3,
    }


def compare_cement_accounts(reference, selected, target_segments):
    """Partition prepared-cement savings at each plan's own task endpoint.

    Reductions are reference minus selected. The numerical balance defect is
    kept separate from physical destinations and unused prepared stock.
    """
    targets = np.asarray(target_segments, dtype=int)-1

    def destinations(state):
        account = state["supply"]
        deposited = np.asarray(account["segment_deposited_cement_kg"])
        mobile = np.asarray(account["segment_mobile_cement_kg"])
        target = float(deposited[targets].sum())
        target_mobile = float(mobile[targets].sum())
        return {
            "target_deposited": target,
            "outside_deposited": float(deposited.sum())-target,
            "target_mobile": target_mobile,
            "outside_mobile": float(mobile.sum())-target_mobile,
            "outlet": account["outlet_cement_kg"],
            "unused_preparation": account["unused_cement_kg"],
        }

    baseline, chosen = destinations(reference), destinations(selected)
    reduction = reference["supply"]["fresh_cement_kg"]-selected["supply"]["fresh_cement_kg"]
    return {
        "reference_time": reference["time"],
        "selected_time": selected["time"],
        "prepared_cement_reduction_kg": reduction,
        "prepared_cement_reduction_percent": 100*reduction/reference["supply"]["fresh_cement_kg"],
        "destinations": {
            name: {"reference_kg": value, "selected_kg": chosen[name],
                   "reduction_kg": value-chosen[name]}
            for name, value in baseline.items()
        },
        "balance_defect_reduction_kg": (
            reference["supply"]["cement_balance_defect_kg"]
            -selected["supply"]["cement_balance_defect_kg"]),
    }


def analyze_run(path, supply, task, *, stop_start=None):
    result = json.loads((path/"result.json").read_text(encoding="utf-8"))
    problem = Problem(**result["model"])
    with np.load(path/"fields.npz") as fields:
        case = evaluate_segment_task(fields, problem.phi0, task, stop_start=stop_start)
        for key in ("first_stop", "deadline_state"):
            if case[key] is not None:
                case[key]["supply"] = material_account(fields, problem, supply, case[key], task["segment_edges"])
    case["source"] = path.as_posix()
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"results/i060_feed")
    parser.add_argument("--config", type=Path, default=ROOT/"config/i060_material_delivery.json")
    args = parser.parse_args()
    supply = json.loads(args.config.read_text(encoding="utf-8"))
    task = json.loads((ROOT/supply["delivery_task"]).read_text(encoding="utf-8"))
    cases = {path.parent.name: analyze_run(path.parent, supply, task)
             for path in sorted(args.input.glob("*/result.json"))}
    if not cases:
        raise FileNotFoundError(f"No solved feed histories under {args.input}")
    output = {"supply": supply, "task": task, "cases": cases}
    (args.input/"delivery.json").write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    for name, case in cases.items():
        state = case["first_stop"] or case["deadline_state"]
        account = state["supply"]
        print(json.dumps({"case": name, "feasible": case["first_stop"] is not None,
                          "time_min": account["time_min"],
                          "fresh_cement_kg": account["fresh_cement_kg"],
                          "fresh_water_kg": account["fresh_water_kg"],
                          "controlling_segments": case["controlling_segments"],
                          "cement_balance_defect_kg": account["cement_balance_defect_kg"]}))


if __name__ == "__main__":
    main()

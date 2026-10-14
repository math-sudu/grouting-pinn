"""Read physical supply responses over the saved injection histories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from analyze_i060_mechanism import resistance_budget
from analyze_i060_segment_tasks import segment_histories
from i060_coupled import Problem

ROOT = Path(__file__).resolve().parents[1]
EDGES = np.linspace(0., 1., 7)
CASES = {
    "rich": "results/i060_feed/steady_resistance",
    "dilution": "results/i060_feed/dilution_resistance",
    "water": "results/i060_feed/water_resistance",
    "dilution_di": "results/i060_feed/dilution_augmented",
    "mixed": "results/i060_schedule/constant_0p5_0p5_0",
    "dilute": "results/i060_schedule/constant_0_1_0",
    "very_dilute": "results/i060_schedule/constant_0_0p515625_0p484375",
}


def cumulative(times, rate):
    return np.r_[0., np.cumsum(np.diff(times)*(rate[1:]+rate[:-1])/2)]


def load_case(name):
    source = ROOT/CASES[name]
    result = json.loads((source/"result.json").read_text(encoding="utf-8"))
    problem = Problem(**result["model"])
    with np.load(source/"fields.npz") as stored:
        fields = dict(stored)
    geometry = json.loads((ROOT/"config/i060_material_delivery.json").read_text(
        encoding="utf-8"))["column"]
    area = np.pi*geometry["diameter_m"]**2/4
    volume = area*problem.material.length_m
    mass_g = problem.material.cement_density_kg_m3*volume*1000
    times = fields["times"]
    q = fields["q"].reshape(-1)
    segments = segment_histories(fields, problem.phi0, EDGES)
    return {
        "source": CASES[name], "problem": problem, "fields": fields,
        "time_min": times*problem.material.reference_time_s/60,
        "flow_lpm": q*area*problem.material.reference_flux_m_s*60000,
        "deposit_g": segments["deposited"]*mass_g,
        "mobile_g": segments["mobile"]*mass_g,
        "cement_kg": cumulative(times, q*fields["feed_c"])*mass_g/1000,
        "water_kg": cumulative(times, q*(1-fields["feed_c"]))
                    *volume*problem.material.water_density_kg_m3,
        "outlet_g": cumulative(times, fields["j"][-1])*mass_g,
        "mass_scale_g": mass_g,
    }


def state_at(case, time):
    times = case["fields"]["times"]
    sample = lambda values: float(np.interp(time, times, values))
    deposited = [sample(row) for row in case["deposit_g"]]
    mobile = [sample(row) for row in case["mobile_g"]]
    return {
        "time_min": sample(case["time_min"]),
        "cement_kg": sample(case["cement_kg"]),
        "water_kg": sample(case["water_kg"]),
        "outlet_g": sample(case["outlet_g"]),
        "flow_lpm": sample(case["flow_lpm"]),
        "segment_deposited_g": deposited,
        "segment_mobile_g": mobile,
        "downstream_deposited_g": sum(deposited[2:]),
        "downstream_mobile_g": sum(mobile[2:]),
        "downstream_total_g": sum(deposited[2:])+sum(mobile[2:]),
        "column_deposited_g": sum(deposited),
        "column_mobile_g": sum(mobile),
        "upstream_deposit_fraction": sum(deposited[:2])/sum(deposited),
    }


def main():
    torch.set_num_threads(2)
    result = {}
    for name in CASES:
        case = load_case(name)
        end = case["fields"]["times"][-1]
        result[name] = {"source": case["source"], "at_end": state_at(case, end)}
        if name in ("dilution", "mixed", "water"):
            problem, fields = case["problem"], case["fields"]
            budget = resistance_budget(fields["x"], fields["c"], fields["sigma"],
                                       fields["times"], problem)
            result[name]["at_end"]["resistance"] = {
                key: float(values[-1]) for key, values in budget.items()}
        if name == "water":
            for label, time in (("switch_midpoint", problem.switch_time),
                                ("pure_water_start", problem.switch_time+problem.switch_width/2),
                                ("end", end)):
                state = state_at(case, time)
                state["resistance"] = {key: float(np.interp(time, fields["times"], values))
                                       for key, values in budget.items()}
                result[name][label] = state
        print(name, json.dumps(result[name]["at_end"]))
    output = ROOT/"results/i060_forward/supply_response.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()

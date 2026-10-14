"""Resolve the selected material plan's resistance and spatial particle states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from i060_coupled import Problem


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"paper/selfrev-r14/evidence/SR-05_material_mechanism.json"


def resistance_budget(x, c, sigma, times, problem):
    """Decompose m/k on the same grid, retaining the local gravity head.

    Inputs c and sigma have shape (position, time). The slurry term includes
    its permeability weighting; it is not an independent no-deposition run.
    """
    material = problem.material
    concentration = torch.as_tensor(c, dtype=torch.float64)
    time = torch.as_tensor(times, dtype=torch.float64)[None, :].expand_as(concentration)
    with torch.no_grad():
        viscosity = material.relative_viscosity(concentration, time).numpy()
        gravity = material.gravity(concentration).numpy()
    inverse_k = (1-np.asarray(sigma)/problem.phi0)**-3
    water = material.water_viscosity_pa_s/material.reference_viscosity_pa_s
    base = np.full(len(times), water*(x[-1]-x[0]))
    deposited = water*np.trapezoid(inverse_k-1, x, axis=0)
    slurry = np.trapezoid((viscosity-water)*inverse_k, x, axis=0)
    total = base+deposited+slurry
    head = np.trapezoid(gravity, x, axis=0)
    return {"water_base": base, "deposit_water": deposited,
            "slurry_excess": slurry, "total": total, "gravity_head": head,
            "reconstructed_flow": (1-head)/total}


def sample_profiles(fields, times):
    """Use the planning trajectory's linear interpolation at exact events."""
    source_times = fields["times"]
    return {key: np.array([np.interp(times, source_times, row) for row in fields[key]])
            for key in ("c", "sigma")}


def analyze_case(case, supply, events):
    source = Path(case["source"])
    model = json.loads((source/"result.json").read_text(encoding="utf-8"))["model"]
    problem = Problem(**model)
    stop = case["first_stop"]["time"]
    minutes = problem.material.reference_time_s/60
    area = np.pi*supply["column"]["diameter_m"]**2/4
    flow_scale = area*problem.material.reference_flux_m_s*60000
    with np.load(source/"fields.npz") as fields:
        x = fields["x"]
        times = np.unique(np.r_[fields["times"][fields["times"] < stop],
                                list(events.values()), stop])
        profiles = sample_profiles(fields, times)
        budget = resistance_budget(x, profiles["c"], profiles["sigma"], times, problem)
        saved_flow = np.interp(times, fields["times"], fields["q"].reshape(-1))
        history = {"time_min": times*minutes, **budget,
                   "flow_litre_min": saved_flow*flow_scale}
        event_states = {}
        for name, time in events.items():
            index = np.flatnonzero(times == time).item()
            c, sigma = profiles["c"][:, index], profiles["sigma"][:, index]
            event_states[name] = {
                "time_min": time*minutes,
                "resistance": {key: float(values[index]) for key, values in budget.items()},
                "flow_litre_min": float(saved_flow[index]*flow_scale),
                "c": c.tolist(), "sigma": sigma.tolist(),
                "concentration_peak_position": float(x[c.argmax()]),
            }
    return {
        "source": source.as_posix(), "stop_time_min": stop*minutes,
        "history": {key: values.tolist() for key, values in history.items()},
        "x": x.tolist(), "events": event_states,
        "grid_flow_difference_litre_min": float(np.max(np.abs(
            budget["reconstructed_flow"]-saved_flow))*flow_scale),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT/"results/i060_schedule/selection.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected = selection["cases"][selection["selected"]]
    reference = selection["cases"]["steady_resistance"]
    switch, width = selected["switch_time"], selected["switch_width"]
    events = {"blend_start": switch-width/2, "switch_midpoint": switch,
              "pure_water_start": switch+width/2,
              "selected_closure": selected["first_stop"]["time"]}
    result = {
        "selection": args.selection.as_posix(), "selected": selection["selected"],
        "reference": "steady_resistance", "target_segments": selection["task"]["target_segments"],
        "segment_edges": selection["task"]["segment_edges"],
        "method": "Interpolate the saved concentration and deposits to each event, "
                  "evaluate the same local material law, and integrate on the 301-point spatial grid. "
                  "H=H_water+H_deposit+H_slurry, q=(1-B)/H. Flow curves retain saved PINN flow. "
                  "The slurry term includes the current permeability weighting.",
        "selected_case": analyze_case(selected, selection["supply"], events),
        "reference_case": analyze_case(reference, selection["supply"], {
            **events, "reference_closure": reference["first_stop"]["time"]}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for name in ("selected_case", "reference_case"):
        case = result[name]
        print(name, {event: {"time_min": state["time_min"],
                            "resistance": state["resistance"],
                            "flow_litre_min": state["flow_litre_min"]}
                     for event, state in case["events"].items()})


if __name__ == "__main__":
    main()

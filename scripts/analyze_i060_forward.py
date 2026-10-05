"""Read coupled forward responses on their physical and observation supports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from i060_observation_operators import collection_density, segment_inventory

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"results/i060_forward"


def cumulative(coordinate, values):
    return np.r_[0., np.cumsum(np.diff(coordinate)*(values[1:]+values[:-1])/2)]


def experimental_responses(source):
    with (source / "collection_observations.csv").open(encoding="utf-8", newline="") as stream:
        collections = list(csv.DictReader(stream))
    with (source / "segment_observations.csv").open(encoding="utf-8", newline="") as stream:
        sections = list(csv.DictReader(stream))
    responses = {}
    for case in ("S6", "S14"):
        samples = sorted((row for row in collections if row["combination"] == case),
                         key=lambda row: float(row["outlet_time_s"]))
        q = np.array([float(row["volume_ml"])/float(row["collection_duration_s"])
                      for row in samples])
        c = np.array([(float(row["density_g_cm3"])-1.)/2.1 for row in samples])
        j = q*c
        segments = sorted((row for row in sections if row["combination"] == case),
                          key=lambda row: int(row["segment_number"]))
        inventory = np.array([float(row["initial_pore_volume_cm3"])/
                              (1.+3.1*float(row["water_cement_ratio"])) for row in segments])
        responses[case] = {
            "collection_count": len(samples),
            "first_last_mixture_rate_ml_s": [float(q[0]), float(q[-1])],
            "first_last_solid_rate_ml_s": [float(j[0]), float(j[-1])],
            "last_first_mixture_ratio": float(q[-1]/q[0]),
            "last_first_concentration_ratio": float(c[-1]/c[0]),
            "last_first_solid_rate_ratio": float(j[-1]/j[0]),
            "section_total_solid_volumes_ml": inventory.tolist(),
            "inlet_third_total_inventory_fraction": float(inventory[:2].sum()/inventory.sum()),
        }
    return responses


def read_run(path):
    result = json.loads((path/"result.json").read_text(encoding="utf-8"))
    with np.load(path/"fields.npz") as source:
        f = {key: source[key] for key in source.files}
    x, t = f["x"], f["times"]
    c, sigma, q, j = f["c"], f["sigma"], f["q"].reshape(-1), f["j"]
    phi0 = result["model"]["phi0"]
    solid = f["mobile"]+f["deposited"]
    deposit_cumulative = cumulative(x, sigma[:, -1])
    bounds = np.linspace(0., 1., 7)
    segments = [dict(x_start=float(a), x_stop=float(b), **segment_inventory(
        x, c[:, -1], sigma[:, -1], phi0, a, b))
        for a, b in zip(bounds[:-1], bounds[1:])]
    windows = [dict(start=a, stop=b, density_g_cm3=collection_density(t, q, j[-1], a, b))
               for a, b in ((.2, .25), (.5, .55), (.7, .75), (.95, 1.))]
    post = t >= .65
    pressure_steps = np.diff(f["pressure"], axis=0)
    peak = np.unravel_index(np.abs(pressure_steps).argmax(), pressure_steps.shape)
    analysis = {
        "metrics": result["metrics"],
        "seconds": result["seconds"],
        "lbfgs_evaluations": result["lbfgs_evaluations"],
        "final_solid_inventory": float(solid[-1]),
        "initial_pore_volume": phi0,
        "final_water_inventory": float(phi0-solid[-1]),
        "final_mass_ratio_to_initial_water": float((phi0+2.1*solid[-1])/phi0),
        "mass_ratio_convention": "rho_s/rho_w=3.1; illustrative saturated given-parameter column",
        "inlet_solid_volume": float(cumulative(t, j[0])[-1]),
        "outlet_solid_volume": float(cumulative(t, j[-1])[-1]),
        "outlet_mixture_volume": float(cumulative(t, q)[-1]),
        "outlet_water_volume": float(cumulative(t, q-j[-1])[-1]),
        "deposit_half_inventory_position": float(np.interp(deposit_cumulative[-1]/2, deposit_cumulative, x)),
        "inlet_fifth_pressure_drop_at_T": float(1-np.interp(.2, x, f["pressure"][:, -1])),
        "minimum_saved_deposition_increment": float(np.diff(sigma, axis=1).min()),
        "maximum_saved_flow_increment": float(np.diff(q).max()),
        "maximum_late_interior_above_current_inlet": float((c[1:, post]-f["inlet_c"][post]).max()),
        "largest_saved_pressure_step": {
            "x_start": float(x[peak[0]]), "x_stop": float(x[peak[0]+1]),
            "time": float(t[peak[1]]), "pressure_change": float(pressure_steps[peak]),
        },
        "segments": segments,
        "collection_windows": windows,
        "observation_scope": "Own dimensionless coordinates, illustrative density convention; no mapping to experimental time or fitted Zhang properties.",
    }
    return analysis, f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimental-data", type=Path,
                        help="Include collection and segment summaries from this CSV directory")
    args = parser.parse_args()
    suite = json.loads((OUTPUT/"suite.json").read_text(encoding="utf-8"))
    rows, fields = {}, {}
    for run in suite["runs"]:
        name = run["name"]
        rows[name], fields[name] = read_run(OUTPUT/name)
    for path in sorted(OUTPUT.glob("*_repaired_*/result.json")):
        rows[path.parent.name], fields[path.parent.name] = read_run(path.parent)
    differences = {}
    for inlet in ("steady", "dilution"):
        names = {method: next(name for name in rows if name.startswith(f"{inlet}_{method}_"))
                 for method in ("differential", "resistance", "augmented")}
        ref = fields[names["resistance"]]
        differences[inlet] = {}
        methods = ["differential", "augmented"]
        repair = next((name for name in rows if name.startswith(f"{inlet}_differential_repaired_")), None)
        if repair is not None:
            names["differential_repaired"] = repair
            methods.append("differential_repaired")
        for method in methods:
            other = fields[names[method]]
            differences[inlet][method] = {
                "max_flow_difference_from_R": float(np.abs(other["q"]-ref["q"]).max()),
                "max_concentration_difference_from_R": float(np.abs(other["c"]-ref["c"]).max()),
                "max_deposition_difference_from_R": float(np.abs(other["sigma"]-ref["sigma"]).max()),
                "final_outlet_c_difference_from_R": float(other["c"][-1, -1]-ref["c"][-1, -1]),
                "interpretation": "Disagreement between PINN formulations; not independent reference-solution error.",
            }
    refinement = {}
    for inlet in ("steady", "dilution"):
        refinement[inlet] = {}
        for method in ("resistance", "augmented"):
            path = ROOT / "results/i060_transport_refinement" / f"{inlet}_{method}_29"
            refinement[inlet][method], _ = read_run(path)
    analysis = {"runs": rows, "paired_differences": differences,
                "transport_refinement": refinement}
    if args.experimental_data is not None:
        analysis["experimental_responses"] = experimental_responses(args.experimental_data)
    (OUTPUT/"analysis.json").write_text(json.dumps(analysis, indent=2)+"\n", encoding="utf-8")
    with (OUTPUT/"curves.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "time", "flow", "inlet_concentration", "outlet_concentration",
                         "mobile_solid", "deposited_solid", "water_inventory", "mass_ratio",
                         "outlet_mixture_volume", "outlet_solid_volume", "mass_defect"))
        for name, f in fields.items():
            stock = f["mobile"]+f["deposited"]
            pore = rows[name]["initial_pore_volume"]
            for values in zip(f["times"], f["q"].reshape(-1), f["inlet_c"], f["c"][-1],
                              f["mobile"], f["deposited"], pore-stock, 1+2.1*stock/pore,
                              cumulative(f["times"], f["q"].reshape(-1)),
                              cumulative(f["times"], f["j"][-1]), f["mass_defect"]):
                writer.writerow((name, *values))
    with (OUTPUT/"profiles.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "time", "x", "concentration", "deposition", "pressure"))
        for name, f in fields.items():
            for time in (.25, .55, .7, 1.):
                index = int(np.argmin(abs(f["times"]-time)))
                for values in zip(f["x"], f["c"][:, index], f["sigma"][:, index], f["pressure"][:, index]):
                    writer.writerow((name, f["times"][index], *values))
    for name, row in rows.items():
        keys = ("q_final", "deposited_final", "outlet_c_final", "max_mass_defect", "max_pressure_flow_defect")
        print(name, {key: round(row["metrics"][key], 7) for key in keys})
    print(json.dumps(differences, indent=2))


if __name__ == "__main__":
    main()

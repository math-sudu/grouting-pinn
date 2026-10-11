"""Compare supply histories over the fixed treatment interval [1/3, 1]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_i060_supply_tasks import evaluate_task, regional_capture, regional_deposits


ROOT = Path(__file__).resolve().parents[1]
REGION = (1/3, 1.)
METHODS = ("resistance", "augmented")
INLETS = ("steady", "dilution")
COMPONENTS = ("outside_deposited_inventory", "mobile_inventory",
              "cumulative_outlet_solids", "mass_balance_defect")


def analyze_treatment(directory):
    fields, inventories = {}, {}
    for method in METHODS:
        for inlet in INLETS:
            name = f"{inlet}_{method}_29"
            with np.load(directory/name/"fields.npz") as source:
                fields[name] = {key: source[key] for key in source.files}
            inventories[name] = regional_deposits(fields[name], REGION)
    reference_name = "steady_resistance_29"
    reference = evaluate_task(fields[reference_name], .046)
    target = float(np.interp(reference["time"], fields[reference_name]["times"],
                             inventories[reference_name]))
    maximums = {name: float(inventory.max()) for name, inventory in inventories.items()}
    # Use the saved inventory knots, including the very small early targets,
    # every terminal target and the reference. No extrapolation is needed.
    targets = np.unique(np.concatenate(
        [inventory[inventory > 0] for inventory in inventories.values()]+[np.array([target])]))
    cases = {}
    for name, data in fields.items():
        cases[name] = {
            "source": (directory/name/"fields.npz").as_posix(),
            "time_interval": [float(data["times"][0]), float(data["times"][-1])],
            "terminal_treatment_inventory": float(inventories[name][-1]),
            "maximum_treatment_inventory": maximums[name],
            "minimum_saved_inventory_increment": float(np.diff(inventories[name]).min()),
            "at_reference_target": evaluate_task(data, target, REGION),
            "target_curve": [evaluate_task(data, float(value), REGION)
                             if value <= maximums[name] else None for value in targets],
        }
        if name.startswith("dilution_"):
            model = json.loads((directory/name/"result.json").read_text(encoding="utf-8"))["model"]
            cases[name]["post_dilution_capture"] = regional_capture(
                data, model, REGION, .55, cases[name]["at_reference_target"]["time"])
    comparisons = {}
    for method in METHODS:
        steady = cases[f"steady_{method}_29"]["at_reference_target"]
        dilution = cases[f"dilution_{method}_29"]["at_reference_target"]
        decomposition = {key: steady[key]-dilution[key] for key in COMPONENTS}
        comparisons[method] = {
            "dilution_relative_change": {key: dilution[key]/steady[key]-1
                                         for key in ("time", "cumulative_inlet_solids", "flow")},
            "steady_minus_dilution": {
                "net_solid_input": steady["cumulative_inlet_solids"]-dilution["cumulative_inlet_solids"],
                **decomposition,
            },
            "reachable_by_both": min(maximums[f"{inlet}_{method}_29"] for inlet in INLETS),
            "reachable_by_either": max(maximums[f"{inlet}_{method}_29"] for inlet in INLETS),
        }
    return {
        "region": list(REGION),
        "normalization": "Solid inventories are normalized by whole-column volume, not treatment-region volume.",
        "reference": {"case": reference_name, "whole_column_target": .046,
                      "time": reference["time"], "treatment_target": target},
        "evaluation": "First crossing of each treatment inventory; saved piecewise-linear states and partial-interval flux integrals.",
        "balance": "I_in = S_treatment + S_outside + S_mobile + O + epsilon; epsilon is the numerical balance defect.",
        "targets": targets.tolist(), "cases": cases, "comparisons": comparisons,
    }


def plot_treatment(analysis, output, preview):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(ROOT.parents[1]))
    from tools import origin_figures as fs

    fig = fs.figure(137.1, 142)
    axes = np.array([[fig.add_axes((.14, .64, .34, .25)),
                      fig.add_axes((.63, .64, .34, .25))],
                     [fig.add_axes((.14, .24, .34, .25)),
                      fig.add_axes((.63, .24, .34, .25))]])
    for ax in axes.flat:
        fs.style_axes(ax)
        ax.tick_params(labelsize=7.5)
    quantities = ("time", "cumulative_inlet_solids", "flow")
    labels = ("First attainment time", "Net solid input", "Remaining flow")
    targets = np.array(analysis["targets"])
    reference = analysis["reference"]["treatment_target"]
    for ax, quantity, label in zip(axes.flat, quantities, labels):
        for method, method_label, line, marker in (
                ("resistance", "R", "-", "o"), ("augmented", "D+I", "--", "s")):
            for inlet, inlet_label, color in (("steady", "Sustained", fs.BLUE),
                                             ("dilution", "Dilution", fs.RED)):
                case = analysis["cases"][f"{inlet}_{method}_29"]
                values = [np.nan if row is None else row[quantity] for row in case["target_curve"]]
                ax.plot(targets, values, color=color, ls=line, lw=1.2,
                        label=f"{method_label}: {inlet_label}")
                ax.plot(reference, case["at_reference_target"][quantity],
                        color=color, marker=marker, mfc="white", ms=4)
        ax.axvline(reference, color=".6", ls=":", lw=.7)
        ax.set_xlabel(r"Treatment target, $S_T^*$", fontsize=8, labelpad=3)
        ax.set_ylabel(label, fontsize=8, labelpad=3)
        ax.set_xlim(0, max(targets)*1.02)
    axes[0, 0].set_ylim(0, 1.04)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.55, .992),
               ncol=2, frameon=False, fontsize=7, columnspacing=2)
    ax = axes[1, 1]
    colors = (".7", fs.BLUE, fs.RED, "white")
    component_labels = ("Outside deposits", "Mobile inventory", "Outlet solids", "Balance remainder")
    positive, negative = np.zeros(2), np.zeros(2)
    for key, color, label, hatch in zip(COMPONENTS, colors, component_labels, ("", "//", "..", "xx")):
        values = np.array([analysis["comparisons"][method]["steady_minus_dilution"][key]
                           for method in METHODS])
        bottom = np.where(values >= 0, positive, negative)
        ax.bar([0, 1], values, bottom=bottom, width=.38, color=color, label=label,
               hatch=hatch, edgecolor=".25", linewidth=.4)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    net = [analysis["comparisons"][method]["steady_minus_dilution"]["net_solid_input"]
           for method in METHODS]
    ax.plot([0, 1], net, "kD", ms=4, label="Net input difference")
    ax.axhline(0, color=".3", lw=.6)
    ax.set(xticks=[0, 1], xticklabels=["R", "D+I"],
           ylabel="Sustained minus dilution", xlabel="Reference target")
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    ax.set_ylim(-.001, max(positive)*1.7)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="lower center",
               bbox_to_anchor=(.54, .035), ncol=2, columnspacing=1.5)
    for index, ax in enumerate(axes.flat):
        ax.set_title(f"({chr(97+index)})", loc="left", fontsize=8.5, pad=6)
    fs.save(fig, output.with_suffix(".pdf"), size_mm=(137.1, 142),
            preview_path=None if preview is None else preview/"treatment_tasks.png")
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        from matplotlib.colors import to_rgb
        for ax in fig.axes:
            legend = ax.get_legend()
            lines = list(ax.lines)+(legend.get_lines() if legend is not None else [])
            patches = list(ax.patches)+(legend.get_patches() if legend is not None else [])
            for line in lines:
                if line.get_color() in (fs.BLUE, fs.RED):
                    line.set_color(".15" if line.get_color() == fs.BLUE else ".5")
            for patch in patches:
                grey = np.dot(to_rgb(patch.get_facecolor()), [.2126, .7152, .0722])
                patch.set_facecolor(str(grey))
        fig.savefig(preview/"treatment_tasks_grayscale.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"results/i060_transport_refinement")
    parser.add_argument("--output", type=Path, default=ROOT/"results/i060_forward/treatment_tasks.json")
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--no-plot", action="store_true", help="Export numerical analysis only")
    args = parser.parse_args()
    analysis = analyze_treatment(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if not args.no_plot:
        plot_treatment(analysis, args.output, args.preview)
    print(json.dumps({"reference": analysis["reference"], "comparisons": analysis["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()

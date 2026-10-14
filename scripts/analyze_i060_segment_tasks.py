"""Evaluate simultaneous segment delivery on saved particle trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from i060_observation_operators import interval_integral

ROOT = Path(__file__).resolve().parents[1]


def segment_histories(fields, phi0, edges):
    """Integrate mobile and deposited actual solids, including off-grid edges."""
    x = fields["x"]
    sigma = fields["sigma"]
    mobile = (phi0 - sigma) * fields["c"]
    phases = {}
    for name, values in (("mobile", mobile), ("deposited", sigma)):
        phases[name] = np.array([
            [interval_integral(x, profile, start, stop) for profile in values.T]
            for start, stop in zip(edges[:-1], edges[1:])
        ])
    phases["total"] = phases["mobile"] + phases["deposited"]
    return phases


def simultaneous_intervals(times, inventories, requirements, deadline):
    """Intersect all linear inventory inequalities on every saved time interval.

    Inventories have shape (requirements, times). Closed intervals include an
    isolated feasible instant, and separate feasible windows remain separate.
    """
    times = np.asarray(times, dtype=float)
    inventories = np.asarray(inventories, dtype=float)
    requirements = np.asarray(requirements, dtype=float)
    if (times.ndim != 1 or len(times) < 2
            or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0)):
        raise ValueError("Saved times must be finite and strictly increasing")
    if (requirements.ndim != 1 or not len(requirements)
            or inventories.shape != (len(requirements), len(times))
            or not np.all(np.isfinite(inventories))
            or not np.all(np.isfinite(requirements)) or np.any(requirements <= 0)):
        raise ValueError("Expected finite segment histories and positive requirements")
    if not times[0] <= deadline <= times[-1]:
        raise ValueError("Deadline lies outside the saved trajectory")

    start, stop = times[:-1], times[1:]
    duration = stop-start
    slack = inventories[:, :-1]-requirements[:, None]
    change = np.diff(inventories, axis=1)
    crossing = np.divide(-slack, change, out=np.zeros_like(slack), where=change != 0)
    lower = np.maximum(0., np.max(np.where(change > 0, crossing, -np.inf), axis=0))
    upper = np.minimum(np.minimum(1., (deadline-start)/duration),
                       np.min(np.where(change < 0, crossing, np.inf), axis=0))
    valid = ((start <= deadline) & (lower <= upper)
             & ~np.any((change == 0) & (slack < 0), axis=0))
    lefts = np.where(lower == 0, start, start+lower*duration)[valid]
    rights = np.where(upper == 1, stop, start+upper*duration)[valid]
    windows = []
    for left, right in zip(lefts, rights):
        if windows and left <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], float(right))
        else:
            windows.append([float(left), float(right)])
    return windows


def clip_windows(windows, start):
    """Intersect closed attainment windows with the allowed stopping domain."""
    return [[max(left, start), right] for left, right in windows if right >= start]


def stopping_state(fields, histories, target_indices, time, edges):
    """Read segment inventories and integrate boundary transport to one time."""
    times = fields["times"]
    flow = fields["q"].reshape(-1)
    state = {
        name: np.array([np.interp(time, times, series) for series in values])
        for name, values in histories.items()
    }

    def accumulated(values):
        if time == times[0]:
            return 0.
        return interval_integral(times, values, times[0], time)

    inlet = accumulated(fields["j"][0])
    advective = accumulated(flow * fields["c"][0])
    outlet = accumulated(fields["j"][-1])
    target_mobile = float(state["mobile"][target_indices].sum())
    target_deposited = float(state["deposited"][target_indices].sum())
    outside_mobile = float(state["mobile"].sum()) - target_mobile
    outside_deposited = float(state["deposited"].sum()) - target_deposited
    initial = float(histories["total"][:, 0].sum())
    edge_flux = np.array([np.interp(edges, fields["x"], profile)
                          for profile in fields["j"].T]).T
    segment_net = np.array([accumulated(series) for series in edge_flux[:-1]-edge_flux[1:]])
    return {
        "time": float(time),
        "flow": float(np.interp(time, times, flow)),
        "segment_mobile": state["mobile"].tolist(),
        "segment_deposited": state["deposited"].tolist(),
        "segment_total": state["total"].tolist(),
        "segment_balance_defect": (histories["total"][:, 0]+segment_net-state["total"]).tolist(),
        "material_partition": {
            "net_inlet_solid_volume": inlet,
            "inlet_advective_solid_volume": advective,
            "inlet_dispersive_solid_volume": inlet - advective,
            "outlet_solid_volume": outlet,
            "mixture_volume": accumulated(flow),
            "target_mobile_solid_volume": target_mobile,
            "target_deposited_solid_volume": target_deposited,
            "outside_mobile_solid_volume": outside_mobile,
            "outside_deposited_solid_volume": outside_deposited,
            "initial_solid_volume": initial,
            "balance_defect": initial + inlet - outlet - float(state["total"].sum()),
        },
    }


def evaluate_segment_task(fields, phi0, task, *, stop_start=None):
    phase = task["endpoint_phase"]
    if phase not in ("total", "deposited"):
        raise ValueError("The endpoint phase must be total or deposited")
    edges = np.asarray(task["segment_edges"], dtype=float)
    segments = np.asarray(task["target_segments"], dtype=int)
    means = np.asarray(task["minimum_mean_solid_fraction"], dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0)
            or edges[0] != fields["x"][0] or edges[-1] != fields["x"][-1]):
        raise ValueError("Segment edges must partition the entire saved column")
    if (segments.ndim != 1 or not len(segments) or len(set(segments)) != len(segments)
            or np.any(segments < 1) or np.any(segments >= len(edges))
            or means.shape != segments.shape):
        raise ValueError("Requirements must identify distinct one-based column segments")
    target_indices = segments - 1
    requirements = means * np.diff(edges)[target_indices]
    histories = segment_histories(fields, phi0, edges)
    times = fields["times"]
    deadline = task["deadline"]
    windows = simultaneous_intervals(
        times, histories[phase][target_indices], requirements, deadline)
    admissible = clip_windows(windows, times[0] if stop_start is None else stop_start)
    at_deadline = stopping_state(fields, histories, target_indices, deadline, edges)
    sampled = times <= deadline
    maximum = np.maximum(histories[phase][:, sampled].max(axis=1),
                         at_deadline[f"segment_{phase}"])
    at_stop = None
    controlling = []
    if admissible:
        at_stop = stopping_state(fields, histories, target_indices, admissible[0][0], edges)
        delivered = np.asarray(at_stop[f"segment_{phase}"])[target_indices]
        # Equality is tested at floating-point roundoff, not a dose tolerance.
        equality = np.abs(delivered - requirements) <= (
            16 * np.finfo(float).eps * np.maximum(np.abs(requirements), np.abs(delivered)))
        controlling = segments[equality].tolist()
    return {
        "endpoint_phase": phase,
        "requirements": requirements.tolist(),
        "feasible_intervals": windows,
        "admissible_intervals": admissible,
        "first_stop": at_stop,
        "controlling_segments": controlling,
        "deadline_state": at_deadline,
        "maximum_segment_inventory_before_deadline": maximum.tolist(),
        "unattained_segments": segments[maximum[target_indices] < requirements].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/i060_transport_refinement")
    parser.add_argument("--task", type=Path, default=ROOT / "config/i060_segment_delivery.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/i060_forward/segment_tasks.json")
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    cases = {}
    for method in ("resistance", "augmented"):
        for inlet in ("steady", "dilution"):
            name = f"{inlet}_{method}_29"
            source = args.input / name / "fields.npz"
            model = json.loads(source.with_name("result.json").read_text(encoding="utf-8"))["model"]
            with np.load(source) as fields:
                case = evaluate_segment_task(fields, model["phi0"], task)
            case["source"] = source.as_posix()
            cases[name] = case
            stop = case["first_stop"]
            description = (f"first stop = {stop['time']:.8f}, controlling segments = {case['controlling_segments']}"
                           if stop else f"no joint attainment; unattained segments = {case['unattained_segments']}")
            print(f"{name}: {description}")
    result = {
        "task": task,
        "field_family": "Existing resident-concentration inlet fields with constant viscosity.",
        "units": "Solid inventories and cumulative volumes are normalized by whole-column bulk volume.",
        "input_definition": "Signed net column-face input, with advective and dispersive contributions reported separately.",
        "evaluation": "Spatial trapezoids, then piecewise-linear segment histories; joint inequality intersection to the deadline and boundary-flux integration to the first common feasible instant.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

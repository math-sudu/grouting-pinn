"""Collection-window and whole-segment observation operators."""

import numpy as np

RHO_W, RHO_C = 1.0, 3.1  # g/cm^3


def interval_integral(coordinate, values, start, stop):
    """Integrate a sampled scalar on its actual support, without extrapolation."""
    coordinate, values = np.asarray(coordinate), np.asarray(values)
    if coordinate.ndim != 1 or values.shape != coordinate.shape:
        raise ValueError("Expected matching one-dimensional samples")
    if not np.all(np.diff(coordinate) > 0):
        raise ValueError("Sample coordinates must be strictly increasing")
    if not coordinate[0] <= start < stop <= coordinate[-1]:
        raise ValueError("Observation window lies outside the saved field support")
    inside = (coordinate > start) & (coordinate < stop)
    nodes = np.concatenate(([start], coordinate[inside], [stop]))
    return float(np.trapezoid(np.interp(nodes, coordinate, values), nodes))


def collection_density(times, flow, solid_flux, start, stop, rho_s=RHO_C, rho_w=RHO_W):
    """Mass/volume of an effluent collection; fluxes are volume per area/time.

    solid_flux is j(L,t). At a zero-dispersion outlet it equals q(t)c(L,t).
    A common area and dimensional time factor cancel in this density ratio.
    """
    volume = interval_integral(times, flow, start, stop)
    if volume <= 0:
        raise ValueError("A collected density requires positive collected volume")
    solids = interval_integral(times, solid_flux, start, stop)
    return rho_w + (rho_s-rho_w)*solids/volume


def segment_inventory(x, c, sigma, phi0, start, stop, rho_s=RHO_C, rho_w=RHO_W):
    """Whole-segment saturated inventory and the thesis-equivalent W/C.

    The returned density uses initial pore volume. Interpreting it as a delayed
    measured mass requires accounting for post-stop drainage and mass loss.
    """
    c, sigma = np.asarray(c), np.asarray(sigma)
    mobile = interval_integral(x, (phi0-sigma)*c, start, stop)
    deposited = interval_integral(x, sigma, start, stop)
    solid = mobile+deposited
    pore = phi0*(stop-start)
    if not 0 < solid < pore:
        raise ValueError("Finite W/C requires both water and solids in the segment")
    return {
        "mobile_solid_volume_per_area": mobile,
        "deposited_solid_volume_per_area": deposited,
        "total_solid_volume_per_area": solid,
        "solid_fraction_initial_pore_volume": solid/pore,
        "density_g_cm3": rho_w+(rho_s-rho_w)*solid/pore,
        "equivalent_water_cement_ratio": rho_w*(pore-solid)/(rho_s*solid),
    }

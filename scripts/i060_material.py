"""Local Newtonian grout viscosity and gravity for the delivery case.

The measured age/concentration table is interpolated with shape-preserving
cubic Hermite polynomials. Interpolation first in age, then in concentration,
retains the measured values and the pure-water limit without a fitted curve.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def pchip(knots, values, points):
    """Evaluate PCHIP, with values on the last axis and broadcast query axes."""
    h = knots.diff()
    delta = values.diff(dim=-1)/h
    left, right = delta[..., :-1], delta[..., 1:]
    same_sign = left*right > 0
    # A flat interval has zero slope. Avoid divisions by zero in both branches
    # so that automatic differentiation also remains finite at the water node.
    safe_left = torch.where(same_sign, left, torch.ones_like(left))
    safe_right = torch.where(same_sign, right, torch.ones_like(right))
    w1, w2 = 2*h[1:]+h[:-1], h[1:]+2*h[:-1]
    middle = torch.where(same_sign, (w1+w2)/(w1/safe_left+w2/safe_right), 0.)

    def endpoint(h0, h1, d0, d1):
        slope = ((2*h0+h1)*d0-h0*d1)/(h0+h1)
        slope = torch.where(slope*d0 <= 0, 0., slope)
        return torch.where((d0*d1 <= 0) & (slope.abs() > 3*d0.abs()), 3*d0, slope)

    slopes = torch.cat((
        endpoint(h[0], h[1], delta[..., 0], delta[..., 1])[..., None],
        middle,
        endpoint(h[-1], h[-2], delta[..., -1], delta[..., -2])[..., None],
    ), dim=-1)
    index = torch.searchsorted(knots, points.contiguous(), right=True).clamp(1, len(knots)-1)-1
    values = values.expand(*points.shape, len(knots))
    slopes = slopes.expand_as(values)

    def take(array, offset):
        return array.gather(-1, (index+offset)[..., None]).squeeze(-1)

    width = h[index]
    u = (points-knots[index])/width
    return ((2*u**3-3*u**2+1)*take(values, 0)
            +(u**3-2*u**2+u)*width*take(slopes, 0)
            +(-2*u**3+3*u**2)*take(values, 1)
            +(u**3-u**2)*width*take(slopes, 1))


@dataclass(frozen=True)
class GroutMaterial:
    cement_density_kg_m3: float
    water_density_kg_m3: float
    water_viscosity_pa_s: float
    reference_viscosity_pa_s: float
    batch_age_at_start_min: float
    length_m: float
    initial_permeability_m2: float
    mechanical_pressure_difference_pa: float
    gravity_along_flow_m_s2: float
    concentrations: list[float]
    ages_min: list[float]
    viscosity_table_pa_s: list[list[float]]

    @classmethod
    def from_config(cls, config):
        source = json.loads((ROOT/config["rheology_data"]).read_text(encoding="utf-8"))
        material, column = config["material"], config["column"]
        ratios = source["water_cement_mass_ratios"]
        rho_s, rho_w = material["cement_density_kg_m3"], material["water_density_kg_m3"]
        water_mu = material["water_viscosity_pa_s"]
        first_ratio = config["feed"]["first_water_cement_mass_ratio"]
        ref_mu = source["dynamic_viscosity_pa_s"][0][ratios.index(first_ratio)]
        obj = cls(
            rho_s, rho_w, water_mu, ref_mu, material["batch_age_at_start_min"],
            column["length_m"], column["initial_permeability_m2"],
            column["mechanical_pressure_difference_pa"], column["gravity_along_flow_m_s2"],
            [0.]+[1/(1+ratio*rho_s/rho_w) for ratio in reversed(ratios)],
            source["hydration_ages_min"],
            [[water_mu]+list(reversed(row)) for row in source["dynamic_viscosity_pa_s"]],
        )
        final_age = obj.batch_age_at_start_min+obj.reference_time_s*config["model"]["end"]/60
        if not obj.ages_min[0] <= obj.batch_age_at_start_min <= final_age <= obj.ages_min[-1]:
            raise ValueError("The injection horizon lies outside the measured hydration ages.")
        max_rho = rho_w+(rho_s-rho_w)*obj.concentrations[-1]
        if max_rho*obj.gravity_along_flow_m_s2*obj.length_m >= obj.mechanical_pressure_difference_pa:
            raise ValueError("The specified pressure cannot sustain upward flow across the concentration range.")
        return obj

    @property
    def reference_flux_m_s(self):
        return (self.initial_permeability_m2*self.mechanical_pressure_difference_pa
                /(self.reference_viscosity_pa_s*self.length_m))

    @property
    def reference_time_s(self):
        return self.length_m/self.reference_flux_m_s

    @property
    def initial_flow(self):
        water_head = (self.water_density_kg_m3*self.gravity_along_flow_m_s2
                      *self.length_m/self.mechanical_pressure_difference_pa)
        return (1-water_head)*self.reference_viscosity_pa_s/self.water_viscosity_pa_s

    def solid_fraction(self, water_cement_mass_ratio):
        return 1/(1+water_cement_mass_ratio*self.cement_density_kg_m3/self.water_density_kg_m3)

    def viscosity(self, c, t):
        age = self.batch_age_at_start_min+t*self.reference_time_s/60
        ages = c.new_tensor(self.ages_min)
        table = c.new_tensor(self.viscosity_table_pa_s)
        at_age = torch.stack([pchip(ages, table[:, j], age)
                              for j in range(table.shape[1])], dim=-1)
        return pchip(c.new_tensor(self.concentrations), at_age, c)

    def relative_viscosity(self, c, t):
        return self.viscosity(c, t)/self.reference_viscosity_pa_s

    def gravity(self, c):
        density = self.water_density_kg_m3+(self.cement_density_kg_m3-self.water_density_kg_m3)*c
        return density*self.gravity_along_flow_m_s2*self.length_m/self.mechanical_pressure_difference_pa

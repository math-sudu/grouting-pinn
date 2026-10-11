# Grouting PINN

Code for **A resistance-coupled physics-informed neural network for pressure-controlled permeation grouting**, by Pengcheng Zhu and Tielin Chen.

The state network learns concentration, deposition and dispersive flux. Deposition changes both mobile pore volume and permeability. The resistance formulation (R) reconstructs pressure and flow by differentiable quadrature. The integral-augmented formulation (D+I, `augmented`) learns the hydraulic fields and enforces local Darcy and global resistance constraints.

The integrated inlet-startup function controls the deposited state and learned hydraulic maps. It gives zero initial deposition and hydraulic rates for finite network outputs. The concentration map fixes the changing inlet value while allowing higher concentrations inside the column after dilution. R uses 2340 active parameters; D+I uses 6790.

## Install and check

Use Python 3.10 or newer. Runs use CPU double precision and PyTorch 2.8.0.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s scripts -p "test_i060_*.py"
```

The tests cover initial values and rates, boundary conditions, hydraulic feedback, phase storage, collection operators and first-attainment integration.

## Reproduce the study

Run these commands from the repository root:

```sh
python scripts/run_i060_suite.py
python scripts/analyze_i060_forward.py
python scripts/analyze_i060_supply_tasks.py
python scripts/analyze_i060_treatment_tasks.py --no-plot
```

The configuration `config/i060_forward.json` specifies six runs: R and D+I for sustained supply and dilution at retention coefficient 4, and sustained supply at coefficient 8. Each run uses seed 29, width 32, 32-point hydraulic quadrature, 4000 Adam steps and up to 600 L-BFGS iterations. It then continues with 64-point quadrature and up to 1200 L-BFGS iterations on the stated denser transport points. Parallel runs share the CPU; elapsed times are not used for a speed comparison.

Each base run writes `model.pt`, `fields.npz` and `result.json` under `results/i060_forward/<case>/`; continuation writes the same files under `results/i060_transport_refinement/<case>/`. The suite also records each stage's console log. All manuscript hydraulic and particle results use the continuation fields.

The state-map identifier is saved in each result. Weights from the earlier temporal map must be retrained before continuation with this version, because the same weight shapes represent different physical fields.

## Read the supplied histories

When the six field directories are already available, run only the three analysis commands. They produce:

- `analysis.json`, `curves.csv` and `profiles.csv`: the six complete solutions and paired hydraulic/particle responses.
- `supply_tasks.json`: first attainment of the common whole-column deposited target 0.046, including input, mixture throughput, mobile inventory, outlet delivery and deposit placement.
- `treatment_tasks.json`: first attainment of deposited targets in the fixed interval [1/3, 1], their reachable ranges, material partition and post-dilution regional capture integral.

All solid inventories are divided by whole-column volume. Attainment uses interpolation between saved times and integrates the final partial time interval. Unattained targets remain unattained. The treatment reference is the regional inventory when sustained R first reaches the whole-column target.

`--no-plot` exports the complete numerical treatment analysis using NumPy alone. The manuscript figure rendering is maintained in the author's GeoSkills workspace. Saved numerical fields and experimental records are available from the corresponding author, Tielin Chen (tlchen1@bjtu.edu.cn).

Experimental CSVs can be included in the summary with:

```sh
python scripts/analyze_i060_forward.py --experimental-data data/zhang_2020
```

## Main files

| File | Purpose |
| --- | --- |
| `scripts/i060_coupled.py` | Neural fields, physics, training and field export |
| `scripts/i060_coupled_pilot.py` | Shared network and differentiation helpers |
| `scripts/refine_i060_transport.py` | Matched transport continuation |
| `scripts/run_i060_suite.py` | Six-case reproduction command |
| `scripts/analyze_i060_forward.py` | Current fields and physical summaries |
| `scripts/analyze_i060_supply_tasks.py` | Target attainment and transport integration |
| `scripts/analyze_i060_treatment_tasks.py` | Treatment requirements and capture |
| `scripts/i060_observation_operators.py` | Collection and segment operators |

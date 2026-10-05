# Grouting PINN

Training and analysis code for **A resistance-coupled physics-informed neural
network for particle retention and injectability in pressure-controlled
permeation grouting**, by Pengcheng Zhu and Tielin Chen.

Public repository: https://github.com/math-sudu/grouting-pinn

The model couples suspension transport, particle retention, pore-volume loss
and permeability through a differentiable hydraulic-resistance integral.
The supplied configuration runs sustained and diluted inlet histories with
three PINN hydraulic formulations: resistance integration (`resistance`, R),
local Darcy enforcement (`differential`, D), and local Darcy enforcement with
an integral constraint (`augmented`, D+I).

The state network has four raw outputs, with the outlet concentration offset
evaluated independently from the interior profile. At width 32, R has 2340
active parameters; the two additional hydraulic networks in D and D+I bring
their total to 6790.

## Installation

Use Python 3.10 or newer and install the dependencies in a virtual environment:

```sh
python -m pip install -r requirements.txt
```

The implementation runs on CPU in double precision. The manuscript runs used
PyTorch 2.8.0 with two CPU threads per run. The publication checks used Python
3.13.7, PyTorch 2.8.0 and NumPy 2.4.4.

Run all commands below from the repository root.

## Check the implementation

```sh
python -m unittest discover -s scripts -p "test_i060_*.py" -v
```

The tests cover initial and boundary conditions, mobile and deposited storage,
hydraulic-feedback gradients, collection-window integration and segment
inventories. A short run checks the training and export pipeline:

```sh
python scripts/i060_coupled.py --method resistance --inlet dilution --width 8 --order 8 --adam 2 --lbfgs 1 --output results/smoke
```

This short run is an execution check. Use the full configuration below for the
manuscript cases.

## Reproduce the numerical study

```sh
python scripts/run_i060_suite.py --config config/i060_forward.json
python scripts/repair_i060_darcy.py --input results/i060_forward/steady_differential_29 --output results/i060_forward/steady_differential_repaired_29
python scripts/refine_i060_transport.py --input results/i060_forward/steady_resistance_29 --output results/i060_transport_refinement/steady_resistance_29
python scripts/refine_i060_transport.py --input results/i060_forward/dilution_resistance_29 --output results/i060_transport_refinement/dilution_resistance_29
python scripts/refine_i060_transport.py --input results/i060_forward/steady_augmented_29 --output results/i060_transport_refinement/steady_augmented_29
python scripts/refine_i060_transport.py --input results/i060_forward/dilution_augmented_29 --output results/i060_transport_refinement/dilution_augmented_29
python scripts/analyze_i060_forward.py
```

The suite runs all six inlet/formulation combinations using seed 29, width 32,
32-point resistance quadrature, 4,000 Adam steps and up to 600 L-BFGS
iterations. The second command reproduces the additional residual-adaptive
training of the sustained-supply D case described in the manuscript.
The four R and D+I transport continuations use the same denser column, outlet
and inlet-transition samples. Their complete fields and histories supply the
particle comparison; the R fields supply the physical-response figures and
table. The original runs supply the paired hydraulic comparison.

Each baseline run writes `model.pt`, `fields.npz`, and `result.json` under
`results/i060_forward/`; transport continuations write the same files under
`results/i060_transport_refinement/`. The suite also records training logs and `suite.json`.
Analysis produces `analysis.json`, including the original suite and continuation
summaries. The original-suite `curves.csv` and `profiles.csv` provide the hydraulic
comparison; the physical-response plots read the continuation `fields.npz`
directly. These outputs include flow, concentration, deposited material, pressure,
solid inventories and collection-window observables. Training durations depend
on the CPU; floating point results can vary across dependency versions and hardware.

## Code map

| File | Purpose |
| --- | --- |
| `scripts/i060_coupled.py` | Physical problem, neural fields, residuals, training and evaluation |
| `scripts/i060_coupled_pilot.py` | Original feasibility model and shared network/autodifferentiation helpers |
| `scripts/run_i060_suite.py` | Six-case training suite |
| `scripts/repair_i060_darcy.py` | Residual-adaptive continuation of the local Darcy formulation |
| `scripts/refine_i060_transport.py` | Transport continuation for the outlet-delivery assessment |
| `scripts/i060_observation_operators.py` | Collection density and whole-segment inventories |
| `scripts/analyze_i060_forward.py` | Numerical summaries and CSV export |
| `config/i060_forward.json` | Manuscript training configuration |

The dimensionless forward cases are fully specified in the code and require
no experimental input files. Digitized experimental observations and the saved
numerical data used in the manuscript are available from the corresponding
author, Tielin Chen (tlchen1@bjtu.edu.cn), on request.
When those experimental CSV files are available, include their collection and
segment summaries by passing their directory explicitly:

```sh
python scripts/analyze_i060_forward.py --experimental-data data/zhang_2020
```

Without this option, the analysis command processes the numerical study alone.

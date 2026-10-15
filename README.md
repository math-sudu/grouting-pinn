# Grouting PINN

Code for **History-dependent cement placement in grouting using a resistance-embedded PINN**, by Pengcheng Zhu and Tielin Chen.

The particle networks represent concentration, dispersive flux and irreversible deposition. The resistance-embedded PINN (REmb-PINN, R in the code and figures) reconstructs pressure and flow by differentiable quadrature. The integral-augmented formulation (D+I, `augmented`) learns the hydraulic fields and enforces local Darcy and global resistance constraints.

Material-feed cases use a total-solid-flux inlet, a cumulative nonnegative capture rate, concentration- and age-dependent viscosity, and gravity. Six-segment total-solid and capture balances supplement the pointwise equations. The constant-viscosity cases use a prescribed resident inlet concentration. The saved `model` and `config` identify each case's physical inputs and state map.

## Install and check

Use Python 3.10 or newer and PyTorch 2.8.0. Calculations use double precision; field export runs on CPU, and training supports CPU or CUDA.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s scripts -p "test_i060_*.py"
```

The tests cover initial and boundary conditions, irreversible capture, flux and storage identities, measured rheology, gravity, hydraulic gradients and observation operators.

## Reconstruct the manuscript results

Run these commands from the repository root:

```sh
python scripts/export_i060_fields.py results/i060_feed results/i060_schedule results/i060_transport_refinement
python scripts/analyze_i060_forward.py
python scripts/analyze_i060_supply_response.py
```

The supplied `model.pt` and `result.json` files contain the final weights, physical model and recorded training settings. The export command reconstructs `fields.npz` beside each checkpoint. The material cases use 257 capture nodes and 64-point resistance quadrature; the numerical fields include the exact feed-transition events.

| Checkpoint directory | Case |
| --- | --- |
| `results/i060_feed/steady_resistance` | Sustained rich feed, R |
| `results/i060_feed/dilution_resistance` | Rich-to-dilute feed, R |
| `results/i060_feed/dilution_augmented` | Rich-to-dilute feed, D+I |
| `results/i060_feed/water_resistance` | Rich-to-water feed, R |
| `results/i060_schedule/constant_0p5_0p5_0` | Constant equal-volume stock blend |
| `results/i060_schedule/constant_0_1_0` | Constant water–cement ratio 5 |
| `results/i060_schedule/constant_0_0p515625_0p484375` | Constant water–cement ratio 10 |
| `results/i060_transport_refinement/{steady,dilution}_{resistance,augmented}_29` | Four constant-viscosity R/D+I cases |

The supply analysis writes `results/i060_forward/supply_response.json`: cement and water inputs, outlet cement, six-segment mobile and deposited inventories, downstream totals, and resistance components. Downstream inventory covers segments 3–6. Water-tail event entries distinguish the switch midpoint, pure-water start and final observation. The constant-viscosity analysis writes `analysis.json`, `curves.csv` and `profiles.csv` in the same output directory.

The material column is 0.30 m long and 0.10 m in diameter, with a 100 kPa pressure difference, initial permeability `1e-11 m2` and a 21.3 min observation interval. The two stocks have water–cement mass ratios 2 and 5 and share a hydration clock. Viscosity data and their source DOI are in `data/yang_2020/rheology.json`; geometry, feeds and inventory scales are in `config/i060_material_delivery.json`.

## Particle origins at blend start

`solve_i060_origins.py` holds each material R solution's flow, pore volume and capture capacity fixed and solves two nonnegative particle cohorts with a separate PINN. The old cohort is mobile at `t=0.525` (11.1825 min); the new cohort enters afterward. Both have their own transport equation, inlet flux and segment mass balances. Capture is integrated from the common retention law. The second cohort is solved directly, not obtained by subtracting from the total state.

```sh
python scripts/solve_i060_origins.py dilution --initialize
python scripts/solve_i060_origins.py mixed --initialize
python scripts/solve_i060_origins.py dilution --resume --nx 120 --nt 256 --adam 0 --lbfgs 2200
python scripts/solve_i060_origins.py mixed --resume --nx 120 --nt 256 --adam 0 --lbfgs 2200
```

The first two commands fit a linear-output initialization; the subsequent `--resume` commands impose nonnegative concentrations and exactly zero initial new-particle inventory on a denser sample set. Outputs are `results/i060_origins/<case>/model.pt` and `result.json`. The summary contains initial mobile mass, later cement supply, downstream and whole-column capture, outlet transport, remaining mobile mass, and consistency with the parent state. Add `--device cpu` to run without CUDA. With saved cohort weights, `--evaluate-only --device cpu` reconstructs these quantities without training. The parent checkpoints already supplied above are the only required inputs; exporting their dense fields is optional for this analysis.

For the reported solutions, old particles contribute about 88.3 g downstream under rich-to-dilute supply and 61.0 g under constant mixing; later supply contributes about 19.3 and 22.9 g. Their differences, together with the 0.27 g pre-existing deposit difference, account for the approximately 24 g final advantage.

## Common-deposit and stage accounts

After exporting the parent fields and completing the two particle-origin runs above, run:

```sh
python scripts/analyze_i060_staging.py
```

This reads the saved histories without changing the supply schedules or retraining the parent models. It writes `results/i060_forward/staging_analysis.json`, using the constant mixture's final deposits in segments 3–6 as the target and locating its first attainment by linear interpolation. The reported rich-to-dilute history reaches 227.83 g at 18.32 min, compared with 21.30 min for constant mixing, using 17.5% less water and 4.0% less cement, with 12.9% less outlet cement. The output also gives complete segment inventories, material accounts before and after blend start, and the old-particle inventory/capture-fraction decomposition.

The manuscript's figure rendering is maintained in the author's GeoSkills workspace. Experimental records are available from the corresponding author, Tielin Chen (tlchen1@bjtu.edu.cn).

Experimental CSVs can be included in the summary with:

```sh
python scripts/analyze_i060_forward.py --experimental-data data/zhang_2020
```

## Training

`scripts/run_i060_suite.py` runs the four configured constant-viscosity R/D+I cases, followed by transport continuation. `scripts/i060_coupled.py` trains the material cases, for example:

```sh
python scripts/i060_coupled.py --method resistance --inlet dilution --material-config config/i060_material_delivery.json --capture-nodes 257 --order 64 --spatial-samples 80 --temporal-samples 512 --outlet-samples 32 --inventory-loss --capture-inventory-loss --device cuda --output results/new_dilution
```

`--inlet steady`, `dilution` or `water` selects the feed history. `--initial-recipe H L W` sets the volume fractions of rich stock, dilute stock and water; `--switch-time` sets the transition midpoint. `--resume` continues compatible saved weights. The supplied checkpoints retain the settings actually used for the manuscript, including continuation stages.

## Main files

| File | Purpose |
| --- | --- |
| `scripts/i060_coupled.py` | Neural fields, physics, training and field export |
| `scripts/i060_material.py` | Differentiable measured rheology and gravity |
| `scripts/i060_coupled_pilot.py` | Shared network and differentiation helpers |
| `scripts/export_i060_fields.py` | Reconstruct fields from supplied weights |
| `scripts/refine_i060_transport.py` | Matched transport continuation |
| `scripts/run_i060_suite.py` | Four-case constant-viscosity training |
| `scripts/analyze_i060_forward.py` | Current fields and physical summaries |
| `scripts/analyze_i060_supply_response.py` | Material inputs, destinations and resistance |
| `scripts/analyze_i060_staging.py` | Common downstream-deposit events and stage accounts |
| `scripts/solve_i060_origins.py` | Independent old/new particle cohorts on fixed R states |
| `scripts/i060_observation_operators.py` | Collection and segment operators |

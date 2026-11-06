# Geometric Discretization Effects on Gravity Forward Modeling

Reproducible Python code and synthetic numerical results for comparing **rectangular, regular hexagonal, and Voronoi polygonal** gravity-model discretizations. The notebook also demonstrates gravity responses of individual prisms and a spherical tesseroid.

**Repository:** https://github.com/professordelimar/prisms_effects

## Scope

- Analytical and numerical verification of gravity forward operators.
- Mesh convergence, computational benchmarks, and regional–residual differences.
- A 21-member mesh ensemble and controlled density/interface-depth sensitivity tests.
- High-resolution figures exported individually for the associated manuscript.

**Important:** This is a **synthetic numerical study**, not an inversion of observed Marajó gravity data. The supplied example Bouguer CSV is retained as input material but is **not used** to construct the revised synthetic reference field. The reference is evaluated by numerical quadrature and checked for refinement, not claimed to be an exact analytical solution.

## Setup

Requires Python 3.10+ (recommended). From the repository root:

```bash
python -m pip install -r requirements.txt
jupyter lab prismas.ipynb
```

Inside `prismas.ipynb`, use:

```python
FULL_PUBLICATION_RUN = True
STRICT_CRAMERI = True
```

Then select **Restart Kernel and Run All Cells**. The publication run is computationally intensive; set `FULL_PUBLICATION_RUN = False` for a shorter diagnostic run. Runtime depends on your system. The scientific `roma_r` palette requires `cmcrameri`.

## Repository layout

```text
prismas.ipynb                     # Main reproducible notebook
gravimetria_marajo.py             # Gravity forward-modeling functions
examples/                         # Supplied synthetic surfaces and sample configurations
outputs/reproducible_figures/     # Selected numerical outputs and experiment manifest
requirements.txt                 # Python dependencies
.gitignore                       # Excludes generated cache and large figure exports
```

Running the notebook creates `Figures/` with publication panels (`figure1a`–`figure5e`, `figureS1a`–`figureS1c`) and `outputs/reproducible_figures/cache/` with intermediate computations. These large or reproducible generated files are intentionally not committed.

The committed `outputs/reproducible_figures/` files provide the numerical arrays (`.npz`), benchmark and sensitivity tables (`.csv`), and the full execution record (`experiment_manifest.json`). Not every data file in `examples/` is used by the main notebook; the manuscript figures rely primarily on the tabulated topography and Moho surfaces.

## Reproducibility and limitations

The notebook defines its grid, model parameters, random seed, operators and convergence checks. Benchmarked runtimes are hardware-dependent. The quadrature refinement check is made at selected validation stations and is not a rigorous error bound for the whole mapped area. Mesh-ensemble dispersion is a **numerical discretization diagnostic**, not an estimate of geological uncertainty. Always consult `outputs/reproducible_figures/experiment_manifest.json` when quoting numerical results.

## Citation

If you use this code, please cite the associated manuscript when its bibliographic details become available, and link to this repository. Source code and results should be cited with the repository commit hash for reproducibility.

## Author

Professor Delimar — [GitHub](https://github.com/professordelimar)

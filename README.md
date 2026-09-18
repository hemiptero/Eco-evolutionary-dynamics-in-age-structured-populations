# Linking allele-frequency change to non-linear matrix demography

Source code and simulation scripts accompanying the manuscript **"Linking
allele-frequency change to non-linear matrix demography: a hybrid
eco-evolutionary framework"** (José Arturo Alcántara Rodriguez, UNAM;
submitted for peer review).

## Overview

The framework integrates single-locus Mendelian inheritance with non-linear,
stage-structured demography. Rather than assembling genetic and demographic
transitions into a single projection operator via vec-permutation, it decouples
the two steps: density-dependent survival acts on the state vector under Ricker
regulation, while inheritance operates on a panmictic gamete pool. Every
operation therefore stays at the scale of the genotype-specific vital-rate
matrices, and because the density penalty is confined to the survival
component, the Jacobian at equilibrium remains available in closed form.

## Scripts

| File | Produces |
| --- | --- |
| `plot_fig2_buteo.py` | Figure 2 — structural check against the *Buteo buteo* colour polymorphism |
| `plot_fig3_archetype.py` | Figure 3 — invasion dynamics in the fast-lived archetype |
| `plot_fig4_1D_sweep.py` | Figure 4 — one-dimensional survival-advantage sweep |
| `plot_fig5_2D_heatmap.py` | Figure 5 — two-dimensional stability landscape |
| `reverse_engineering.py` | Appendix A — reconstruction of the DD and LL matrices |
| `appendix_B_bifurcation.py` | Appendix B / S.M.1 — Jacobian, bifurcation threshold and classification |

Figures are written to `figures/`.

## Requirements

Python 3.8 or later, with NumPy, SciPy and Matplotlib:

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
python3 plot_fig5_2D_heatmap.py
```

`plot_fig5_2D_heatmap.py` runs 2,500 independent 1500-generation simulations
and takes several minutes. The other scripts complete in well under a minute.

## Expected output

Running the scripts reproduces the values reported in the manuscript:

```
appendix_B_bifurcation.py
  Bifurcation threshold: 8.96% early-life survival advantage
  lambda_dom = +0.154838 +0.987958i,  |lambda_dom| = 1.000018
  arg = 81.093 deg  ->  Neimark-Sacker, predicted quasi-period 4.44 generations

plot_fig5_2D_heatmap.py
  Resident equilibrium at f = 1: [1068.6  75.65  26.01] females (2340.5 individuals)
  Analytical bifurcation threshold at f = 1: 8.96% advantage
  CV = 0.05 crossed at 11.51% advantage
  CV = 0.25 crossed at 15.14% advantage
  CV = 0.45 crossed at 36.80% advantage

plot_fig3_archetype.py
  Resident equilibrium: [1068.6  75.65  26.01] females (2340.5 individuals)
  CV over the final 100 generations: 0.2496

plot_fig2_buteo.py
  Asymptotic genotype frequencies (generation 50): DD 0.109, DL 0.693, LL 0.198
```

## Notes on implementation

The density penalty is applied **per stage**, `phi_i = exp(-N_i / K_eff,i)`,
matching equations 9-10 of the manuscript. A scalar total-density formulation
yields a different dynamical system and a different bifurcation threshold.

In `appendix_B_bifurcation.py` the equilibrium is located by numerical
continuation rather than forward iteration, because beyond the bifurcation the
fixed point is unstable and iteration no longer converges to it. The same
applies in `plot_fig5_2D_heatmap.py` above a baseline fecundity of roughly
f = 1.15, and in `plot_fig3_archetype.py`, where the resident's own fixed
point loses stability or the emerging oscillation requires an extended
(1500-generation) horizon to resolve past the invasion transient.

## Citation

Alcántara Rodriguez, J. A. (2026). *Linking allele-frequency change to
non-linear matrix demography: a hybrid eco-evolutionary framework*.
Manuscript submitted for peer review.

## License

MIT — see [LICENSE.txt](LICENSE.txt).

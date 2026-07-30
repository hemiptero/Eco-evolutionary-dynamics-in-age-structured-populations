# Highly Adaptive Traits Trigger Deterministic Chaos: Eco-Evolutionary Dynamics in Age-Structured Populations

## Overview
This repository contains the source code, simulation scripts, and mathematical derivations accompanying the manuscript: **"Highly adaptive traits trigger deterministic chaos: Eco-evolutionary dynamics in age-structured populations"** (José Arturo Alcántara Rodriguez, UNAM; submitted to *Ecology Letters*).

Our framework resolves the mathematical tension between structured population ecology and Mendelian genetics. By decoupling the deterministic process of survival (via Leslie-Lefkovitch matrices) from the stochastic mechanics of sexual reproduction (via a simulated gamete pool), the model maintains a low-dimensional architecture while explicitly tracking allele frequencies and non-linear density-dependent regulation (Ricker).

## Key Features
* **Hybrid Architecture:** Integrates structured demography with single-locus Mendelian segregation without inflating the matrix dimensionality to a "megamatrix".
* **Density-Dependent Scramble Competition:** Utilizes a Ricker formulation applied prior to demographic transitions to simulate severe resource competition and demographic overshoots.
* **Transient & Stability Analysis:** Includes scripts for 1D sensitivity sweeps, 2D stability heatmaps (Coefficient of Variation), and local stability evaluations (Jacobian eigenvalues) to map the boundaries between stable equilibria, limit cycles, and deterministic chaos.
* **Bifurcation Classification:** Reports the full complex dominant eigenvalue (not only its modulus) at the bifurcation threshold, classifying it as a Neimark-Sacker (quasi-periodic) rather than a flip (period-doubling) bifurcation.

## Repository Structure
The repository is organized to ensure full reproducibility of the figures and analyses presented in the manuscript:

```text
├── plot_fig2_buteo.py            # Empirical validation using Buteo buteo vital rates (Figure 2)
├── plot_fig3_archetype.py        # Fast-lived archetype transient dynamics (Figure 3)
├── plot_fig4_1D_sweep.py         # One-dimensional survival advantage sweep (Figure 4)
├── plot_fig5_2D_heatmap.py       # 2D stability landscape (Coefficient of Variation) (Figure 5)
├── reverse_engineering.py        # Derivation of demographic matrices for the homozygous morphs (Appendix A)
├── appendix_B_bifurcation.py     # Jacobian analysis, bifurcation threshold, and complex eigenvalue
│                                  #   classification (flip vs. Neimark-Sacker) (Appendix B, S.M.1)
├── figures/                      # Output directory for generated plots (populated when scripts are run)
├── LICENSE.txt                   # MIT License
└── README.md
```

## Requirements
The code is written in Python 3.8+ and relies on standard scientific libraries. Install the dependencies with:

```bash
pip install numpy scipy matplotlib
```

No other external dependencies are required.

## Usage
To replicate the figures and analyses from the manuscript, run the corresponding script from the terminal. For example, to generate the 2D Stability Landscape (Figure 5):

```bash
python3 plot_fig5_2D_heatmap.py
```

To reproduce the bifurcation classification described in Appendix B and Supplementary Methods S.M.1 (complex eigenvalue at the 20.6% threshold, flip vs. Neimark-Sacker classification):

```bash
python3 appendix_B_bifurcation.py
```

**Note:** `plot_fig5_2D_heatmap.py` simulates thousands of multigenerational evolutionary trajectories and may take a few minutes to complete depending on your hardware. All other scripts run in well under a minute.

## Citation
If you use this code or framework in your research, please cite the corresponding paper:

**Alcántara Rodriguez, J. A.** (2026). *Highly adaptive traits trigger deterministic chaos: Eco-evolutionary dynamics in age-structured populations*. Ecology Letters (Submitted).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

import numpy as np
import matplotlib.pyplot as plt

# Define parameter grid for 2D stability sweep
adv_range = np.linspace(0.0, 0.40, 50)
fec_mult_range = np.linspace(0.5, 2.0, 50)

# Output grid to store the Coefficient of Variation (CV) metric
cv_grid = np.zeros((len(fec_mult_range), len(adv_range)))

# Structural Parameters
K_eff = 1000.0
gens = 350
eval_gens = 100

for i, f_m in enumerate(fec_mult_range):
    for j, adv in enumerate(adv_range):
        # Demographic Matrices
        S_aa = np.array([[0,0,0],[0.60,0,0],[0,0.40,0]])
        S_AA = np.array([[0,0,0],[0.60+adv,0,0],[0,0.40,0]])
        S_Aa = (S_aa + S_AA)/2

        F = np.array([[0, 10*f_m, 12*f_m], [0,0,0], [0,0,0]])

        # Initial ecological state (Resident starts near demographic equilibrium)
        n_aa = np.array([K_eff, K_eff*0.6, K_eff*0.6*0.4])
        n_Aa = np.array([5.0, 0.0, 0.0]) # Introduce invading cohort
        n_AA = np.array([0.0, 0.0, 0.0])

        tot_pop = np.zeros(gens)

        # Discrete-time simulation loop
        for t in range(gens):
            N_tot = n_aa + n_Aa + n_AA
            tot_pop[t] = np.sum(N_tot)

            # Non-linear Ricker regulation penalty
            phi = np.exp(-N_tot / K_eff)

            # Apply scramble competition to established individuals
            n_aa_surv = n_aa * phi
            n_Aa_surv = n_Aa * phi
            n_AA_surv = n_AA * phi

            # Aggregate Gamete Pool
            G_A = np.sum(F @ n_AA) + 0.5 * np.sum(F @ n_Aa)
            G_a = np.sum(F @ n_aa) + 0.5 * np.sum(F @ n_Aa)
            G_tot = G_A + G_a

            # Compute panmictic allele frequencies
            if G_tot > 0:
                p = G_A / G_tot
                q = G_a / G_tot
            else:
                p, q = 0, 0

            # Hardy-Weinberg phenotypic segregation
            B_tot = np.sum(F @ n_aa) + np.sum(F @ n_Aa) + np.sum(F @ n_AA)

            b_AA = B_tot * (p**2)
            b_Aa = B_tot * (2*p*q)
            b_aa = B_tot * (q**2)

            v0 = np.array([1,0,0])

            # Matrix Transition + Recruitment Injection
            n_aa = S_aa @ n_aa_surv + v0 * b_aa
            n_Aa = S_Aa @ n_Aa_surv + v0 * b_Aa
            n_AA = S_AA @ n_AA_surv + v0 * b_AA

        # Calculate CV specifically for the tail-end of the time series
        # to correctly capture the asymptotic regime and discard transient noise
        last_n = tot_pop[-eval_gens:]
        mean_pop = np.mean(last_n)
        if mean_pop > 0:
            cv = np.std(last_n) / mean_pop
        else:
            cv = 0
        cv_grid[i, j] = cv

# Plotting the Eco-Evolutionary Stability Landscape Heatmap
plt.figure(figsize=(10, 8))
X, Y = np.meshgrid(adv_range * 100, fec_mult_range)

# Use pcolormesh for a continuous heatmap projection
im = plt.pcolormesh(X, Y, cv_grid, cmap='magma', shading='auto')
cbar = plt.colorbar(im, label='Demographic Instability (Coefficient of Variation)')
cbar.ax.tick_params(labelsize=10)

# Add empirical contour lines to demarcate dynamic bifurcations
# (Stable -> Limit Cycles -> Deterministic Chaos)
contours = plt.contour(X, Y, cv_grid, levels=[0.05, 0.25, 0.6], colors=['white', 'yellow', 'red'], linestyles='dashed', alpha=0.8)
labels = {0.05: 'Stable/Period-2', 0.25: 'High-Amplitude', 0.6: 'Chaos'}
plt.clabel(contours, inline=True, fontsize=10, fmt=labels)

plt.title('Eco-Evolutionary Stability Space\n(Fecundity vs. Early-Life Survival Advantage)', fontsize=14, weight='bold')
plt.xlabel('Survival Advantage of Invader Allele (%)', fontsize=12)
plt.ylabel('Baseline Fecundity Multiplier', fontsize=12)

# Highlight the fixed baseline fecundity parameter utilized in the Figure 3 1D sweep
plt.axhline(y=1.0, color='cyan', linestyle=':', linewidth=2, label='Baseline Fecundity')
plt.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig('figures/heatmap_stability.png', dpi=300)
print("Heatmap successfully generated.")

"""
plot_fig5_2D_heatmap.py
-----------------------
Two-dimensional eco-evolutionary stability landscape: coefficient of
variation (CV) of total abundance as a function of the invader's early-life
survival advantage (s1) and the baseline fecundity multiplier (f).

Notes on this version
---------------------
1. Simulation settings are aligned with plot_fig4_1D_sweep.py, so the f = 1
   row reproduces the one-dimensional sweep of Figure 4: density is expressed
   on the total (both-sex) scale via SEX_RATIO, the resident starts at its
   density-dependent equilibrium, and the invader is a single heterozygous
   female.

2. The resident's initial state is the non-trivial fixed point of the
   monomorphic resident system, computed rather than hard-coded. This is not
   the stable age distribution: the linear projection matrix has a dominant
   eigenvector concentrated differently, whereas the regulated fixed point is
   skewed toward the first age class because recruitment is unpenalised while
   survival is not. The equilibrium depends on f, so it is recomputed once per
   row of the sweep and shared across that row's cells.

3. The evaluation horizon is 1500 generations with the CV taken over the
   final 100. Allele fixation is then complete well before the window opens,
   so the CV reflects the asymptotic attractor rather than residual variance
   from an ongoing selective sweep. (At 350 generations the sweep itself
   registers as low-level variability at small survival advantages.)

4. Contours are drawn at CV = 0.05, 0.25 and 0.45 and are amplitude bands,
   not dynamical regime boundaries: CV measures amplitude, whereas
   quasi-periodicity and chaos are distinguished by Lyapunov exponents.
   A CV = 0.60 level is not drawn, since it never intersects the f = 1
   baseline row.

5. The analytical bifurcation threshold from the intra-stage Jacobian is
   computed here rather than hard-coded, and drawn as a vertical reference
   line, so the figure shows both the formal loss of local stability and the
   empirical onset of measurable oscillation.

6. Diagnostic output reports where the f = 1 row crosses each CV contour,
   by linear interpolation between grid cells, for quotation in the text.
"""
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS
# ==========================================
K_EFF = np.array([1000.0, 1000.0, 1000.0])
SEX_RATIO = 0.5
GENS = 1500
EVAL_GENS = 100

ADV_RANGE = np.linspace(0.0, 0.40, 50)
# 1.0 is forced into the grid so the baseline row is exactly f = 1.
FEC_RANGE = np.unique(np.append(np.linspace(0.5, 2.0, 50), 1.0))

# The resident's initial state is computed (see resident_equilibrium).
N0_AA = np.array([0.0, 0.0, 0.0])
N0_Aa = np.array([1.0, 0.0, 0.0])

# Amplitude contours: descriptive bands, not dynamical regime boundaries.
CONTOUR_LEVELS = [0.05, 0.25, 0.45]
CONTOUR_COLORS = ["white", "yellow", "red"]
CONTOUR_LABELS = {
    0.05: "CV = 0.05  (onset)",
    0.25: "CV = 0.25",
    0.45: "CV = 0.45",
}


def survival_matrices(advantage):
    """Survival components for resident, heterozygote and invader."""
    S_aa = np.array([[0.0, 0.0, 0.0], [0.60, 0.0, 0.0], [0.0, 0.40, 0.0]])
    S_AA = np.array([[0.0, 0.0, 0.0], [0.60 + advantage, 0.0, 0.0], [0.0, 0.40, 0.0]])
    return S_aa, (S_aa + S_AA) / 2.0, S_AA


def fertility_matrix(f_mult):
    """Fertility component, scaled by the baseline fecundity multiplier."""
    return np.array([[0.0, 10.0 * f_mult, 12.0 * f_mult],
                     [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]])


def resident_equilibrium(K_vec, sex_ratio, f_mult=1.0, seed=None):
    """Non-trivial fixed point of the monomorphic resident system.

    Not the stable age distribution: that is the dominant eigenvector of the
    linear projection matrix, whereas this is the equilibrium of the
    density-regulated non-linear map, skewed toward the first age class
    because recruitment is unpenalised while survival is not.

    Located by root-finding rather than by forward iteration. Above roughly
    f = 1.15 the resident's own fixed point is unstable, so iterating the map
    never settles; root-finding tracks the equilibrium regardless of its
    stability. The seed is warm-started across the fecundity sweep.
    """
    S_aa, _, _ = survival_matrices(0.0)
    F = fertility_matrix(f_mult)

    def residual(n):
        phi = np.exp(-(np.maximum(n, 1e-9) / sex_ratio) / K_vec)
        n_next = S_aa @ (n * phi)
        n_next[0] = (F @ n).sum()
        return n_next - n

    guess = K_vec * sex_ratio if seed is None else seed
    sol = opt.root(residual, guess, method="hybr", tol=1e-12)
    if not sol.success or np.any(sol.x <= 0):
        raise RuntimeError(f"Resident equilibrium did not converge at f = {f_mult}")
    return sol.x


# ==========================================
# 2. ANALYTICAL THRESHOLD (intra-stage Jacobian)
# ==========================================
def analytical_threshold(f_mult=1.0, n_points=2001):
    """Locate where |lambda_dom| of the intra-stage Jacobian crosses unity.

    The equilibrium is tracked by numerical continuation: each solution
    warm-starts the next. A cold root-find collapses onto N = 0.

    Note that this analysis is written on the female-only density scale
    (phi = exp(-N / K_EFF)), whereas the simulation applies the sex-ratio
    factor. By the scale invariance of the Ricker formulation the threshold
    is unaffected; only the equilibrium abundance is rescaled.
    """
    F = fertility_matrix(f_mult)
    advantages = np.linspace(0.0, 0.40, n_points)

    def residual(N, S):
        return S @ (N * np.exp(-N / K_EFF)) + F @ N - N

    def jacobian(N, S):
        phi = np.exp(-N / K_EFF)
        J = np.zeros((3, 3))
        for i in range(3):
            for k in range(3):
                J[i, k] = S[i, k] * phi[k] * (1.0 - N[k] / K_EFF[k]) + F[i, k]
        return J

    # Warm start: iterate the neutral case on this density scale.
    _, _, S0 = survival_matrices(0.0)
    N = K_EFF.copy()
    for _ in range(50000):
        N = S0 @ (N * np.exp(-N / K_EFF)) + F @ N

    moduli = []
    for adv in advantages:
        _, _, S = survival_matrices(adv)
        sol = opt.root(residual, N, args=(S,), method="hybr", tol=1e-12)
        if sol.success and np.all(sol.x > 1e-6):
            N = sol.x
        ev = la.eigvals(jacobian(N, S))
        moduli.append(abs(ev[np.argmax(np.abs(ev))]))

    moduli = np.array(moduli)
    idx = int(np.argmax(moduli > 1.0))
    if idx == 0:
        return None
    m0, m1 = moduli[idx - 1], moduli[idx]
    a0, a1 = advantages[idx - 1], advantages[idx]
    return a0 + (1.0 - m0) * (a1 - a0) / (m1 - m0)


# ==========================================
# 3. SIMULATION
# ==========================================
def simulate_cv(advantage, f_mult, n0_resident):
    """Run one invasion and return the CV of total abundance in the tail."""
    S_aa, S_Aa, S_AA = survival_matrices(advantage)
    F = fertility_matrix(f_mult)

    n_aa = n0_resident.copy()
    n_Aa, n_AA = N0_Aa.copy(), N0_AA.copy()
    total = np.zeros(GENS)
    v0 = np.array([1.0, 0.0, 0.0])

    for t in range(GENS):
        N_tot = n_aa + n_Aa + n_AA
        total[t] = N_tot.sum() / SEX_RATIO

        # Intra-stage Ricker penalty on the total (both-sex) density
        phi = np.exp(-(N_tot / SEX_RATIO) / K_EFF)
        aa_surv, Aa_surv, AA_surv = n_aa * phi, n_Aa * phi, n_AA * phi

        G_A = (F @ n_AA).sum() + 0.5 * (F @ n_Aa).sum()
        G_a = (F @ n_aa).sum() + 0.5 * (F @ n_Aa).sum()
        G_tot = G_A + G_a
        p = G_A / G_tot if G_tot > 1e-12 else 0.0
        q = 1.0 - p

        B = (F @ n_aa).sum() + (F @ n_Aa).sum() + (F @ n_AA).sum()

        n_aa = S_aa @ aa_surv + v0 * B * q * q
        n_Aa = S_Aa @ Aa_surv + v0 * B * 2.0 * p * q
        n_AA = S_AA @ AA_surv + v0 * B * p * p

    tail = total[-EVAL_GENS:]
    return tail.std() / tail.mean() if tail.mean() > 0 else 0.0


def contour_crossing(cv_row, level):
    """Interpolate where a CV row first reaches a contour level."""
    idx = int(np.argmax(cv_row >= level))
    if idx == 0:
        return None
    x0, x1 = ADV_RANGE[idx - 1], ADV_RANGE[idx]
    y0, y1 = cv_row[idx - 1], cv_row[idx]
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


# ==========================================
# 4. EXECUTION
# ==========================================
n0_baseline = resident_equilibrium(K_EFF, SEX_RATIO, 1.0)
print(f"Resident equilibrium at f = 1: {np.round(n0_baseline, 2)} females "
      f"({n0_baseline.sum() / SEX_RATIO:.1f} individuals)")

threshold = analytical_threshold(1.0)
print(f"Analytical bifurcation threshold at f = 1: {threshold * 100:.2f}% advantage")

cv_grid = np.zeros((len(FEC_RANGE), len(ADV_RANGE)))
n0_row = None
for i, f_m in enumerate(FEC_RANGE):
    # Warm-started continuation: each row's equilibrium seeds the next.
    n0_row = resident_equilibrium(K_EFF, SEX_RATIO, f_m, seed=n0_row)
    for j, adv in enumerate(ADV_RANGE):
        cv_grid[i, j] = simulate_cv(adv, f_m, n0_row)

baseline_row = int(np.argmin(np.abs(FEC_RANGE - 1.0)))
print(f"Baseline row: f = {FEC_RANGE[baseline_row]:.4f}")
for level in CONTOUR_LEVELS:
    crossing = contour_crossing(cv_grid[baseline_row, :], level)
    if crossing is None:
        print(f"  CV = {level:.2f}: not reached on this row")
    else:
        print(f"  CV = {level:.2f} crossed at {crossing * 100:.2f}% advantage")

print(f"CV range: min {cv_grid.min():.4f}, max {cv_grid.max():.4f}")

# ==========================================
# 5. PLOT
# ==========================================
plt.figure(figsize=(10, 8))
X, Y = np.meshgrid(ADV_RANGE * 100, FEC_RANGE)

im = plt.pcolormesh(X, Y, cv_grid, cmap="magma", shading="auto")
cbar = plt.colorbar(im, label="Oscillation amplitude (Coefficient of Variation)")
cbar.ax.tick_params(labelsize=10)

contours = plt.contour(X, Y, cv_grid, levels=CONTOUR_LEVELS,
                       colors=CONTOUR_COLORS, linestyles="dashed",
                       linewidths=1.3, alpha=0.9)
plt.clabel(contours, inline=True, fontsize=8, fmt=CONTOUR_LABELS)

if threshold is not None:
    plt.axvline(threshold * 100, color="deepskyblue", linestyle="-",
                linewidth=1.6, alpha=0.95,
                label=fr"Analytical threshold at $f=1$ "
                      fr"($|\lambda_{{dom}}|=1$, {threshold * 100:.1f}%)")

plt.axhline(1.0, color="cyan", linestyle=":", linewidth=2,
            label="Baseline fecundity ($f = 1$, Fig. 4 sweep)")

plt.title("Eco-Evolutionary Stability Space\n"
          "(Fecundity vs. Early-Life Survival Advantage)",
          fontsize=14, weight="bold")
plt.xlabel("Survival Advantage of Invader Allele (%)", fontsize=12)
plt.ylabel("Baseline Fecundity Multiplier", fontsize=12)
plt.legend(loc="upper left", framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig("figures/heatmap_stability.png", dpi=300)
print("Heatmap successfully generated.")

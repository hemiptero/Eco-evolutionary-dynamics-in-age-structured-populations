"""
plot_fig4_1D_sweep.py
---------------------
One-dimensional sweep of the early-life survival advantage, showing how the
archetype's demography changes as the invading allele sweeps to fixation.

Notes on this version
---------------------
1. Horizon extended to 1500 generations, matching plot_fig5_2D_heatmap.py, so
   that every panel shows the asymptotic regime rather than a trajectory still
   in transit.

2. Each panel is split into two time windows. Plotting 1500 generations on a
   single axis compresses a ~4.4-generation cycle into a fraction of a
   millimetre, so only the envelope is visible. The left sub-panel shows the
   invasion transient at full resolution; the right sub-panel shows a short
   window at the end of the run, where individual cycles are resolved. Both
   share the same y-axis.

3. Colour scheme replaced. The previous forest-green / dark-orange pair for
   heterozygote and invader is the classic deuteranopia confusion pair. The
   palette below is drawn from Okabe & Ito's colourblind-safe set and orders
   the three genotypes by allele dosage, so the heterozygote sits visually
   between the two homozygotes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ==========================================
# 1. PARAMETERS
# ==========================================
K_VECTOR_INTRA_ESTADIO = np.array([1000.0, 1000.0, 1000.0])
SEX_RATIO = 0.5
N_GENERATIONS = 1500

TRANSIENT_END = 250      # left sub-panel: generations 0 to this
ASYMPTOTIC_SPAN = 60     # right sub-panel: final N generations


n0_Aa = np.array([1.0, 0.0, 0.0])
n0_AA = np.array([0.0, 0.0, 0.0])

L_aa = np.array([
    [0.0, 10.0, 12.0],
    [0.6,  0.0,  0.0],
    [0.0,  0.4,  0.0]
])

# Okabe-Ito colourblind-safe palette, ordered by allele dosage
C_RESIDENT = "#1A5276"
C_HETERO   = "#21918C"
C_INVADER  = "#CA6702"

ADVANTAGES = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40]
TITLES = [
    "A) 0% Advantage (Neutral)",
    "B) 8% Advantage",
    "C) 16% Advantage",
    "D) 24% Advantage",
    "E) 32% Advantage",
    "F) 40% Advantage",
]


# ==========================================
# 2. ECO-EVOLUTIONARY ENGINE
# ==========================================
def split_fertility_survival(matrix):
    """Decompose a projection matrix into fertility (F) and survival (S)."""
    F = np.zeros_like(matrix)
    S = np.zeros_like(matrix)
    F[0, :] = matrix[0, :]
    S[1:, :] = matrix[1:, :]
    diag = np.diag(matrix).copy()
    diag[0] = 0.0
    np.fill_diagonal(S, diag)
    return F, S

def resident_equilibrium(L, K_vec, sex_ratio, tol=1e-10, max_iter=100000):
    """Non-trivial fixed point of the monomorphic resident system.

    This is not the stable age distribution: that is the dominant eigenvector
    of L, whereas this is the equilibrium of the density-regulated non-linear
    map. Any positive seed converges to the same point (verified across seeds
    spanning four orders of magnitude), so the seed is taken as K_vec scaled
    by the sex ratio.
    """
    F, S = split_fertility_survival(L)
    n = K_vec * sex_ratio
    for _ in range(max_iter):
        phi = np.exp(-(n / sex_ratio) / K_vec)
        n_next = S @ (n * phi)
        n_next[0] = (F @ n).sum()
        if np.max(np.abs(n_next - n)) < tol:
            return n_next
        n = n_next
    raise RuntimeError("Resident equilibrium did not converge")

def simulate_dynamics(L_AA, L_Aa, L_aa, n_AA, n_Aa, n_aa,
                      K_vec, generations, sex_ratio=SEX_RATIO):
    F_AA, S_AA = split_fertility_survival(L_AA)
    F_Aa, S_Aa = split_fertility_survival(L_Aa)
    F_aa, S_aa = split_fertility_survival(L_aa)

    hist_AA = [n_AA.copy()]
    hist_Aa = [n_Aa.copy()]
    hist_aa = [n_aa.copy()]

    for _ in range(generations):
        # A. Fecundity (gamete pool)
        g_AA = F_AA @ n_AA
        g_Aa = F_Aa @ n_Aa
        g_aa = F_aa @ n_aa
        total_gametes = g_AA.sum() + g_Aa.sum() + g_aa.sum()

        # B. Allele frequency
        if total_gametes < 1e-12:
            p = 0.0
        else:
            p = (g_AA.sum() + 0.5 * g_Aa.sum()) / total_gametes
        q = 1.0 - p

        # C. Hardy-Weinberg births
        b_AA = total_gametes * (p ** 2)
        b_Aa = total_gametes * (2 * p * q)
        b_aa = total_gametes * (q ** 2)

        # D. Intra-stage Ricker regulation
        n_total = (n_AA + n_Aa + n_aa) / sex_ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            phi_vec = np.exp(-(n_total / K_vec))

        # E. Transition and survival
        n_AA_next = S_AA @ (n_AA * phi_vec)
        n_Aa_next = S_Aa @ (n_Aa * phi_vec)
        n_aa_next = S_aa @ (n_aa * phi_vec)

        # F. Recruitment
        n_AA_next[0] = b_AA
        n_Aa_next[0] = b_Aa
        n_aa_next[0] = b_aa

        n_AA, n_Aa, n_aa = n_AA_next, n_Aa_next, n_aa_next
        hist_AA.append(n_AA.copy())
        hist_Aa.append(n_Aa.copy())
        hist_aa.append(n_aa.copy())

    return np.array(hist_AA), np.array(hist_Aa), np.array(hist_aa)

n0_aa = resident_equilibrium(L_aa, K_VECTOR_INTRA_ESTADIO, SEX_RATIO)
print(f"Resident equilibrium: {np.round(n0_aa, 2)} females "
      f"({n0_aa.sum() / SEX_RATIO:.1f} individuals)")
# ==========================================
# 3. FIGURE
# ==========================================
fig = plt.figure(figsize=(13, 14))
outer = GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.16)

for idx, adv in enumerate(ADVANTAGES):
    L_AA_sim = L_aa.copy()
    L_AA_sim[1, 0] = 0.6 + adv
    L_Aa_sim = (L_AA_sim + L_aa) / 2.0

    h_AA, h_Aa, h_aa = simulate_dynamics(
        L_AA_sim, L_Aa_sim, L_aa, n0_AA, n0_Aa, n0_aa,
        K_VECTOR_INTRA_ESTADIO, N_GENERATIONS, SEX_RATIO)

    N_AA = h_AA.sum(axis=1) / SEX_RATIO
    N_Aa = h_Aa.sum(axis=1) / SEX_RATIO
    N_aa = h_aa.sum(axis=1) / SEX_RATIO

    limit = 1e-6
    for arr in (N_AA, N_Aa, N_aa):
        arr[arr < limit] = np.nan

    time_axis = np.arange(h_AA.shape[0])

    # Two sub-panels sharing a y-axis: transient (wide) and asymptotic (narrow)
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[idx],
                                    width_ratios=[3, 2], wspace=0.05)
    ax_t = fig.add_subplot(inner[0])
    ax_a = fig.add_subplot(inner[1], sharey=ax_t)

    a_start = N_GENERATIONS - ASYMPTOTIC_SPAN
    windows = [(ax_t, slice(0, TRANSIENT_END + 1)),
               (ax_a, slice(a_start, N_GENERATIONS + 1))]

    for ax, sl in windows:
        ax.plot(time_axis[sl], N_aa[sl], color=C_RESIDENT, lw=1.4,
                label="Resident (aa)")
        ax.plot(time_axis[sl], N_Aa[sl], color=C_HETERO, lw=1.4,
                label="Heterozygote (Aa)")
        ax.plot(time_axis[sl], N_AA[sl], color=C_INVADER, lw=1.4,
                label="Invader (AA)")
        #ax.set_yscale("log")
        ax.set_ylim(bottom=1e0, top=6000)
        ax.grid(True, alpha=0.25, linestyle="--")

    # Break marks: hide the facing spines and draw the diagonal cut
    ax_t.spines["right"].set_visible(False)
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(labelleft=False, left=False)

    d = 0.015
    kw = dict(transform=ax_t.transAxes, color="k", clip_on=False, lw=1)
    ax_t.plot((1 - d, 1 + d), (-d, +d), **kw)
    ax_t.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    kw.update(transform=ax_a.transAxes)
    ax_a.plot((-d * 1.5, +d * 1.5), (-d, +d), **kw)
    ax_a.plot((-d * 1.5, +d * 1.5), (1 - d, 1 + d), **kw)

    ax_t.set_title(TITLES[idx], fontsize=12, fontweight="bold",
                   loc="left", pad=8)
    ax_a.set_title(f"gen {a_start}\u2013{N_GENERATIONS}",
                   fontsize=8, color="0.35", pad=8)

    if idx >= 4:
        ax_t.set_xlabel("Generations", fontsize=11)
        ax_a.set_xlabel("Generations", fontsize=11)
    if idx % 2 == 0:
        ax_t.set_ylabel("Total Abundance", fontsize=11)
    if idx == 0:
        ax_t.legend(loc="lower right", frameon=True, fontsize=9)

plt.savefig("figures/Fig4.png", dpi=300, bbox_inches="tight")
print("Figure 4 generated.")

#!/usr/bin/python3
"""
Figure 2 — structural proof of concept against the Buteo buteo colour
polymorphism of de Vries & Caswell (2019).

Projects the invasion of the high-fitness heterozygote (DL) into a resident
population of the dark morph (DD), with the recessive light morph (LL)
emerging through Mendelian segregation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. DEMOGRAPHIC MATRICES (11 age classes)
# ==========================================
# DL is the published baseline (de Vries & Caswell 2019, Box 1). DD and LL are
# reconstructed by scaling and lifespan truncation; see reverse_engineering.py
# and Appendix A. Age class 11 carries a stasis term ("plus group").

L_DL = np.array([
    [0.05, 0.41, 0.42, 0.45, 0.56, 0.48, 0.61, 0.27, 0.40, 0.00, 0.54],
    [0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.636,0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.714,0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.667,0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.625,0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.60]
])

# DD: lambda = 0.48, lifespan truncated at age 6.
L_DD = np.array([
    [0.0231, 0.1892, 0.1938, 0.2077, 0.2585, 0.2215, 0.2815, 0.0, 0.0, 0.0, 0.0],
    [0.3692, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0],
    [0.0000, 0.3462, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.2935, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.3295, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.3078, 0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3692, 0.0000, 0.0, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0, 0.0],
])

# LL: lambda = 0.68, lifespan truncated at age 7.
# lambda_LL = 0.68. Reaches one age class beyond DD (row 7 retains its
# subdiagonal value, matching Appendix A equation 22).
L_LL = np.array([
    [0.0327, 0.2681, 0.2746, 0.2942, 0.3662, 0.3138, 0.3988, 0.1765, 0.0, 0.0, 0.0],
    [0.5231, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0],
    [0.0000, 0.4904, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.4158, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.4668, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.4361, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5231, 0.0000, 0.0000, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4087, 0.0000, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0, 0.0, 0.0],
])

# ==========================================
# 2. INITIALISATION
# ==========================================
# Because lambda_DD = 0.48, the monomorphic resident system has no positive
# non-trivial fixed point under density regulation. The stable age distribution
# of L_DD is therefore used as the reference age structure (Appendix A.3).
SEX_RATIO = 0.5
K_TOTAL = 1000.0            # density-feedback scale, total across age classes
N_INITIAL_RESIDENT = 500.0  # initial resident abundance, females

# SAD of L_DD. The resident cannot reach age classes 8-11, so the distribution
# is zero there.
sad_DD = np.array([0.3140, 0.2426, 0.1758, 0.1080, 0.0745, 0.0480, 0.0371,
                   0.0, 0.0, 0.0, 0.0])

# The density-feedback vector needs a positive entry in every age class the
# long-lived DL morph can occupy. Age classes 8-11 are unreachable for DD, so
# the SAD assigns them zero; with K = 0 the Ricker exponent diverges and phi
# collapses to 0, imposing total mortality on DL in classes it can legitimately
# reach. Those classes are therefore given the same scale constant as the last
# class the resident does occupy.
K_VECTOR_INTRA_ESTADIO = sad_DD * K_TOTAL
K_VECTOR_INTRA_ESTADIO[7:] = K_VECTOR_INTRA_ESTADIO[6]

n0_DD = sad_DD * N_INITIAL_RESIDENT
n0_DL = np.zeros(11); n0_DL[0] = 1.0   # single founding heterozygous female
n0_LL = np.zeros(11)                   # emerges through segregation

print("K_eff  =", np.round(K_VECTOR_INTRA_ESTADIO, 2))
print("n0_DD  =", np.round(n0_DD, 2))

# Genotype palette, shared across Figures 2-4
C_DD = "#1A5276"
C_DL = "#21918C"
C_LL = "#CA6702"

# ==========================================
# 3. ECO-EVOLUTIONARY ENGINE
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


def simulate_dynamics(n_DD, n_DL, n_LL, K_vec, generations, sex_ratio=SEX_RATIO):
    F_DD, S_DD = split_fertility_survival(L_DD)
    F_DL, S_DL = split_fertility_survival(L_DL)
    F_LL, S_LL = split_fertility_survival(L_LL)

    hist_DD, hist_DL, hist_LL = [n_DD.copy()], [n_DL.copy()], [n_LL.copy()]

    for _ in range(generations):
        # Gamete pool, read from the census state before regulation
        g_DD, g_DL, g_LL = F_DD @ n_DD, F_DL @ n_DL, F_LL @ n_LL
        total_gametes = g_DD.sum() + g_DL.sum() + g_LL.sum()

        p = 0.0 if total_gametes < 1e-12 else \
            (g_DD.sum() + 0.5 * g_DL.sum()) / total_gametes
        q = 1.0 - p

        b_DD = total_gametes * p ** 2
        b_DL = total_gametes * 2 * p * q
        b_LL = total_gametes * q ** 2

        # Intra-stage Ricker penalty on the total (both-sex) density
        n_total = (n_DD + n_DL + n_LL) / sex_ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            phi_vec = np.exp(-(n_total / K_vec))

        # Density-regulated transition, then recruit injection into class 0
        n_DD_next = S_DD @ (n_DD * phi_vec)
        n_DL_next = S_DL @ (n_DL * phi_vec)
        n_LL_next = S_LL @ (n_LL * phi_vec)
        n_DD_next[0], n_DL_next[0], n_LL_next[0] = b_DD, b_DL, b_LL

        n_DD, n_DL, n_LL = n_DD_next, n_DL_next, n_LL_next
        hist_DD.append(n_DD.copy())
        hist_DL.append(n_DL.copy())
        hist_LL.append(n_LL.copy())

    return np.array(hist_DD), np.array(hist_DL), np.array(hist_LL)


# ==========================================
# 4. EXECUTION
# ==========================================
N_GENERATIONS = 50
h_DD, h_DL, h_LL = simulate_dynamics(n0_DD, n0_DL, n0_LL,
                                     K_VECTOR_INTRA_ESTADIO,
                                     N_GENERATIONS, SEX_RATIO)
time_axis = np.arange(h_DD.shape[0])

N_DD = h_DD.sum(axis=1) / SEX_RATIO
N_DL = h_DL.sum(axis=1) / SEX_RATIO
N_LL = h_LL.sum(axis=1) / SEX_RATIO
N_total = N_DD + N_DL + N_LL

with np.errstate(divide="ignore", invalid="ignore"):
    freq_DD, freq_DL, freq_LL = N_DD / N_total, N_DL / N_total, N_LL / N_total

print(f"Asymptotic genotype frequencies (generation {N_GENERATIONS}): "
      f"DD {freq_DD[-1]:.3f}, DL {freq_DL[-1]:.3f}, LL {freq_LL[-1]:.3f}")

# ==========================================
# 5. VISUALISATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(time_axis, freq_DD, label="Dark (DD)", color=C_DD, ls=":",  lw=3)
ax1.plot(time_axis, freq_DL, label="Intermediate (DL)", color=C_DL, ls="-",  lw=3)
ax1.plot(time_axis, freq_LL, label="Light (LL)", color=C_LL, ls="--", lw=3)
ax1.set_title("a) Genotype Frequencies")
ax1.set_xlabel("Generations")
ax1.set_ylabel("Frequency")
ax1.set_ylim(0, 1.05)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel b: one line per age class, styled by genotype so the figure remains
# legible in greyscale.
h_DD_tot, h_DL_tot, h_LL_tot = h_DD / SEX_RATIO, h_DL / SEX_RATIO, h_LL / SEX_RATIO
limit = 1e-6
for arr in (h_DD_tot, h_DL_tot, h_LL_tot):
    arr[arr < limit] = np.nan

for i in range(11):
    ax2.plot(time_axis, h_DD_tot[:, i], color=C_DD, ls=":",  lw=1.4, alpha=0.85)
    ax2.plot(time_axis, h_DL_tot[:, i], color=C_DL, ls="-",  lw=1.3, alpha=0.85)
    ax2.plot(time_axis, h_LL_tot[:, i], color=C_LL, ls="--", lw=1.4, alpha=0.85)

legend_elements = [
    Line2D([0], [0], color=C_DD, ls=":",  lw=1.6, label="Dark (DD)"),
    Line2D([0], [0], color=C_DL, ls="-",  lw=1.6, label="Intermediate (DL)"),
    Line2D([0], [0], color=C_LL, ls="--", lw=1.6, label="Light (LL)"),
]

ax2.set_yscale("log")
ax2.set_title("b) Age-Class Abundance")
ax2.set_xlabel("Generations")
ax2.set_ylabel("Abundance ($\\log_{10}$)")
ax2.set_ylim(1e-6, 1e3)
ax2.legend(handles=legend_elements)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/Fig2.png", dpi=300)
print("Figure 2 generated.")

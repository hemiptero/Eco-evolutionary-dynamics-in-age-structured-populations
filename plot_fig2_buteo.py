#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. DEMOGRAPHIC MATRICES (11x11)
# ==========================================
# Source: de Vries & Caswell (2019).
# L_DL represents the heterozygote with the highest fitness (advantage).
# L_DD (resident) and L_LL (recessive) are derived via reverse engineering
# based on relative fitness reported in the source and truncated lifespans.

# --- DL MATRIX (HETEROZYGOTE / INVADER) ---
# Baseline survival up to age 11.
L_DL = np.array([
    [0.05, 0.41, 0.42, 0.45, 0.56, 0.48, 0.61, 0.27, 0.40, 0.00, 0.54],  # Fecundity
    [0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # Survival 0->1
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

# --- DD MATRIX (RESIDENT) ---
# Truncated mortality: survival becomes 0 at age 6 (max age = 5).
L_DD = np.array([
    [0.0231, 0.1892, 0.1938, 0.2077, 0.2585, 0.2215, 0.2815, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.3692, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.3462, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.2935, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.3295, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.3078, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3692, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # Force 0 survival at age 6
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
])

# --- LL MATRIX (RECESSIVE) ---
# Truncated mortality: survival becomes 0 at age 7 (max age = 6).
L_LL = np.array([
    [0.0327, 0.2681, 0.2746, 0.2942, 0.3662, 0.3138, 0.3988, 0.1765, 0.0000, 0.0000, 0.0000],
    [0.5231, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.4904, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.4158, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.4668, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.4361, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5231, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4087, 0.0000, 0.0000, 0.0000, 0.0000],  # Force 0 survival at age 7
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
])

# ==========================================
# 2. STRUCTURAL VECTORS & INITIALIZATION
# ==========================================

# Initial total resident population (estimated from Figure 2b in the original paper).
# Age-0 abundance ~157. Stable Age Distribution (SAD) for age 0 ~0.3140.
# N_total = 157 / 0.3140 = 500 total individuals.
N_INITIAL_RESIDENT = 500

# --- CARRYING CAPACITY (K) VECTOR ---
# Assumed to be equal to the initial population size (equilibrium start).
# A small "cryptic niche" (0.0371) is added for older ages to allow DL
# to survive biologically in age classes where DD would normally experience 100% mortality.
k_value = N_INITIAL_RESIDENT
missing_Ks = 0.0371  # Minimum baseline K for older age classes

sad_DD = np.array([0.3140, 0.2426, 0.1758, 0.1080, 0.0745, 0.0480, 0.0371, missing_Ks, missing_Ks, missing_Ks, missing_Ks])

K_VECTOR_INTRA_ESTADIO = sad_DD * k_value

# --- INITIAL ABUNDANCE VECTORS (n0) ---
SEX_RATIO = 0.5

# 1. Residents (DD): initialize at effective carrying capacity
n0_DD = sad_DD * N_INITIAL_RESIDENT

# 2. Invaders (DL): introduce 1 female in age class 0
n0_DL = np.zeros(11)
n0_DL[0] = 1.0

# 3. Recessives (LL): initialize at 0 (will emerge via Mendelian segregation)
n0_LL = np.zeros(11)

# ==========================================
# 3. ECO-EVOLUTIONARY ENGINE
# ==========================================
def split_fertility_survival(matrix):
    """Decomposes the projection matrix into separate fertility (F) and survival (S) matrices."""
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

    hist_DD = [n_DD.copy()]
    hist_DL = [n_DL.copy()]
    hist_LL = [n_LL.copy()]

    for t in range(generations):
        # A. Fecundity (gamete pool contribution)
        # Calculate potential gametes produced by breeding adults of each genotype
        g_DD = F_DD @ n_DD
        g_DL = F_DL @ n_DL
        g_LL = F_LL @ n_LL
        total_gametes = g_DD.sum() + g_DL.sum() + g_LL.sum()

        # Calculate allele frequency (p) assuming panmixia
        if total_gametes < 1e-12:
            p = 0.0
        else:
            p = (g_DD.sum() + 0.5 * g_DL.sum()) / total_gametes
        q = 1.0 - p

        # B. Births (Hardy-Weinberg segregation)
        b_DD = total_gametes * (p ** 2)
        b_DL = total_gametes * (2 * p * q)
        b_LL = total_gametes * (q ** 2)

        # C. Ecological regulation (density-dependent scramble competition)
        # Calculates survival penalty factor (phi) based on total ecological density
        n_total = (n_DD + n_DL + n_LL) / sex_ratio

        # Ricker function implementation
        with np.errstate(divide='ignore', invalid='ignore'):
            # If K=0 (no niche), the exponent -> -inf, resulting in phi = 0 (infinite competition)
            phi_vec = np.exp(-(n_total / K_vec))

        # D. Transition (survival & aging)
        # Apply the density-dependent penalty (phi) to established cohorts before transition
        n_DD_next = S_DD @ (n_DD * phi_vec)
        n_DL_next = S_DL @ (n_DL * phi_vec)
        n_LL_next = S_LL @ (n_LL * phi_vec)

        # E. Recruitment: inject segregated newborns into age class 0
        n_DD_next[0] = b_DD
        n_DL_next[0] = b_DL
        n_LL_next[0] = b_LL

        # Update system state
        n_DD, n_DL, n_LL = n_DD_next, n_DL_next, n_LL_next
        hist_DD.append(n_DD.copy())
        hist_DL.append(n_DL.copy())
        hist_LL.append(n_LL.copy())

    return np.array(hist_DD), np.array(hist_DL), np.array(hist_LL)


# ==========================================
# 4. EXECUTION
# ==========================================
N_generations = 50
h_DD, h_DL, h_LL = simulate_dynamics(n0_DD, n0_DL, n0_LL, K_VECTOR_INTRA_ESTADIO, N_generations, SEX_RATIO)
time_axis = np.arange(h_DD.shape[0])

# ==========================================
# 5. VISUALIZATION
# ==========================================
# Calculate total individuals (adjusted by sex ratio)
N_DD = h_DD.sum(axis=1) / SEX_RATIO
N_DL = h_DL.sum(axis=1) / SEX_RATIO
N_LL = h_LL.sum(axis=1) / SEX_RATIO
N_total = N_DD + N_DL + N_LL

# Genotype frequencies
with np.errstate(divide='ignore', invalid='ignore'):
    freq_DD = N_DD / N_total
    freq_DL = N_DL / N_total
    freq_LL = N_LL / N_total

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Frequencies
ax1.plot(time_axis, freq_DD, label='Dark (DD)', color='#08519c', lw=3)
ax1.plot(time_axis, freq_DL, label='Intermediate (DL)', color='forestgreen', lw=3)
ax1.plot(time_axis, freq_LL, label='Light (LL)', color='darkorange', lw=3)
ax1.set_title('a) Genotype Frequencies (Explicit Data)')
ax1.set_ylim(0, 1.05)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel B: Abundance (age-class specific)
h_DD_tot = h_DD / SEX_RATIO
h_DL_tot = h_DL / SEX_RATIO
h_LL_tot = h_LL / SEX_RATIO

# Filter near-zero values to avoid log-scale rendering artifacts
limit = 1e-6
h_DD_tot[h_DD_tot < limit] = np.nan
h_DL_tot[h_DL_tot < limit] = np.nan
h_LL_tot[h_LL_tot < limit] = np.nan

for i in range(11):
    ax2.plot(time_axis, h_DD_tot[:, i], color='#08519c', ls='-', lw=1.5, alpha=0.8)
    ax2.plot(time_axis, h_DL_tot[:, i], color='forestgreen', ls='-', lw=1.5, alpha=0.8)
    ax2.plot(time_axis, h_LL_tot[:, i], color='darkorange', ls='-', lw=1.5, alpha=0.8)

legend_elements = [Line2D([0], [0], color='#08519c', ls='-', label='Dark (DD)'),
                   Line2D([0], [0], color='forestgreen', ls='-', label='Intermediate (DL)'),
                   Line2D([0], [0], color='darkorange', ls='-', label='Light (LL)')]

ax2.set_yscale('log')
ax2.set_title('b) Age-Class Abundance (Total Individuals)')
ax2.set_xlabel('Generations')
ax2.set_ylabel('Abundance ($log_{10}$)')

# Y-axis limits dynamically adjusted to match Caswell's original scale
ax2.set_ylim(1e-6, 1e3)

ax2.legend(handles=legend_elements)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/Fig2.png', dpi=300)
plt.show()

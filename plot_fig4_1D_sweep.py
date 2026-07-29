import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. BASELINE PARAMETERS AND VECTORS
# ==========================================
K_VECTOR_INTRA_ESTADIO = np.array([1000.0, 1000.0, 1000.0])
SEX_RATIO = 0.5
N_generations = 350

# Initial conditions
n0_aa = np.array([700.0, 200.0, 100.0])
n0_Aa = np.array([10.0, 0.0, 0.0])
n0_AA = np.array([0.0, 0.0, 0.0])

# Resident matrix (baseline)
L_aa = np.array([
    [0.0, 10, 12],
    [0.6, 0.0, 0.0],
    [0.0, 0.4, 0.0]
])

# ==========================================
# 2. ECO-EVOLUTIONARY ENGINE (CORE ALGORITHM)
# ==========================================
def split_fertility_survival(matrix):
    F = np.zeros_like(matrix)
    S = np.zeros_like(matrix)
    F[0, :] = matrix[0, :]
    S[1:, :] = matrix[1:, :]
    diag = np.diag(matrix).copy()
    diag[0] = 0.0
    np.fill_diagonal(S, diag)
    return F, S


def simulate_dynamics(L_AA, L_Aa, L_aa, n_AA, n_Aa, n_aa, K_vec, generations, sex_ratio=SEX_RATIO):
    F_AA, S_AA = split_fertility_survival(L_AA)
    F_Aa, S_Aa = split_fertility_survival(L_Aa)
    F_aa, S_aa = split_fertility_survival(L_aa)

    hist_AA = [n_AA.copy()]
    hist_Aa = [n_Aa.copy()]
    hist_aa = [n_aa.copy()]

    for t in range(generations):
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

        # D. Density-dependent competition (intra-stage scramble)
        n_total = (n_AA + n_Aa + n_aa) / sex_ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            phi_vec = np.exp(-(n_total / K_vec))

        # E. Transition & survival
        n_AA_next = S_AA @ (n_AA * phi_vec)
        n_Aa_next = S_Aa @ (n_Aa * phi_vec)
        n_aa_next = S_aa @ (n_aa * phi_vec)

        # F. Recruitment
        n_AA_next[0] = b_AA
        n_Aa_next[0] = b_Aa
        n_aa_next[0] = b_aa

        # G. State update
        n_AA, n_Aa, n_aa = n_AA_next, n_Aa_next, n_aa_next
        hist_AA.append(n_AA.copy())
        hist_Aa.append(n_Aa.copy())
        hist_aa.append(n_aa.copy())

    return np.array(hist_AA), np.array(hist_Aa), np.array(hist_aa)


# ==========================================
# 3. PARAMETER SWEEP AND PLOTTING (FIGURE 4)
# ==========================================
# Additive gradient for early-life survival advantage (0% to 40%)
advantages = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40]
titles = [
    'A) 0% Advantage (Neutral)',
    'B) 8% Advantage',
    'C) 16% Advantage',
    'D) 24% Advantage',
    'E) 32% Advantage',
    'F) 40% Advantage'
]

fig, axs = plt.subplots(3, 2, figsize=(12, 14), sharex=True)
axs = axs.flatten()

for idx, adv in enumerate(advantages):
    # Dynamically generate matrices across the advantage gradient
    L_AA_sim = L_aa.copy()
    L_AA_sim[1, 0] = 0.6 + adv  # Apply survival advantage to transition P(2,1)
    L_Aa_sim = (L_AA_sim + L_aa) / 2.0  # Assume codominance

    # Execute simulation
    h_AA, h_Aa, h_aa = simulate_dynamics(L_AA_sim, L_Aa_sim, L_aa, n0_AA, n0_Aa, n0_aa, K_VECTOR_INTRA_ESTADIO, N_generations, SEX_RATIO)

    # Calculate total abundance per genotype (adjusted for sex ratio)
    N_AA_tot = h_AA.sum(axis=1) / SEX_RATIO
    N_Aa_tot = h_Aa.sum(axis=1) / SEX_RATIO
    N_aa_tot = h_aa.sum(axis=1) / SEX_RATIO

    # Filter near-zero values to preserve proper logarithmic scaling visualization
    limit = 1e-6
    N_AA_tot[N_AA_tot < limit] = np.nan
    N_Aa_tot[N_Aa_tot < limit] = np.nan
    N_aa_tot[N_aa_tot < limit] = np.nan

    time_axis = np.arange(h_AA.shape[0])

    # Plot trajectories
    axs[idx].plot(time_axis, N_aa_tot, color='#08519c', lw=1.5, alpha=0.9, label='Resident (aa)')
    axs[idx].plot(time_axis, N_Aa_tot, color='forestgreen', lw=1.5, alpha=0.9, label='Heterozygote (Aa)')
    axs[idx].plot(time_axis, N_AA_tot, color='darkorange', lw=1.5, alpha=0.9, label='Invader (AA)')

    axs[idx].set_yscale('log')
    axs[idx].set_title(titles[idx], fontsize=13, fontweight='bold', pad=10)
    axs[idx].grid(True, alpha=0.3, linestyle='--')
    axs[idx].set_ylim(bottom=1e0)  # Fix lower limit at 1 individual

    # Conditional formatting for subplot axes
    if idx >= 4:
        axs[idx].set_xlabel('Generations', fontsize=12)
    if idx % 2 == 0:
        axs[idx].set_ylabel('Total Abundance ($log_{10}$)', fontsize=12)

    if idx == 0:
        axs[idx].legend(loc='lower right', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('figures/Fig4.png', dpi=300)
plt.show()

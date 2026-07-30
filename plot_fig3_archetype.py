#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. THEORETICAL MATRICES (3x3 - Fast-lived Archetype)
# ==========================================

# --- L_aa (RESIDENT) ---
L_aa = np.array([
    [0.0, 10.0, 12.0],
    [0.6,  0.0,  0.0],
    [0.0,  0.4,  0.0]
])

# --- L_AA (ADVANTAGEOUS INVADER) ---
L_AA = np.array([
    [0.0, 10.0, 12.0],
    [0.75, 0.0,  0.0],
    [0.0,  0.4,  0.0]
])

# --- L_Aa (HETEROZYGOTE) ---
L_Aa = (L_AA + L_aa) / 2.0

# ==========================================
# 2. VECTORS & INITIALIZATION
# ==========================================
K_VECTOR_INTRA_ESTADIO = np.array([1000.0, 1000.0, 1000.0])
SEX_RATIO = 0.5

# Residents (aa): initialize near demographic equilibrium
n0_aa = np.array([700.0, 200.0, 100.0])

# Heterozygotes (Aa): introduce exactly a single female invader into age class 0
n0_Aa = np.array([1.0, 0.0, 0.0])

# Advantageous homozygotes (AA): initialize at 0 (emerges via segregation)
n0_AA = np.array([0.0, 0.0, 0.0])

# ==========================================
# 3. ECO-EVOLUTIONARY ENGINE
# ==========================================
def split_fertility_survival(matrix):
    """Decomposes the projection matrix into separate fertility (F) and survival (S) matrices."""
    F = np.zeros_like(matrix)
    S = np.zeros_like(matrix)
    F[0, :] = matrix[0, :]
    S[1:, :] = matrix[1:, :]
    return F, S


def simulate_dynamics(n_AA, n_Aa, n_aa, K_vec, generations, sex_ratio=SEX_RATIO):
    F_AA, S_AA = split_fertility_survival(L_AA)
    F_Aa, S_Aa = split_fertility_survival(L_Aa)
    F_aa, S_aa = split_fertility_survival(L_aa)

    hist_AA = [n_AA.copy()]
    hist_Aa = [n_Aa.copy()]
    hist_aa = [n_aa.copy()]

    for t in range(generations):
        # A. Fecundity (gamete pool aggregation)
        g_AA = F_AA @ n_AA
        g_Aa = F_Aa @ n_Aa
        g_aa = F_aa @ n_aa
        total_gametes = g_AA.sum() + g_Aa.sum() + g_aa.sum()

        # Calculate allele frequency (p) assuming panmixia
        if total_gametes < 1e-12:
            p = 0.0
        else:
            p = (g_AA.sum() + 0.5 * g_Aa.sum()) / total_gametes
        q = 1.0 - p

        # B. Births (Hardy-Weinberg principle)
        b_AA = total_gametes * (p ** 2)
        b_Aa = total_gametes * (2 * p * q)
        b_aa = total_gametes * (q ** 2)

        # C. Ecological regulation (Ricker scramble competition)
        n_total = (n_AA + n_Aa + n_aa) / sex_ratio

        with np.errstate(divide='ignore', invalid='ignore'):
            phi_vec = np.exp(-(n_total / K_vec))

        # D. Transition (density-regulated survival and aging)
        n_AA_next = S_AA @ (n_AA * phi_vec)
        n_Aa_next = S_Aa @ (n_Aa * phi_vec)
        n_aa_next = S_aa @ (n_aa * phi_vec)

        # E. Recruitment: injected back into age class 0
        n_AA_next[0] = b_AA
        n_Aa_next[0] = b_Aa
        n_aa_next[0] = b_aa

        # Update system state
        n_AA, n_Aa, n_aa = n_AA_next, n_Aa_next, n_aa_next
        hist_AA.append(n_AA.copy())
        hist_Aa.append(n_Aa.copy())
        hist_aa.append(n_aa.copy())

    return np.array(hist_AA), np.array(hist_Aa), np.array(hist_aa)


# ==========================================
# 4. EXECUTION
# ==========================================
N_generations = 450
h_AA, h_Aa, h_aa = simulate_dynamics(n0_AA, n0_Aa, n0_aa, K_VECTOR_INTRA_ESTADIO, N_generations, SEX_RATIO)
time_axis = np.arange(h_AA.shape[0])

# ==========================================
# 5. VISUALIZATION
# ==========================================
N_AA = h_AA.sum(axis=1) / SEX_RATIO
N_Aa = h_Aa.sum(axis=1) / SEX_RATIO
N_aa = h_aa.sum(axis=1) / SEX_RATIO
N_total = N_AA + N_Aa + N_aa

with np.errstate(divide='ignore', invalid='ignore'):
    freq_AA = N_AA / N_total
    freq_Aa = N_Aa / N_total
    freq_aa = N_aa / N_total

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Color palette matching Figure 2
c_aa = '#08519c'      # aa
c_Aa = 'forestgreen'  # Aa
c_AA = 'darkorange'   # AA

# Panel A: Frequencies
ax1.plot(time_axis, freq_aa, label='Resident (aa)', color=c_aa, lw=3)
ax1.plot(time_axis, freq_Aa, label='Heterozygote (Aa)', color=c_Aa, lw=3)
ax1.plot(time_axis, freq_AA, label='Invader (AA)', color=c_AA, lw=3)
ax1.set_title('a) Genotype Frequencies')
ax1.set_xlabel('Generations')
ax1.set_ylabel('Frequency')
ax1.set_ylim(-0.05, 1.05)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel B: Abundances (age-class specific transients)
h_AA_tot = h_AA / SEX_RATIO
h_Aa_tot = h_Aa / SEX_RATIO
h_aa_tot = h_aa / SEX_RATIO

# Filter near-zero values for clean log rendering
limit = 1e-4
h_AA_tot[h_AA_tot < limit] = np.nan
h_Aa_tot[h_Aa_tot < limit] = np.nan
h_aa_tot[h_aa_tot < limit] = np.nan

# Plot trajectories for the 3 age classes across generations
for i in range(3):
    ax2.plot(time_axis, h_aa_tot[:, i], color=c_aa, lw=1.5, alpha=0.7)
    ax2.plot(time_axis, h_Aa_tot[:, i], color=c_Aa, lw=1.5, alpha=0.7)
    ax2.plot(time_axis, h_AA_tot[:, i], color=c_AA, lw=1.5, alpha=0.7)

legend_elements = [
    Line2D([0], [0], color=c_aa, lw=3, label='Resident (aa)'),
    Line2D([0], [0], color=c_Aa, lw=3, label='Heterozygote (Aa)'),
    Line2D([0], [0], color=c_AA, lw=3, label='Invader (AA)')
]

ax2.set_yscale('log')
ax2.set_title('b) Age-Class Abundance (Transient Oscillations)')
ax2.set_xlabel('Generations')
ax2.set_ylabel(r'Abundance ($\log_{10}$)')
ax2.set_ylim(bottom=1e-1)
ax2.legend(handles=legend_elements)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/Fig3.png', dpi=300)
plt.show()

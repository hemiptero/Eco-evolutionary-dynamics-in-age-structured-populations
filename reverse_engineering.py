import numpy as np

"""
=============================================================================
DESCRIPTION:
This script performs the mathematical derivation of the demographic
projection matrices for the homozygous morphs (DD and LL) of the common
buzzard (Buteo buteo). Because explicit stage-specific vital rates for these
morphs were not published in the original analytical model (de Vries &
Caswell, 2019), this script reverse-engineers them using a baseline-and-
scaling approach.

KEY OPERATIONS:
1. Baseline scaling: multiplies the empirical vital rates of the high-fitness
   heterozygous morph (DL) by a specific factor to exactly match the target
   asymptotic growth rates (lambda = 0.48 for DD; lambda = 0.68 for LL).
2. Lifespan truncation: enforces biological lifespan constraints by zeroing
   out demographic transitions once the oldest reachable age class stops
   receiving inflow (row truncation one index earlier than the column/
   fecundity truncation, since the last living age class still contributes
   fecundity but produces no further survivors).
3. SAD & carrying capacity allocation: computes the dominant eigenvector
   (Stable Age Distribution, SAD) of the resident matrix to distribute the
   total effective carrying capacity (K_eff) across age classes.

OUTPUT:
Generates the formatted NumPy arrays (matrices and vectors) required to
parameterize the main eco-evolutionary simulation in 'plot_fig2_buteo.py'.
=============================================================================
"""

# --- 1. BASELINE MATRIX (DL HETEROZYGOTE) ---
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

# --- 2. REVERSE ENGINEERING ALGORITHM ---
# DD: rows/cols truncated from index 7 onward (row 6 retains its subdiagonal
#     survival value, matching equation 21 in Appendix A).
factor_DD = 0.48 / 1.04
L_DD = L_DL.copy() * factor_DD
L_DD[7:, :] = 0
L_DD[:, 7:] = 0

# LL: rows truncated from index 7 onward, columns (fecundity) from index 8
#     onward (row 6 retains its subdiagonal survival value, matching
#     equation 22 in Appendix A).
factor_LL = 0.68 / 1.04
L_LL = L_DL.copy() * factor_LL
L_LL[7:, :] = 0
L_LL[:, 8:] = 0

# --- 3. STRUCTURAL VECTOR CALCULATION ---
vals, vecs = np.linalg.eig(L_DD)
sad = np.real(vecs[:, np.argmax(np.real(vals))])
sad = np.abs(sad) / np.sum(np.abs(sad))

K_TOTAL = 1000.0
K_VEC = sad * K_TOTAL

N0_TOTAL = 500.0
N0_DD = sad * N0_TOTAL

# --- 4. FORMATTED CONSOLE OUTPUT ---
def print_matrix(name, M):
    print(f"{name} = np.array([")
    for row in M:
        print(f"    [{', '.join(f'{x:.4f}' for x in row)}],")
    print("])")
    print()


def print_vector(name, V):
    print(f"{name} = np.array([{', '.join(f'{x:.4f}' for x in V)}])")
    print()


print_matrix("L_DD", L_DD)
print_matrix("L_LL", L_LL)
print_vector("K_VECTOR_INTRA_ESTADIO", K_VEC)
print_vector("n0_DD", N0_DD)
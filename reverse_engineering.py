import numpy as np

"""
Appendix A — reconstruction of the DD and LL projection matrices.

Explicit vital rates for the homozygous Buteo buteo morphs were not published
in the source analytical model (de Vries & Caswell, 2019). This script derives
them from the high-fitness heterozygote (DL) by:

  1. Scaling the baseline vital rates to match the reported asymptotic growth
     rates (lambda = 0.48 for DD, 0.68 for LL).
  2. Truncating transitions once the oldest reachable age class stops receiving
     inflow; the row is truncated one index earlier than the fecundity column,
     since the last living class still reproduces but produces no survivors.
  3. Computing the stable age distribution of the resident matrix to allocate
     the density-feedback constant and the initial abundance across age classes.

Output: the formatted arrays used to parameterise plot_fig2_buteo.py.
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

# LL: rows truncated from index 8 onward, columns (fecundity) from index 8
#     onward (row 7 retains its subdiagonal survival value, matching
#     equation 22 in Appendix A: LL reaches one age class beyond DD).
factor_LL = 0.68 / 1.04
L_LL = L_DL.copy() * factor_LL
L_LL[8:, :] = 0
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
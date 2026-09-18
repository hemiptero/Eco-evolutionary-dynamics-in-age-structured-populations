"""
appendix_B_bifurcation.py  (corrected)
--------------------------------------
Local stability (Jacobian and dominant eigenvalue) of the age-structured
population under INTRA-STAGE Ricker regulation, and classification of the
bifurcation as the early-life survival advantage crosses its threshold.

CORRECTION relative to the previous version
-------------------------------------------
The earlier script applied a SCALAR total-density penalty,

    phi = exp(-sum(N) / K_eff)                      # all stages share one phi

whereas Methods equations 9-10 and the simulation scripts
(plot_fig3_archetype.py, plot_fig4_1D_sweep.py, plot_fig5_2D_heatmap.py)
apply an INTRA-STAGE penalty,

    phi_i = exp(-N_i / K_eff,i)                     # one phi per stage

These are different dynamical systems. This script now implements the
intra-stage version, so the analytical threshold corresponds to the model
actually simulated in Figures 3-5.

Model:  N_{t+1} = S (N_t o phi(N_t)) + F N_t ,   phi_i = exp(-N_i / K_i)

Jacobian (element-wise density, so phi_k depends only on N_k):

    J[i,k] = S[i,k] * phi_k * (1 - N_k / K_k) + F[i,k]

Two implementation notes:
  * The non-trivial equilibrium is found by pseudo-arclength-free numerical
    continuation: the solution at each advantage warm-starts the next. A
    cold root-find collapses onto the trivial equilibrium N = 0.
  * Forward iteration alone cannot be used past the bifurcation, since the
    fixed point is then unstable; root-finding tracks it regardless.
"""
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

K_EFF = np.array([1000.0, 1000.0, 1000.0])
F_MAT = np.array([[0.0, 10.0, 12.0],
                  [0.0,  0.0,  0.0],
                  [0.0,  0.0,  0.0]])


def survival_matrix(advantage):
    """Survival component S of the monomorphic AA invader."""
    return np.array([[0.0,            0.0, 0.0],
                     [0.60 + advantage, 0.0, 0.0],
                     [0.0,            0.40, 0.0]])


def residual(N, S):
    """Fixed-point condition N_{t+1} - N_t = 0 under intra-stage regulation."""
    phi = np.exp(-N / K_EFF)
    return S @ (N * phi) + F_MAT @ N - N


def jacobian(N, S):
    """Exact Jacobian at N (equation 25, intra-stage form)."""
    phi = np.exp(-N / K_EFF)
    J = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            J[i, k] = S[i, k] * phi[k] * (1.0 - N[k] / K_EFF[k]) + F_MAT[i, k]
    return J


def evaluate_local_stability(n_points=4001, max_advantage=0.40):
    """Continue the non-trivial equilibrium and track the dominant eigenvalue."""
    advantages = np.linspace(0.0, max_advantage, n_points)

    # Warm start: iterate the neutral case to its stable fixed point.
    N = np.array([700.0, 200.0, 100.0])
    S0 = survival_matrix(0.0)
    for _ in range(50000):
        N = S0 @ (N * np.exp(-N / K_EFF)) + F_MAT @ N

    moduli, eigs, equilibria = [], [], []
    for adv in advantages:
        S = survival_matrix(adv)
        sol = opt.root(residual, N, args=(S,), method="hybr", tol=1e-12)
        if sol.success and np.all(sol.x > 1e-6):
            N = sol.x
        ev = la.eigvals(jacobian(N, S))
        dom = ev[np.argmax(np.abs(ev))]
        moduli.append(abs(dom))
        eigs.append(dom)
        equilibria.append(N.copy())

    moduli = np.array(moduli)
    eigs = np.array(eigs)

    idx = int(np.argmax(moduli > 1.0))
    m0, m1 = moduli[idx - 1], moduli[idx]
    a0, a1 = advantages[idx - 1], advantages[idx]
    critical = a0 + (1.0 - m0) * (a1 - a0) / (m1 - m0)

    print("--- LOCAL STABILITY ANALYSIS (INTRA-STAGE JACOBIAN) ---")
    print(f"Bifurcation threshold: {critical * 100:.2f}% early-life survival advantage")
    print(f"Equilibrium at threshold: N* = {np.round(equilibria[idx], 2)}"
          f"  (total {equilibria[idx].sum():.1f})")

    plt.figure(figsize=(8, 5))
    plt.plot(advantages * 100, moduli, color="indigo", lw=2)
    plt.axhline(1.0, color="red", ls="--",
                label=r"Stability threshold $(|\lambda_{dom}| = 1)$")
    plt.axvline(critical * 100, color="gray", ls=":",
                label=f"Bifurcation onset (~{critical * 100:.2f}%)")
    plt.title("Analytical Local Stability Analysis\n"
              "(Intra-stage Ricker regulation, decoupled Jacobian)",
              weight="bold")
    plt.xlabel("Early-Life Survival Advantage (%)")
    plt.ylabel(r"Dominant Eigenvalue $(|\lambda_{dom}|)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/appendix_B_eigenvalues_final.png", dpi=300)

    return advantages, eigs, idx, equilibria


def classify_bifurcation_type(advantages, eigs, idx):
    """Classify the crossing eigenvalue as flip or Neimark-Sacker."""
    lam = eigs[idx]
    print()
    print("--- BIFURCATION TYPE CLASSIFICATION ---")
    print(f"  lambda_dom  = {lam.real:+.6f} {lam.imag:+.6f}i")
    print(f"  |lambda_dom| = {abs(lam):.6f}")

    if abs(lam.imag) < 1e-6:
        kind = "FLIP (period-doubling)" if lam.real < 0 else "saddle-node / transcritical"
        print(f"  --> REAL eigenvalue: {kind} bifurcation")
        return None

    arg_deg = np.degrees(np.angle(lam))
    quasi_period = 360.0 / arg_deg
    print(f"  arg(lambda_dom) = {arg_deg:.3f} deg")
    print("  --> COMPLEX-CONJUGATE PAIR: Neimark-Sacker bifurcation")
    print(f"  --> Predicted quasi-period = {quasi_period:.2f} generations")

    # Strong-resonance check: arg must avoid 0, 180, 120 and 90 degrees.
    for res_deg, order in ((0.0, 1), (180.0, 2), (120.0, 3), (90.0, 4)):
        if abs(arg_deg - res_deg) < 2.0:
            print(f"  !! WARNING: within 2 deg of strong resonance {order}:1")
    return quasi_period


if __name__ == "__main__":
    advantages, eigs, idx, equilibria = evaluate_local_stability()
    classify_bifurcation_type(advantages, eigs, idx)

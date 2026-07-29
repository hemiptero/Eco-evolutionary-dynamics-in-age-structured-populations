"""
appendix_B_bifurcation.py
-------------------------
Standalone script to compute the local stability (Jacobian matrix and
dominant eigenvalues) of the age-structured population under Ricker
regulation, and to classify the bifurcation observed as the survival
advantage crosses its critical threshold.

Reproduces the analysis described in Appendix B and Supplementary Methods
S.M.1 of the manuscript:
  1. The scalar-density Jacobian (equation 25) and its dominant eigenvalue
     across the survival-advantage gradient.
  2. The precise bifurcation threshold (|lambda_dom| = 1).
  3. The full complex dominant eigenvalue at that threshold (not just its
     modulus), used to classify the bifurcation as flip (period-doubling,
     real eigenvalue near -1) or Neimark-Sacker (complex-conjugate pair).
"""
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt


def evaluate_local_stability():
    """
    Scan the survival-advantage gradient, compute the Jacobian at each
    point, and identify the bifurcation threshold where |lambda_dom| > 1.
    """
    K_eff = 1000.0
    advantages = np.linspace(0.0, 0.40, 100)
    dominant_eigenvalues = []
    dominant_eigenvalues_complex = []
    N_stars = []

    print("--- LOCAL STABILITY ANALYSIS (JACOBIAN) ---")
    guess = np.array([K_eff, K_eff * 0.6, K_eff * 0.6 * 0.4])

    for adv in advantages:
        S_AA = np.array([[0, 0, 0], [0.60 + adv, 0, 0], [0, 0.40, 0]])
        F_AA = np.array([[0, 10, 12], [0, 0, 0], [0, 0, 0]])

        # 1. Locate the equilibrium N* via root-finding
        def system_eq(N):
            N = np.maximum(N, 0)
            phi = np.exp(-np.sum(N) / K_eff)
            N_next = S_AA @ (N * phi) + F_AA @ N
            return N_next - N

        sol = opt.root(system_eq, guess, method='hybr')
        N_star = np.maximum(sol.x, 0)
        guess = N_star  # warm-start next iteration
        total_N_star = np.sum(N_star)

        # 2. Assemble the Jacobian at the exact equilibrium
        phi_star = np.exp(-total_N_star / K_eff)
        dphi_dN = -(1.0 / K_eff) * phi_star
        J = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                term_S = S_AA[i, j] * phi_star
                term_density = np.dot(S_AA[i, :], N_star) * dphi_dN
                term_F = F_AA[i, j]
                J[i, j] = term_S + term_density + term_F

        # 3. Store the dominant eigenvalue (modulus and full complex value)
        eigenvals = la.eigvals(J)
        idx = np.argmax(np.abs(eigenvals))
        dom_eig = np.abs(eigenvals[idx])
        dominant_eigenvalues.append(dom_eig)
        dominant_eigenvalues_complex.append(eigenvals[idx])
        N_stars.append(N_star.copy())

    dominant_eigenvalues = np.array(dominant_eigenvalues)
    dominant_eigenvalues_complex = np.array(dominant_eigenvalues_complex)

    # Identify the exact crossing point (where the eigenvalue modulus crosses 1.0)
    crossing_idx = np.where(dominant_eigenvalues > 1.0)[0][0]
    critical_adv = advantages[crossing_idx] * 100
    print(f"Mathematical Bifurcation Point Found At: {critical_adv:.2f}% Advantage")

    plt.figure(figsize=(8, 5))
    plt.plot(advantages * 100, dominant_eigenvalues, color='indigo', lw=2)
    plt.axhline(y=1.0, color='red', linestyle='--', label='Stability Threshold $(|\\lambda_{dom}| = 1)$')
    plt.axvline(x=critical_adv, color='gray', linestyle=':', label=f'Exact Bifurcation Onset (~{critical_adv:.1f}%)')
    plt.title('Analytical Local Stability Analysis\n(Root-Finding Decoupled Jacobian)', weight='bold')
    plt.xlabel('Early-Life Survival Advantage (%)')
    plt.ylabel('Dominant Eigenvalue $(|\\lambda_{dom}|)$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/appendix_B_eigenvalues_final.png', dpi=300)
    plt.show()

    return advantages, dominant_eigenvalues_complex, crossing_idx, N_stars


def classify_bifurcation_type(advantages, eigs_complex, crossing_idx):
    """
    Report the full complex dominant eigenvalue at the crossing point
    (not only its modulus) and classify the bifurcation as flip
    (period-doubling) or Neimark-Sacker.
    """
    lam = eigs_complex[crossing_idx]
    critical_adv = advantages[crossing_idx] * 100

    print()
    print("--- BIFURCATION TYPE CLASSIFICATION ---")
    print(f"At the {critical_adv:.2f}% threshold:")
    print(f"  lambda_dom = {lam.real:+.6f} {lam.imag:+.6f}i")
    print(f"  |lambda_dom| = {abs(lam):.6f}")

    is_real = abs(lam.imag) < 1e-6
    if is_real:
        if lam.real < 0:
            print("  --> REAL, near -1: FLIP (period-doubling) bifurcation")
        else:
            print("  --> REAL, near +1: saddle-node/transcritical bifurcation")
        return None
    else:
        arg_deg = np.degrees(np.angle(lam))
        quasi_period = 360.0 / arg_deg
        print(f"  arg(lambda_dom) = {arg_deg:.3f} deg")
        print("  --> COMPLEX-CONJUGATE PAIR: Neimark-Sacker bifurcation")
        print(f"  --> Predicted quasi-period = 360/{arg_deg:.3f} = {quasi_period:.3f} generations")
        return quasi_period


if __name__ == "__main__":
    advantages, eigs_complex, crossing_idx, N_stars = evaluate_local_stability()
    predicted_period = classify_bifurcation_type(advantages, eigs_complex, crossing_idx)

    if predicted_period is not None:
        print()
        print(f"Predicted quasi-period: {predicted_period:.3f} generations.")

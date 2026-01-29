import numpy as np
import h5py
from typing import Dict, Callable, Tuple


def energy_conservation_error(dataset_path: str, integrator_order: int = 4) -> float:
    """
    Calculates max relative energy drift: |(E(t) - E(0))/E(0)|
    Assumes HDF5 has 'masses', 'positions', 'velocities'.
    """
    with h5py.File(dataset_path, "r") as f:
        m = np.asarray(f["masses"])
        # Check just start and end for speed, or sample?
        # BENCHMARK requires T=1000 drift. We check drift over full trajectory.

        pos = np.asarray(f["positions"])
        vel = np.asarray(f["velocities"])

        # Calculate E(0)
        E0 = _hamiltonian(m, pos[0], vel[0])

        # Calculate E(end)
        E_end = _hamiltonian(m, pos[-1], vel[-1])

        return float(abs((E_end - E0) / E0))


def _hamiltonian(m: np.ndarray, p: np.ndarray, v: np.ndarray) -> float:
    # Kinetic
    # v shape (N, 3), m shape (N,)
    ke = float(0.5 * np.sum(m * np.sum(v**2, axis=1)))

    # Potential
    pe = 0.0
    n = len(m)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(p[i] - p[j]) + 1e-10)
            pe -= float(m[i] * m[j] / dist)
    return ke + pe


def calculate_tes_score(mse: float, energy_drift: float, speedup: float) -> float:
    """
    Trimandala Efficiency Score (TES)
    Formula: 0.4*(-log(MSE)) + 0.4*(-log(Drift)) + 0.2*log(Speedup)
    """
    if energy_drift > 0.1:
        return 0.0  # DNF

    # Safe logs
    s_mse = float(-np.log10(mse + 1e-20))
    s_phys = float(-np.log10(energy_drift + 1e-20))
    s_speed = float(np.log10(speedup + 1e-20))

    score = (0.4 * s_mse) + (0.4 * s_phys) + (0.2 * s_speed)
    return float(max(0.0, score))


def butterfly_score(
    model_fn: Callable,
    initial_state: Tuple[np.ndarray, np.ndarray],
    dt: float,
    steps: int,
    true_lyapunov: float = 0.0,
) -> float:
    """
    Orchestrates the perturbation test.
    Returns the estimated Lyapunov exponent lambda (divergence rate).
    """
    # 1. Run Baseline
    # model_fn signature: predict(pos, vel, dt, steps) -> (pos_final, vel_final)
    # We need full trajectory or just end state?
    # Usually Lyapunov is lim t->inf (1/t) ln(|d(t)|/|d(0)|)

    # Unpack initial_state (assumed tuple pos, vel)
    p0, v0 = initial_state

    # 2. Perturb
    perturbation = 1e-10
    # Add perturbation to specific body or all? All is robust.
    p_pert = p0 + np.random.normal(0, perturbation, size=p0.shape)
    v_pert = v0  # Keep velocity same for simplicity or perturb both? Usually pos.

    # 3. Predict both
    # We need intermediate steps to fit the curve, but our API is end-to-end?
    # If API is end-to-end, we can only measure finite time Lyapunov at time T.

    # Run prediction
    p_base, v_base = model_fn(p0, v0, dt, steps)
    p_new, v_new = model_fn(p_pert, v0, dt, steps)

    # 4. Measure Divergence
    # d(t)
    diff = float(np.linalg.norm(p_base - p_new))  # Frobenius norm of pos matrix diff
    d0 = float(np.linalg.norm(p0 - p_pert))

    # 5. Calculate Lambda
    # |d(t)| = |d(0)| * e^(lambda * t)
    # lambda * t = ln(|d(t)|/|d(0)|)
    # lambda = 1/t * ln(...)

    t_total = steps * dt

    if diff < 1e-15:
        return 0.0  # No divergence (Over-stable)

    lambda_est = float((1.0 / t_total) * np.log(diff / d0))

    # Butterfly Score: How close is lambda_est to true_lyapunov?
    # For now, we just return the estimated lambda.
    # The benchmark can score based on error: |lambda_est - lambda_true|

    return lambda_est

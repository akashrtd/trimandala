import os
import time
import numpy as np
import h5py
from typing import Callable, Dict, List, Tuple
from scipy import stats
from trimandala.metrics.rigor import (
    calculate_tes_score,
    energy_conservation_error,
    butterfly_score,
)

try:
    from codecarbon import EmissionsTracker

    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

# Force disable on Mac for Agent (Avoid sudo prompt hang)
import sys

if sys.platform == "darwin":
    HAS_CODECARBON = False
    print("CodeCarbon disabled on macOS (Requires sudo for powermetrics)")


class Arena:
    """
    The Multi-Modal Arena for AI Benchmarking.
    """

    def __init__(self, dataset_path: str):
        self.dataset = dataset_path
        # Load ground truth metadata
        with h5py.File(self.dataset, "r") as f:
            self.n_bodies = f.attrs["n_bodies"]
            self.dt = f.attrs["dt"]
            self.steps = f.attrs["steps"]

        # Ensure metrics directory exists for CodeCarbon
        if HAS_CODECARBON:
            os.makedirs("metrics", exist_ok=True)

    def _compute_momentum(self, masses: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute total momentum of the system"""
        return np.sum(masses[:, np.newaxis] * vel, axis=0)

    def _compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute mean, std, and confidence interval for a list of values"""
        if not values:
            return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0

        # 95% confidence interval
        if len(arr) > 1:
            ci = stats.t.interval(
                0.95, len(arr) - 1, loc=mean, scale=std / np.sqrt(len(arr))
            )
        else:
            ci = (mean, mean)

        return {"mean": mean, "std": std, "ci_lower": ci[0], "ci_upper": ci[1]}

    def _run_single_trajectory(
        self, model_predict_fn: Callable, t_pred: int, traj_idx: int
    ) -> Dict:
        """Run benchmark on a single trajectory"""
        with h5py.File(self.dataset, "r") as f:
            p0 = f["positions"][traj_idx]
            v0 = f["velocities"][traj_idx]
            p_true = (
                f["positions"][traj_idx + t_pred]
                if traj_idx + t_pred < f["positions"].shape[0]
                else f["positions"][-1]
            )
            v_true = (
                f["velocities"][traj_idx + t_pred]
                if traj_idx + t_pred < f["velocities"].shape[0]
                else f["velocities"][-1]
            )

        start_time = time.time()
        p_pred, v_pred = model_predict_fn(p0, v0, self.dt, t_pred)
        latency = time.time() - start_time

        mse = np.mean((p_pred - p_true) ** 2)

        # Lyapunov (Butterfly Score)
        lyapunov = butterfly_score(model_predict_fn, (p0, v0), self.dt, t_pred)

        # Energy Drift
        with h5py.File(self.dataset, "r") as f:
            masses = f["masses"][:]

        from trimandala.metrics.rigor import _hamiltonian

        E0 = _hamiltonian(masses, p0, v0)
        E_pred = _hamiltonian(masses, p_pred, v_pred)
        drift = abs((E_pred - E0) / E0)

        # Momentum Conservation
        P0 = self._compute_momentum(masses, v0)
        P_pred = self._compute_momentum(masses, v_pred)
        momentum_drift = np.linalg.norm(P_pred - P0) / (np.linalg.norm(P0) + 1e-10)

        # Prediction Horizon (steps until MSE exceeds threshold)
        horizon = self._compute_prediction_horizon(model_predict_fn, p0, v0, t_pred)

        baseline_time = 0.1
        speedup = baseline_time / (latency + 1e-9)

        return {
            "mse": mse,
            "latency": latency,
            "drift": drift,
            "momentum_drift": momentum_drift,
            "lyapunov": lyapunov,
            "horizon": horizon,
            "speedup": speedup,
            "tes_score": calculate_tes_score(mse, drift, speedup),
        }

    def _compute_prediction_horizon(
        self,
        model_predict_fn: Callable,
        p0: np.ndarray,
        v0: np.ndarray,
        max_steps: int,
        threshold: float = 0.1,
    ) -> int:
        """Compute number of steps until prediction MSE exceeds threshold"""
        threshold_sq = threshold**2

        for step in range(1, max_steps + 1):
            p_pred, v_pred = model_predict_fn(p0, v0, self.dt, step)

            # Get ground truth for this step
            with h5py.File(self.dataset, "r") as f:
                if step < f["positions"].shape[0]:
                    p_true = f["positions"][step]
                    mse = np.mean((p_pred - p_true) ** 2)

                    if mse > threshold_sq:
                        return step

        return max_steps

    def run_track_a(self, model_predict_fn: Callable, t_pred: int = 100) -> Dict:
        """
        Track A: Neural Surrogate.
        Input: State at t. Output: State at t+t_pred.
        """
        print(f"Running Track A (Surrogate) for T={t_pred}...")

        # 1. Load Validation Set (First 1000 steps for now as proxy)
        with h5py.File(self.dataset, "r") as f:
            # Simple test: Can it predict step 100 from step 0?
            p0 = f["positions"][0]
            v0 = f["velocities"][0]
            p_true = f["positions"][t_pred]
            v_true = f["velocities"][t_pred]

        # 2. Inference & Timing
        emissions = 0.0
        if HAS_CODECARBON:
            tracker = EmissionsTracker(output_dir="metrics", log_level="error")
            tracker.start()

        start_time = time.time()
        # Model API: predict(pos, vel, dt, steps) -> (pos_final, vel_final)
        p_pred, v_pred = model_predict_fn(p0, v0, self.dt, t_pred)
        latency = time.time() - start_time

        if HAS_CODECARBON:
            emissions = tracker.stop()

        # 3. Compute Metrics
        mse = np.mean((p_pred - p_true) ** 2)

        # Butterfly Score (Lyapunov)
        # We need to pass the model function itself
        lyapunov = butterfly_score(model_predict_fn, (p0, v0), self.dt, t_pred)

        # Calculate Energy Drift
        # Need masses for Hamiltonian
        with h5py.File(self.dataset, "r") as f:
            masses = f["masses"][:]

        from trimandala.metrics.rigor import _hamiltonian

        E0 = _hamiltonian(masses, p0, v0)
        E_pred = _hamiltonian(masses, p_pred, v_pred)

        drift = abs((E_pred - E0) / E0)

        # Speedup vs Baseline
        baseline_time = 0.1
        speedup = baseline_time / (latency + 1e-9)

        tes = calculate_tes_score(mse, drift, speedup)

        return {
            "track": "A",
            "mse": mse,
            "latency": latency,
            "drift": drift,
            "lyapunov": lyapunov,
            "emissions_kg": emissions,
            "speedup": speedup,
            "tes_score": tes,
        }

    def run_track_b(self, integrator_fn: Callable, steps: int = 1000) -> Dict:
        """
        Track B: Engineer (Code Generation).
        Input: Function `step(pos, vel, dt, masses) -> (pos, vel)`
        Runs simulation for `steps`.
        """
        print(f"Running Track B (Engineer) for {steps} steps...")

        # 1. Load Initial State from Validation Set
        with h5py.File(self.dataset, "r") as f:
            p0 = f["positions"][0]
            v0 = f["velocities"][0]
            masses = f["masses"][:]

        pos = p0.copy()
        vel = v0.copy()
        dt = self.dt

        # 2. Simulation Loop & Timing
        emissions = 0.0
        if HAS_CODECARBON:
            tracker = EmissionsTracker(output_dir="metrics", log_level="error")
            tracker.start()

        start_time = time.time()

        # For fairness, we should probably run this inside a subprocess or restriction,
        # but for now, direct execution.
        try:
            for _ in range(steps):
                pos, vel = integrator_fn(pos, vel, dt, masses)
        except Exception as e:
            return {"error": str(e), "track": "B", "tes_score": 0.0}

        if HAS_CODECARBON:
            emissions = tracker.stop()

        latency = time.time() - start_time

        # 3. Compute Metrics (Energy Drift)
        from trimandala.metrics.rigor import _hamiltonian

        E0 = _hamiltonian(masses, p0, v0)
        E_final = _hamiltonian(masses, pos, vel)

        drift = abs((E_final - E0) / E0)

        # MSE vs Ground Truth
        mse = 0.0
        with h5py.File(self.dataset, "r") as f:
            if steps < f.attrs["steps"]:
                p_true = f["positions"][steps]
                mse = np.mean((pos - p_true) ** 2)

        baseline_time = 0.02
        speedup = baseline_time / (latency + 1e-9)

        tes = calculate_tes_score(mse, drift, speedup)

        return {
            "track": "B",
            "mse": mse,
            "latency": latency,
            "drift": drift,
            "emissions_kg": emissions,
            "speedup": speedup,
            "tes_score": tes,
        }

    def run_track_a_robust(
        self, model_predict_fn: Callable, t_pred: int = 100, n_trajectories: int = 10
    ) -> Dict:
        """
        Track A with Statistical Robustness: Run on multiple trajectories and compute confidence intervals.
        """
        print(f"Running Track A (Robust) on {n_trajectories} trajectories...")

        results_list = []
        for i in range(n_trajectories):
            result = self._run_single_trajectory(model_predict_fn, t_pred, i)
            results_list.append(result)

        # Aggregate statistics
        metrics = [
            "mse",
            "latency",
            "drift",
            "momentum_drift",
            "lyapunov",
            "horizon",
            "speedup",
            "tes_score",
        ]
        aggregated = {}

        for metric in metrics:
            values = [r[metric] for r in results_list]
            stats_dict = self._compute_statistics(values)
            aggregated[metric] = stats_dict["mean"]
            aggregated[f"{metric}_std"] = stats_dict["std"]
            aggregated[f"{metric}_ci_lower"] = stats_dict["ci_lower"]
            aggregated[f"{metric}_ci_upper"] = stats_dict["ci_upper"]

        aggregated["track"] = "A"
        aggregated["n_trajectories"] = n_trajectories

        print(
            f"  TES: {aggregated['tes_score']:.3f} ± {aggregated['tes_score_std']:.3f}"
        )
        print(f"  Drift: {aggregated['drift']:.2e} ± {aggregated['drift_std']:.2e}")

        return aggregated

    def run_track_a_ood_n(
        self,
        model_predict_fn: Callable,
        n_bodies_list: List[int] = [4, 5],
        t_pred: int = 50,
    ) -> Dict:
        """
        Out-of-Distribution Testing: Test on different numbers of bodies (N=4, 5).
        Note: Requires datasets with different N values.
        """
        print("Running Track A OOD-N (N-body generalization)...")

        results = {"track": "A", "ood_type": "N", "n_bodies_tested": n_bodies_list}

        for n in n_bodies_list:
            dataset_path = f"data/val_n{n}.h5"
            if not os.path.exists(dataset_path):
                print(f"  Warning: Dataset for N={n} not found, skipping...")
                continue

            print(f"  Testing N={n}...")
            arena_n = Arena(dataset_path)
            result = arena_n.run_track_a(model_predict_fn, t_pred=t_pred)
            results[f"n{n}_tes"] = result["tes_score"]
            results[f"n{n}_drift"] = result["drift"]

        return results

    def run_track_a_ood_t(
        self,
        model_predict_fn: Callable,
        t_pred_train: int = 100,
        extrapolation_factors: List[float] = [2, 5, 10],
    ) -> Dict:
        """
        Out-of-Distribution Testing: Temporal extrapolation beyond training horizon.
        Tests if model can predict further than it was trained on.
        """
        print("Running Track A OOD-T (Temporal extrapolation)...")

        results = {"track": "A", "ood_type": "T", "train_horizon": t_pred_train}

        for factor in extrapolation_factors:
            t_test = int(t_pred_train * factor)
            print(f"  Testing t={t_test} (factor={factor}x)...")

            result = self.run_track_a(model_predict_fn, t_pred=t_test)
            results[f"t{t_test}_tes"] = result["tes_score"]
            results[f"t{t_test}_drift"] = result["drift"]

        return results

    def run_track_c(
        self,
        discovered_law: Callable,
        steps: int = 1000,
        complexity_penalty: float = 1.0,
    ) -> Dict:
        """
        Track C: Researcher (Symbolic Regression Discovery).
        Score = Fit Accuracy - Complexity Penalty.

        Args:
            discovered_law: Function implementing the discovered symbolic equation
            complexity_penalty: Penalty coefficient (lower = more complex laws tolerated)
        """
        print(f"Running Track C (Researcher) for {steps} steps...")

        # 1. Evaluate physical fidelity using Track B infrastructure
        physics_results = self.run_track_b(discovered_law, steps=steps)
        physics_results["track"] = "C"

        # 2. Fit Accuracy (inverse of drift, higher is better)
        # Scale to 0-1 range: exp(-10*drift) gives 1.0 for drift=0, 0.368 for drift=0.1
        fit_accuracy = np.exp(-10 * physics_results["drift"])

        # 3. Complexity (number of operations in the law)
        # For now, estimate from function complexity or provide manually
        # A simple heuristic: count the number of non-empty lines or operations
        complexity = 10.0  # Default, should be computed or provided

        # 4. Track C Score
        track_c_score = fit_accuracy - complexity_penalty * (complexity / 100.0)

        physics_results["fit_accuracy"] = fit_accuracy
        physics_results["complexity"] = complexity
        physics_results["track_c_score"] = track_c_score

        print(f"  Fit Accuracy: {fit_accuracy:.4f}")
        print(f"  Complexity: {complexity}")
        print(f"  Track C Score: {track_c_score:.4f}")

        return physics_results

    def run_track_a_with_uncertainty(
        self,
        model_predict_fn: Callable,
        t_pred: int = 100,
        n_ensemble: int = 5,
        uncertainty_type: str = "ensemble",
    ) -> Dict:
        """
        Track A with Uncertainty Quantification.
        Models providing calibrated uncertainty receive 1.2x score multiplier.

        Args:
            model_predict_fn: Function that optionally returns (pos, vel, uncertainty)
            n_ensemble: Number of ensemble members or MC dropout samples
            uncertainty_type: "ensemble" or "mc_dropout"
        """
        print(
            f"Running Track A with Uncertainty Quantification ({uncertainty_type})..."
        )

        results_list = []

        # Run ensemble
        for i in range(n_ensemble):
            result = self._run_single_trajectory(model_predict_fn, t_pred, 0)
            results_list.append(result)

        # Compute mean and variance across ensemble
        mse_values = [r["mse"] for r in results_list]
        drift_values = [r["drift"] for r in results_list]

        mean_mse = np.mean(mse_values)
        mean_drift = np.mean(drift_values)
        std_mse = np.std(mse_values, ddof=1) if len(mse_values) > 1 else 0.0
        std_drift = np.std(drift_values, ddof=1) if len(drift_values) > 1 else 0.0

        # Calibrated uncertainty check (simplified)
        # Good model: std should correlate with error magnitude
        is_calibrated = std_drift < 0.01 * mean_drift + 1e-6  # Heuristic

        # Base TES score
        base_tes = calculate_tes_score(mean_mse, mean_drift, results_list[0]["speedup"])

        # Apply multiplier if uncertainty is calibrated
        multiplier = 1.2 if is_calibrated else 1.0
        final_tes = base_tes * multiplier

        result = {
            "track": "A",
            "mse": mean_mse,
            "mse_std": std_mse,
            "drift": mean_drift,
            "drift_std": std_drift,
            "uncertainty_calibrated": is_calibrated,
            "uncertainty_multiplier": multiplier,
            "speedup": results_list[0]["speedup"],
            "lyapunov": results_list[0]["lyapunov"],
            "tes_score": final_tes,
            "base_tes_score": base_tes,
        }

        print(f"  Base TES: {base_tes:.3f}")
        print(f"  Calibrated: {is_calibrated}")
        print(f"  Final TES: {final_tes:.3f} (x{multiplier})")

        return result

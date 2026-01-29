import os
import time
import numpy as np
import h5py
from typing import Callable, Dict
from trimandala.metrics.rigor import calculate_tes_score, energy_conservation_error, butterfly_score
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
        with h5py.File(self.dataset, 'r') as f:
            self.n_bodies = f.attrs['n_bodies']
            self.dt = f.attrs['dt']
            self.steps = f.attrs['steps']
        
        # Ensure metrics directory exists for CodeCarbon
        if HAS_CODECARBON:
            os.makedirs("metrics", exist_ok=True)
            
    def run_track_a(self, model_predict_fn: Callable, t_pred: int = 100) -> Dict:
        """
        Track A: Neural Surrogate.
        Input: State at t. Output: State at t+t_pred.
        """
        print(f"Running Track A (Surrogate) for T={t_pred}...")
        
        # 1. Load Validation Set (First 1000 steps for now as proxy)
        with h5py.File(self.dataset, 'r') as f:
            # Simple test: Can it predict step 100 from step 0?
            p0 = f['positions'][0]
            v0 = f['velocities'][0]
            p_true = f['positions'][t_pred]
            v_true = f['velocities'][t_pred]
            
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
        mse = np.mean((p_pred - p_true)**2)
        
        # Butterfly Score (Lyapunov)
        # We need to pass the model function itself
        lyapunov = butterfly_score(model_predict_fn, (p0, v0), self.dt, t_pred)
        
        # Calculate Energy Drift
        # Need masses for Hamiltonian
        with h5py.File(self.dataset, 'r') as f:
            masses = f['masses'][:]
            
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
            "tes_score": tes
        }

    def run_track_b(self, integrator_fn: Callable, steps: int = 1000) -> Dict:
        """
        Track B: Engineer (Code Generation).
        Input: Function `step(pos, vel, dt, masses) -> (pos, vel)`
        Runs simulation for `steps`.
        """
        print(f"Running Track B (Engineer) for {steps} steps...")
        
        # 1. Load Initial State from Validation Set
        with h5py.File(self.dataset, 'r') as f:
            p0 = f['positions'][0]
            v0 = f['velocities'][0]
            masses = f['masses'][:]
            
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
        with h5py.File(self.dataset, 'r') as f:
            if steps < f.attrs['steps']:
                 p_true = f['positions'][steps]
                 mse = np.mean((pos - p_true)**2)

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
            "tes_score": tes
        }

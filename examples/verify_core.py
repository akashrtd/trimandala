
import trimandala.generator as gen
import numpy as np
from trimandala.viz.science import plot_energy_drift, plot_phase_space
from trimandala.metrics.rigor import energy_conservation_error

def main():
    print("=== Trimandala Verification v0.1 ===")
    
    # 1. Define Scenario (Figure-8 Orbit - Chaotic Stability Test)
    # Masses equal
    m = np.array([1.0, 1.0, 1.0])
    
    # Famous Figure-8 Initial Conditions
    p1 = np.array([0.97000436, -0.24308753, 0.0])
    p2 = -p1
    p3 = np.array([0.0, 0.0, 0.0])
    pos = np.array([p1, p2, p3])
    
    v1 = np.array([0.4662036850, 0.4323657300, 0.0])
    v2 = v1
    v3 = -2 * v1
    vel = np.array([v1, v2, v3])
    
    output_file = "verify_run.h5"
    steps = 10000
    dt = 0.001
    
    # 2. Run Simulation (C++ Core + HDF5)
    print(f"Running simulation for {steps} steps...")
    runner = gen.SimulationRunner(output_file, n_bodies=3)
    runner.run(m, pos, vel, steps, dt)
    
    # 3. Check Physics (Energy Drift)
    drift = energy_conservation_error(output_file)
    print(f"Final Energy Drift: {drift:.2e}")
    
    if drift < 1e-9:
        print("[PASS] Physics Core is highly accurate.")
    elif drift < 1e-5:
        print("[WARN] Physics Core is accurate enough for ML, but drift is detectable.")
    else:
        print("[FAIL] Physics Core is broken.")
        
    # 4. Generate Artifacts
    print("Generating Plots...")
    plot_energy_drift(output_file, "verify_energy.png")
    plot_phase_space(output_file, 0, "verify_phase.png")
    
    print("=== Verification Complete ===")

if __name__ == "__main__":
    main()

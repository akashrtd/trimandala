import os
import trimandala.generator as gen
import numpy as np

def generate_dataset(filename, steps, distinct_ics=False):
    print(f"Generating {filename} with {steps} steps...")
    
    # Standard 3-Body Setup (Normalized Units)
    masses = np.array([1.0, 1.0, 1.0])
    
    if not distinct_ics:
        # Figure-8 Stability (Standard)
        p1 = np.array([0.97000436, -0.24308753, 0.0])
        p2 = -p1
        p3 = np.array([0.0, 0.0, 0.0])
        pos = np.array([p1, p2, p3])
        
        v1 = np.array([0.4662036850, 0.4323657300, 0.0])
        v2 = v1
        v3 = -2 * v1
        vel = np.array([v1, v2, v3])
    else:
        # Random Chaotic (Burrau's Problem or pure random)
        # For V1, we stick to Figure-8 but perturbed slightly to create a different trajectory
        # This tests generalization to neighboring phase space
        p1 = np.array([0.97000436, -0.24308753, 0.0]) * 1.01
        p2 = -p1
        p3 = np.array([0.0, 0.0, 0.0])
        pos = np.array([p1, p2, p3])
        
        v1 = np.array([0.4662036850, 0.4323657300, 0.0])
        v2 = v1
        v3 = -2 * v1
        vel = np.array([v1, v2, v3])

    dt = 0.001
    
    runner = gen.SimulationRunner(filename, n_bodies=3)
    runner.run(masses, pos, vel, steps, dt)
    print(f"Done: {filename}")

def main():
    os.makedirs("data", exist_ok=True)
    
    # Train: 1M steps of Standard Figure-8
    generate_dataset("data/train.h5", steps=100_000, distinct_ics=False)
    
    # Val: 20k steps of Perturbed (OOD Check)
    generate_dataset("data/val.h5", steps=20_000, distinct_ics=True)

if __name__ == "__main__":
    main()

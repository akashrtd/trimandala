from trimandala.arena import Arena
from trimandala.baselines.symplectic_euler import symplectic_euler_step
import json

def main():
    print("=== Trimandala Benchmark: Track B (Engineer) ===")
    
    # 1. Setup Arena
    dataset = "data/val.h5"
    arena = Arena(dataset)
    
    # 2. Symplectic Euler Baseline
    print("\n[Evaluating Symplectic Euler 1st Order]")
    
    # Run for 10,000 steps
    # Note: This will be slow in pure Python.
    steps = 10000 
    
    results = arena.run_track_b(symplectic_euler_step, steps=steps)
    print(json.dumps(results, indent=2))
    
    # Check PASS condition
    # For baseline, we expect maybe drift ~1e-2 or 1e-3? 1st order error is O(dt). dt=0.001.
    # Drift 10,000 * 0.001 might be significant.
    # The requirement is TES > 50?
    
    print(f"\nTES Score: {results['tes_score']:.2f}")

if __name__ == "__main__":
    main()

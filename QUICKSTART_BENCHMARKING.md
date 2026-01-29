# Quick Start: Benchmarking with Trimandala

This guide helps you get started with the improved benchmarking features.

---

## Installation

```bash
# Install Trimandala
pip install trimandala

# Install optional dependencies for benchmarking
pip install codecarbon matplotlib seaborn scipy
```

---

## Basic Benchmarking

### 1. Single Trajectory Benchmark

```python
from trimandala.arena import Arena
from trimandala.baselines.linear import LinearBaseline

# Setup
arena = Arena("data/val.h5")
model = LinearBaseline()

# Run benchmark
results = arena.run_track_a(model.predict, t_pred=100)

print(f"TES Score: {results['tes_score']:.3f}")
print(f"Energy Drift: {results['drift']:.2e}")
print(f"Speedup: {results['speedup']:.1f}x")
```

### 2. Robust Benchmarking (Multi-Trajectory)

```python
# Run on 10 trajectories with confidence intervals
results = arena.run_track_a_robust(model.predict, t_pred=100, n_trajectories=10)

print(f"TES: {results['tes_score']:.3f} ± {results['tes_score_std']:.3f}")
print(f"95% CI: [{results['tes_score_ci_lower']:.3f}, {results['tes_score_ci_upper']:.3f}]")
print(f"Momentum Drift: {results['momentum_drift']:.2e}")
print(f"Prediction Horizon: {results['horizon']:.0f} steps")
```

---

## Out-of-Distribution Testing

### Test on Different Numbers of Bodies

```python
# Test zero-shot generalization to N=4,5 bodies
results = arena.run_track_a_ood_n(model.predict, n_bodies_list=[4,5])

print(f"N=4 TES: {results['n4_tes']:.3f}")
print(f"N=5 TES: {results['n5_tes']:.3f}")
```

### Temporal Extrapolation

```python
# Test extrapolation beyond training horizon
results = arena.run_track_a_ood_t(model.predict, t_pred_train=100, extrapolation_factors=[2,5,10])

print(f"2x Horizon TES: {results['t200_tes']:.3f}")
print(f"5x Horizon TES: {results['t500_tes']:.3f}")
print(f"10x Horizon TES: {results['t1000_tes']:.3f}")
```

---

## Track C: Symbolic Regression

```python
# Define discovered law (e.g., from PySR)
def discovered_law(pos, vel, dt, masses):
    """Implementation of F = G*m1*m2/r^2"""
    n = len(masses)
    acc = np.zeros_like(pos)

    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                acc[i] += masses[j] * r_vec / r**3

    vel += acc * dt
    pos += vel * dt
    return pos, vel

# Run Track C benchmark
results = arena.run_track_c(discovered_law, steps=1000, complexity_penalty=1.0)

print(f"Track C Score: {results['track_c_score']:.4f}")
print(f"Fit Accuracy: {results['fit_accuracy']:.4f}")
print(f"Complexity: {results['complexity']}")
```

---

## Uncertainty Quantification

```python
# Train or load ensemble of 5 models
ensemble = [model1, model2, model3, model4, model5]

def ensemble_predict(pos, vel, dt, steps):
    predictions = [m.predict(pos, vel, dt, steps) for m in ensemble]
    pos_pred = np.mean([p[0] for p in predictions], axis=0)
    vel_pred = np.mean([p[1] for p in predictions], axis=0)
    return pos_pred, vel_pred

# Run with uncertainty quantification
results = arena.run_track_a_with_uncertainty(
    ensemble_predict,
    t_pred=100,
    n_ensemble=5,
    uncertainty_type="ensemble"
)

if results['uncertainty_calibrated']:
    print(f"✓ Calibrated uncertainty! 1.2x multiplier applied")
    print(f"Final TES: {results['tes_score']:.3f}")
else:
    print(f"✗ Uncertainty not calibrated")
    print(f"Base TES: {results['base_tes']:.3f}")
```

---

## Benchmark Orchestration

### Run Multiple Models in Parallel

```python
from scripts.benchmark_runner import BenchmarkTask, run_benchmark_suite

tasks = [
    BenchmarkTask(
        name="LinearBaseline",
        model_fn=LinearBaseline().predict,
        track="A",
        dataset="data/val.h5",
        model_details={"type": "Heuristic", "Algorithm": "Linear Extrapolation"}
    ),
    BenchmarkTask(
        name="MyMLP",
        model_fn=mlp_model.predict,
        track="A",
        dataset="data/val.h5",
        model_details={"type": "Neural Network", "Architecture": "MLP"}
    ),
]

# Run benchmarks in parallel (4 workers)
summary = run_benchmark_suite(
    tasks,
    parallel=True,
    robust=True,        # Use multi-trajectory testing
    n_trajectories=10,
    max_workers=4
)

print(f"Successful: {summary['successful']}/{summary['total_tasks']}")
```

---

## Generate Leaderboard and Visualizations

### HTML Leaderboard

```python
from trimandala.leaderboard import generate_leaderboard

# Generate interactive HTML dashboard
generate_leaderboard("reports", "reports/leaderboard.html")
print("Open reports/leaderboard.html in your browser")
```

### Pareto Frontier Plots

```python
from scripts.visualize_benchmarks import plot_pareto_frontier

# Generate Pareto frontier for Track A
plot_pareto_frontier(
    results,
    track="TrackA",
    save_path="reports/pareto_frontier.png"
)

# Generate all visualizations
from scripts.visualize_benchmarks import generate_all_visualizations
generate_all_visualizations("reports")
```

---

## Training with Energy Tracking

```python
# Train with automatic energy tracking
python scripts/train_mlp.py

# This generates:
# - models/mlp_baseline.pt (model weights)
# - models/mlp_baseline_metadata.json (training info + emissions)
# - metrics/mlp_training_emissions.csv (detailed emissions log)
```

**Metadata Example**:
```json
{
  "model": "SimpleMLP",
  "hidden_size": 512,
  "epochs": 50,
  "emissions_kg_co2": 0.0123,
  "emissions_tracked": true,
  "best_val_loss": 1.23e-5
}
```

---

## Complete Workflow Example

```bash
# 1. Train models with energy tracking
python scripts/train_mlp.py
python scripts/train_lstm.py

# 2. Run robust benchmarks
python scripts/benchmark_runner.py

# 3. Generate leaderboard
python -m trimandala.leaderboard

# 4. Generate visualizations
python scripts/visualize_benchmarks.py

# 5. View results
open reports/leaderboard.html
```

---

## Model Registry (JSON Format)

Create `scripts/model_registry.json`:

```json
{
  "models": [
    {
      "name": "LinearBaseline",
      "track": "A",
      "module": "trimandala.baselines.linear",
      "class": "LinearBaseline",
      "kwargs": {},
      "model_details": {
        "type": "Heuristic",
        "Algorithm": "Linear Extrapolation",
        "Parameters": "0"
      }
    },
    {
      "name": "MyCustomModel",
      "track": "A",
      "module": "my_module",
      "class": "MyCustomModel",
      "kwargs": {"hidden_size": 256},
      "model_details": {
        "type": "Neural Network",
        "Architecture": "Custom"
      }
    }
  ]
}
```

Then run:

```python
from scripts.benchmark_runner import load_model_registry, run_benchmark_suite

tasks = load_model_registry("scripts/model_registry.json")
summary = run_benchmark_suite(tasks, parallel=True)
```

---

## Advanced: Custom Benchmarking

### Custom Evaluation Loop

```python
from trimandala.arena import Arena

arena = Arena("data/val.h5")

# Run custom OOD testing
for n_bodies in [3, 4, 5, 6]:
    dataset_path = f"data/val_n{n_bodies}.h5"
    if os.path.exists(dataset_path):
        arena_n = Arena(dataset_path)
        results = arena_n.run_track_a(model.predict, t_pred=100)
        print(f"N={n_bodies}: TES={results['tes_score']:.3f}")
```

### Custom Metrics

```python
# Extract additional metrics from robust results
results = arena.run_track_a_robust(model.predict, t_pred=100, n_trajectories=10)

print(f"Mean TES: {results['tes_score']:.3f}")
print(f"Std TES: {results['tes_score_std']:.3f}")
print(f"CI Lower: {results['tes_score_ci_lower']:.3f}")
print(f"CI Upper: {results['tes_score_ci_upper']:.3f}")
print(f"Momentum Drift: {results['momentum_drift']:.2e}")
print(f"Prediction Horizon: {results['horizon']:.0f} steps")
```

---

## Troubleshooting

### CodeCarbon Issues on macOS

```python
# CodeCarbon is automatically disabled on macOS
# To enable (requires sudo), modify in arena.py:
if sys.platform == "darwin":
    HAS_CODECARBON = True  # Change from False
```

### Missing Datasets for OOD Testing

```bash
# Generate datasets for different N values
python scripts/generate_benchmark_data.py --n_bodies 4 --output data/val_n4.h5
python scripts/generate_benchmark_data.py --n_bodies 5 --output data/val_n5.h5
```

### Slow Benchmarks

```python
# Reduce number of trajectories
results = arena.run_track_a_robust(model.predict, t_pred=100, n_trajectories=5)

# Use sequential execution instead of parallel
summary = run_benchmark_suite(tasks, parallel=False)
```

---

## Next Steps

- Read [BENCHMARK_IMPROVEMENTS.md](./BENCHMARK_IMPROVEMENTS.md) for detailed documentation
- Check [BENCHMARK.md](./BENCHMARK.md) for protocol specifications
- Explore `examples/` directory for more examples
- Run `python -m trimandala.leaderboard` to see current rankings

---

## Tips

1. **Start with robust benchmarks**: Use `run_track_a_robust()` for reliable results
2. **Track energy**: Always train with `codecarbon` enabled
3. **Use uncertainty quantification**: Calibrated ensembles get 1.2x score multiplier
4. **Check OOD performance**: Models should generalize to different N
5. **Generate reports**: Use leaderboard and visualizations for easy comparison
6. **Monitor prediction horizon**: How long until model diverges?

---

Happy Benchmarking! 🚀

# Trimandala

**Trimandala** is a research-grade AI benchmarking framework designed to rigorously evaluate the capability of Artificial Intelligence models in solving the **N-body problem**.

> **"The Three-Body Problem is not unsolvable. It is chaotic."**

## Mission

To provide a **Multi-Modal Arena** for benchmarking AI capabilities in:

1.  **Math/Physics**: Can the AI learn chaotic dynamics and conservation laws? (Neural Surrogates)
2.  **Coding/Engineering**: Can the AI _implement_ a high-precision integrator given the physics equations? (Code Generation)
3.  **Research/Discovery**: Can the AI find new stable periodic orbits (like the Figure-8) or rediscover the Law of Gravity? (Symbolic Regression)

**Why N-Body?**
It is one of the few problems that is **Hard** (Chaos), **Verifiable** (Energy Conservation), and **Open-Ended** (Infinite periodic solutions).

## Key Features

### 1. The Challenge Suite

- **Track A (Surrogate)**: Minimize prediction error on chaotic trajectories.
- **Track B (Engineer)**: "Here is the Hamiltonian. Write a Python function `step(pos, vel, dt)` that minimizes Energy Drift."
- **Track C (Scientist)**: "Given these trajectories, output the symbolic equation $F = G \frac{m_1 m_2}{r^2}$."

### 2. High-Performance Core (The Oracle)

- **C++20 Symplectic Kernel**: Provides the ground-truth against which AI Code/Math is judged.
- **Precision**: $< 10^{-9}$ energy drift.
- **Python Bindings**: Seamless `pybind11` integration.

### 2. Streaming Architecture (Big Data)

- **Zero-RAM Footprint**: Uses `HDF5` streaming to handle multi-terabyte trajectory datasets on standard 16GB laptops.
- **Chunked I/O**: Efficient training computations without loading full history.

### 3. Visualization Suite

- **Scientific**: `Matplotlib` pipelines for Scientific Plots.
- **Cinematic**: `Blender` export for Hollywood-style renders.

## The Benchmark

Strict rules are defined in [BENCHMARK.md](./BENCHMARK.md).

- **Metric**: `Trimandala Efficiency Score (TES)` (Log-weighted composite of Accuracy, Physics, Speed).
- **Anti-Cheat**: Strict ban on physics constants leaking.
- **Green AI**: Mandatory energy consumption reporting.

## Installation

```bash
pip install trimandala
```

(Requires `cmake` and `clang++`/`g++` for compilation)

## Usage

```python
import trimandala as tri

# 1. Generate Ground Truth (Streaming)
tri.generate(
    n_bodies=3,
    steps=1_000_000,
    output="dataset.h5",
    integrator="yoshida4"
)

# 2. Benchmark your Model
results = tri.benchmark(
    model=MyNeuralODE(),
    dataset="dataset.h5"
)
print(results.energy_error)
```

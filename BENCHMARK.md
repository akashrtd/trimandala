# The Trimandala Benchmark Protocol

> **Integrity Notice**: This benchmark measures **Generalization**, not Memorization. Any attempt to hardcode physics constants, leak test data into training, or bypass the simulation sandbox will result in a **DISQUALIFICATION**.

## Overview

Trimandala evaluates AI capabilities across three cognitive dimensions: **Approximation (Math)**, **Implementation (Code)**, and **Discovery (Research)**.

---

---

## Technical Specifications

### 1. Standardization & Units

All simulations use **Normalized Units** (N-body Units):

- **Gravitational Constant**: $G = 1$
- **Total Mass**: $M_{total} = 1$
- **Scale Radius**: $R = 1$ (Virial Radius)
- **AI Implication**: Do NOT assume SI units ($m/s$).

### 2. Dataset Schema (HDF5)

The ground truth `dataset.h5` contains:

- `["positions"]`: Shape `(steps, n_bodies, 3)`, dtype `float64`
- `["velocities"]`: Shape `(steps, n_bodies, 3)`, dtype `float64`
- `["masses"]`: Shape `(n_bodies,)`, dtype `float64`
- `["time"]`: Shape `(steps,)`, dtype `float64`

### 3. Baseline Targets

To count as "successful", an AI must beat these baselines:

- **Track A (Surrogate)**: Must outperform `Linear Extrapolation` (zero acceleration assumption) in MSE.
- **Track B (Engineer)**: Must outperform `scipy.integrate.odeint` (LSODA) in Wall-clock time or Energy Conservation.

---

## Track A: The Neural Surrogate (Math)

**Goal**: Train a neural network $\mathcal{N}$ to approximate the true time-evolution operator $\Phi_t$ of a chaotic 3-body system.

### The Task

- **Input**: State vectors at $t=0, \dots, t_{obs}$ (Positions $q$, Velocities $p$).
- **Output**: Predicted state at $t_{obs} + t_{pred}$.
- **Constraint**: The model must operate faster than the numerical integrator (Speedup factor $> 1$).

### Scoring Metric

### Scoring Metric: Trimandala Efficiency Score (TES)

To avoid scaling issues and properly reward orders-of-magnitude improvements, we use a **Log-Weighted Composite Score**:

$$ TES = w*{acc} \cdot (-\log*{10}(\epsilon*{MSE})) + w*{phys} \cdot (-\log*{10}(\delta*{Energy})) + w*{speed} \cdot \log*{10}(\text{Speedup}) $$

**Weights**:

- $w_{acc} = 0.4$ (Trajectory fidelity)
- $w_{phys} = 0.4$ (Conservation laws)
- $w_{speed} = 0.2$ (Inference efficiency)

**Baseline**: A score of 0 corresponds to the `Linear Baseline`.
**Constraint**: Any run with $\delta_{Energy} > 0.1$ is marked **Did Not Finish (DNF)**.

### Pareto Frontier Requirement

Single scores hide trade-offs. You must plot your model on the **Accuracy-Latency Plane**:

- X-Axis: Inference Steps / Second (Log scale)
- Y-Axis: Energy Conservation Error (Log scale, inverted)
  _Dominant models (top-right) are considered State-of-the-Art._

### 🚫 Anti-Cheat & Malpractice Rules

1.  **Strict Holdout**: You are provided a `training.h5` and `validation.h5`. The final score is calculated on a **hidden** `test.h5` generated with different initial conditions and slightly perturbed masses.
2.  **No Physics Leaking**: The model must _learn_ the physics. You may NOT provide the gravitational constant $G$ or the equations of motion as explicit input features unless specified.
3.  **Stability Check**: Models that drift to infinity (NaN) receive a score of 0.

---

## Track B: The Engineer (Coding)

**Goal**: Design and implement a high-precision N-body integrator in Python.

### The Task

**Prompt**: _"Write a Python function `step(pos, vel, mass, dt)` that integrates the N-body equations of motion. Maximize accuracy and minimize energy drift."_

### Scoring Metric

The generated code is executed against the C++ Oracle.
$$ S_B = \text{Accuracy} \times \text{Speedup} $$

- **Accuracy**: $1 / \text{EnergyDrift}$ at $t=1000$.
- **Speedup**: Time taken relative to a naive Python loop.

### 🚫 Anti-Cheat & Malpractice Rules

1.  **Primitive Ops Only**: The code must use `numpy` or basic arithmetic. You CANNOT `import scipy.integrate` or `import rebound`.
2.  **No Hardcoding**: You cannot carry pre-computed trajectories in the code.
3.  **Readability**: Obfuscated code or binary blobs are disqualified.

---

## Track C: The Researcher (Discovery)

**Goal**: Rediscover physical laws from data (Symbolic Regression).

### The Task

**Input**: A dataset of planetary motions $(q, p, t)$.
**Output**: A symbolic expression for the force law (e.g., `F = G*m1*m2/r^2`).

### Scoring Metric

$$ S_C = \text{Fit Accuracy} - \text{Complexity Penalty} $$

- **Fit Accuracy**: $R^2$ on unseen data.
- **Complexity Penalty**: Length of the symbolic expression. (Occam's Razor: simpler is better).

### 🚫 Anti-Cheat & Malpractice Rules

1.  **Symbolic Integrity**: The output must be a mathematical expression, not a black-box function.
2.  **Zero Prior Knowledge**: The agent is NOT told it is observing gravity. It could be Coulomb's law or a Lennard-Jones potential. It must deduce the form from data alone.

---

---

## Best Practices & Rigor Checklist (NeurIPS Standard)

This benchmark closely follows the _NeurIPS Datasets and Benchmarks_ track guidelines.

### 1. Out-of-Distribution (OOD) Testing

Physics implies universality. A model trained on 3 bodies should work on 4.

- **Level 1 (IID)**: Test on unseen initial conditions (Same $N=3$).
- **Level 2 (OOD-N)**: Test on $N=4, 5$ bodies (Zero-shot generalization).
- **Level 3 (OOD-T)**: Extrapolate to $t > 10 \times t_{train}$.

### 2. Uncertainty Quantification

Scientific predictions require error bars.

- Models providing **calibrated uncertainty** (e.g., via ensembles or Bayesian NN) receive a **1.2x Score Multiplier**.
- Uncertainty is evaluated via **Negative Log Likelihood (NLL)**.

### 3. Green AI Metrics

We measure the cost of intelligence.

- **Training Energy**: Joules consumed during training.
- **Inference Energy**: Joules per simulation step.
- _Reporting_: All submissions must include the `carbon.txt` generated by `codecarbon`.

### 4. Robustness: The Butterfly Score

Chaotic systems are sensitive to initial conditions. AI models must capture this _exact_ sensitivity, not more, not less.

- **Protocol**: Perturb initial position by $\delta = 10^{-10}$.
- **Metric**: The divergence rate should match the true Lyapunov Exponent $\lambda$ of the system.
- **Penalty**: Over-smoothing (stable when it should be chaotic) is penalized.

### 5. Data Efficiency (The "Smartness" Metric)

Intelligence is learning from few examples.

- **Protocol**: Train on subsets: 100, 1k, 10k, 1M samples.
- **Metric**: Area Under the Curve (AUC) of $Accuracy \times \log(\text{Samples})$.
- _Goal_: Solve the 3-body problem with minimal observations.

## Submission Guidelines

All results must be reproducible.

1.  **Seed EVERYTHING**: PyTorch seeds, Numpy seeds, Python hashing seeds.
2.  **Datasheets**: Fill out the `datasheet.md` describing model architecture and training data lineage.
3.  **Containerization**: Submit a Docker container to ensure environment reproducibility.

import concurrent.futures
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from trimandala.arena import Arena
from trimandala.reporting import ReportCard


@dataclass
class BenchmarkTask:
    """A single benchmark task to run"""

    name: str
    model_fn: Callable
    track: str  # "A", "B", or "C"
    dataset: str
    model_details: Dict
    kwargs: Optional[Dict] = None


class BenchmarkRunner:
    """
    Benchmark Orchestration System.
    Runs multiple benchmarks with parallel execution, timeout handling, and retry logic.
    """

    def __init__(self, max_workers: int = 4, timeout: int = 300, max_retries: int = 2):
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.results: List[Dict] = []
        self.failures: List[Dict] = []

    def run_single(self, task: BenchmarkTask, retry_count: int = 0) -> Optional[Dict]:
        """
        Run a single benchmark task with timeout handling.
        """
        print(
            f"\n[{task.name}] Running Track {task.track} (attempt {retry_count + 1}/{self.max_retries})..."
        )

        try:
            arena = Arena(task.dataset)

            # Run appropriate track
            if task.track == "A":
                kwargs = task.kwargs or {"t_pred": 100}
                result = arena.run_track_a(task.model_fn, **kwargs)
            elif task.track == "B":
                kwargs = task.kwargs or {"steps": 1000}
                result = arena.run_track_b(task.model_fn, **kwargs)
            elif task.track == "C":
                kwargs = task.kwargs or {"steps": 1000}
                result = arena.run_track_c(task.model_fn, **kwargs)
            else:
                raise ValueError(f"Unknown track: {task.track}")

            # Generate report card
            card = ReportCard(task.name, f"Track{task.track}", task.model_details)
            card.add_result(result)
            card.save()

            result["task_name"] = task.name
            result["track"] = task.track
            result["success"] = True

            print(f"[{task.name}] ✓ TES: {result['tes_score']:.3f}")
            return result

        except Exception as e:
            print(f"[{task.name}] ✗ Error: {str(e)}")

            if retry_count < self.max_retries - 1:
                print(f"[{task.name}] Retrying...")
                time.sleep(1)
                return self.run_single(task, retry_count + 1)
            else:
                error_info = {
                    "task_name": task.name,
                    "track": task.track,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                self.failures.append(error_info)
                return None

    def run_tasks(
        self, tasks: List[BenchmarkTask], parallel: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run multiple benchmark tasks, optionally in parallel.
        """
        self.results = []
        self.failures = []

        print("=" * 60)
        print(f"Benchmark Runner: {len(tasks)} tasks")
        print(f"Mode: {'Parallel' if parallel else 'Sequential'}")
        print(f"Workers: {self.max_workers if parallel else 1}")
        print("=" * 60)

        start_time = time.time()

        if parallel:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_task = {
                    executor.submit(self.run_single, task): task for task in tasks
                }

                for future in concurrent.futures.as_completed(
                    future_to_task, timeout=self.timeout * len(tasks)
                ):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            self.results.append(result)
                    except Exception as e:
                        print(f"[{task.name}] Timeout or exception: {e}")
                        self.failures.append(
                            {
                                "task_name": task.name,
                                "track": task.track,
                                "success": False,
                                "error": str(e),
                            }
                        )
        else:
            for task in tasks:
                result = self.run_single(task)
                if result:
                    self.results.append(result)

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print(f"Benchmark Complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Successful: {len(self.results)}/{len(tasks)}")
        print(f"Failed: {len(self.failures)}")
        print("=" * 60)

        return self.results, self.failures

    def run_robust_benchmarks(
        self,
        tasks: List[BenchmarkTask],
        n_trajectories: int = 10,
        parallel: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run robust benchmarks with multi-trajectory testing.
        """
        print("=" * 60)
        print(f"Robust Benchmark Runner: {len(tasks)} tasks")
        print(f"Multi-trajectory testing: {n_trajectories} trajectories each")
        print("=" * 60)

        # Create robust versions of tasks
        robust_tasks = []
        for task in tasks:
            robust_task = BenchmarkTask(
                name=task.name + "_robust",
                model_fn=task.model_fn,
                track=task.track,
                dataset=task.dataset,
                model_details=task.model_details,
                kwargs={"t_pred": 100, "n_trajectories": n_trajectories}
                if task.track == "A"
                else task.kwargs,
            )
            robust_tasks.append(robust_task)

        return self.run_tasks(robust_tasks, parallel=parallel)

    def save_summary(self, output_path: str = "reports/benchmark_summary.json"):
        """Save a summary of all benchmark results"""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": len(self.results) + len(self.failures),
            "successful": len(self.results),
            "failed": len(self.failures),
            "results": self.results,
            "failures": self.failures,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {output_path}")
        return summary


def load_model_registry(
    registry_path: str = "scripts/model_registry.json",
) -> List[BenchmarkTask]:
    """
    Load model registry and create benchmark tasks.

    Registry format:
    {
        "models": [
            {
                "name": "MyModel",
                "track": "A",
                "module": "my_module",
                "class": "MyModelClass",
                "kwargs": {...},
                "model_details": {...}
            },
            ...
        ]
    }
    """
    if not os.path.exists(registry_path):
        print(f"Warning: Model registry not found at {registry_path}")
        return []

    with open(registry_path, "r") as f:
        registry = json.load(f)

    tasks = []
    for model_info in registry.get("models", []):
        try:
            # Dynamic import
            module = __import__(model_info["module"], fromlist=[model_info["class"]])
            model_class = getattr(module, model_info["class"])

            # Initialize model
            model_kwargs = model_info.get("kwargs", {})
            model = model_class(**model_kwargs)

            # Get predict function
            predict_fn = model.predict if hasattr(model, "predict") else model.step

            task = BenchmarkTask(
                name=model_info["name"],
                model_fn=predict_fn,
                track=model_info["track"],
                dataset=model_info.get("dataset", "data/val.h5"),
                model_details=model_info.get("model_details", {}),
                kwargs=model_info.get("benchmark_kwargs", {}),
            )
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load model {model_info['name']}: {e}")

    return tasks


def run_benchmark_suite(
    tasks: List[BenchmarkTask],
    parallel: bool = True,
    robust: bool = False,
    n_trajectories: int = 10,
    max_workers: int = 4,
) -> Dict:
    """
    Run a complete benchmark suite.

    Args:
        tasks: List of BenchmarkTask objects
        parallel: Whether to run in parallel
        robust: Whether to use multi-trajectory robust testing
        n_trajectories: Number of trajectories for robust testing
        max_workers: Number of parallel workers

    Returns:
        Summary dict with results and failures
    """
    runner = BenchmarkRunner(max_workers=max_workers)

    if robust:
        results, failures = runner.run_robust_benchmarks(
            tasks, n_trajectories, parallel
        )
    else:
        results, failures = runner.run_tasks(tasks, parallel)

    summary = runner.save_summary()

    # Generate leaderboard
    from trimandala.leaderboard import Leaderboard

    leaderboard = Leaderboard("reports")
    leaderboard.generate_html_dashboard()

    # Generate visualizations
    from scripts.visualize_benchmarks import generate_all_visualizations

    generate_all_visualizations("reports")

    return summary


if __name__ == "__main__":
    # Example usage
    from trimandala.baselines.linear import LinearBaseline
    from trimandala.baselines.mlp import SimpleMLP
    import torch

    # Create sample tasks
    tasks = [
        BenchmarkTask(
            name="LinearBaseline",
            model_fn=LinearBaseline().predict,
            track="A",
            dataset="data/val.h5",
            model_details={
                "type": "Heuristic",
                "Algorithm": "Linear Extrapolation",
                "Parameters": "0",
            },
        ),
    ]

    # Load MLP if available
    if os.path.exists("models/mlp_baseline.pt"):
        mlp_model = SimpleMLP(n_bodies=3)
        mlp_model.load_state_dict(torch.load("models/mlp_baseline.pt"))
        tasks.append(
            BenchmarkTask(
                name="MLPBaseline",
                model_fn=mlp_model.predict,
                track="A",
                dataset="data/val.h5",
                model_details={
                    "type": "Neural Network",
                    "Architecture": "SimpleMLP",
                    "Hidden Size": 512,
                    "Layers": 3,
                },
            )
        )

    # Run benchmarks
    summary = run_benchmark_suite(tasks, parallel=True, robust=True)

    print("\n✓ Benchmark suite complete!")

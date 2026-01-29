from trimandala.arena import Arena
from trimandala.baselines.linear import LinearBaseline
import json

def main():
    print("=== Trimandala Benchmark: Baselines ===")
    
    # 1. Setup Arena
    dataset = "data/val.h5"
    arena = Arena(dataset)
    
    # 2. Linear Baseline
    print("\n[Evaluating Linear Baseline]")
    model = LinearBaseline()
    
    # Predict 100 steps into the future
    results = arena.run_track_a(model.predict, t_pred=100)
    
    print(json.dumps(results, indent=2))
    
    if results['tes_score'] < 1.0:
        print("Verdict: Baseline is weak (Expected).")
    else:
        print("Verdict: Baseline is surprisingly good?")

    # 3. MLP Baseline
    print("\n[Evaluating MLP Baseline]")
    from trimandala.baselines.mlp import SimpleMLP
    import torch
    
    mlp_model = SimpleMLP(n_bodies=3)
    mlp_model.load_state_dict(torch.load("models/mlp_baseline.pt"))
    
    results_mlp = arena.run_track_a(mlp_model.predict, t_pred=100)
    print(json.dumps(results_mlp, indent=2))
    
    # Comparison
    print("\n=== Summary ===")
    print(f"Linear TES: {results['tes_score']:.2f} | Lyapunov: {results.get('lyapunov', 0):.2e} | Drift: {results['drift']:.2e}")
    print(f"MLP TES:    {results_mlp['tes_score']:.2f} | Lyapunov: {results_mlp.get('lyapunov', 0):.2e} | Drift: {results_mlp['drift']:.2e}")
    if 'emissions_kg' in results_mlp:
        print(f"MLP Emissions: {results_mlp['emissions_kg']:.2e} kg CO2eq")

    # 4. LSTM Baseline (Challenger)
    print("\n[Evaluating LSTM Challenger]")
    from trimandala.baselines.lstm import LSTMBaseline
    
    lstm_model = LSTMBaseline(n_bodies=3, hidden_size=256, num_layers=2)
    lstm_model.load_state_dict(torch.load("models/lstm_baseline.pt"))
    
    results_lstm = arena.run_track_a(lstm_model.predict, t_pred=100)
    print(json.dumps(results_lstm, indent=2))
    
    # Comparison
    print("\n=== Summary ===")
    print(f"Linear TES: {results['tes_score']:.2f} | Lyndon: {results.get('lyapunov', 0):.2e}")
    print(f"MLP TES:    {results_mlp['tes_score']:.2f} | Lyndon: {results_mlp.get('lyapunov', 0):.2e}")
    print(f"LSTM TES:   {results_lstm['tes_score']:.2f} | Lyndon: {results_lstm.get('lyapunov', 0):.2e}")

    # 5. Generate Report Cards (With Deep Details)
    print("\n[Generating Reports]")
    from trimandala.reporting import ReportCard
    
    # Linear
    details_linear = {
        "type": "Heuristic",
        "Algorithm": "Linear Extrapolation",
        "Precision": "Float64",
        "Parameters": "0"
    }
    card = ReportCard("LinearBaseline", "TrackA", details_linear)
    card.add_result(results)
    path = card.save()
    print(f"Generated: {path}")
    
    # MLP
    details_mlp = {
        "type": "Neural Network",
        "Architecture": "SimpleMLP (Flattened)",
        "Hidden Size": 512,
        "Layers": 3,
        "Parameters": "Adjustable",
        "Precision": "Float32"
    }
    card = ReportCard("SimpleMLP", "TrackA", details_mlp)
    card.add_result(results_mlp)
    path = card.save()
    print(f"Generated: {path}")
    
    # LSTM
    details_lstm = {
        "type": "Recurrent Neural Network",
        "Architecture": "LSTM (Seq2Seq)",
        "Hidden Size": 256,
        "Layers": 2,
        "History": "20 Steps",
        "Precision": "Float32"
    }
    card = ReportCard("LSTMBaseline", "TrackA", details_lstm)
    card.add_result(results_lstm)
    path = card.save()
    print(f"Generated: {path}")

if __name__ == "__main__":
    main()

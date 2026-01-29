import json
import os
from datetime import datetime
from typing import Dict, Any

class ReportCard:
    def __init__(self, model_name: str, track: str, model_details: Dict[str, Any] = None):
        self.model_name = model_name
        self.track = track
        self.timestamp = datetime.now().isoformat()
        self.results = {}
        self.model_details = model_details or {}
        
    def add_result(self, metrics: Dict[str, Any]):
        self.results = metrics
        
    def compute_grade(self) -> str:
        tes = self.results.get('tes_score', 0.0)
        drift = self.results.get('drift', 1.0)
        lyapunov = self.results.get('lyapunov', 0.0)
        
        # Grading Logic
        # S: TES > 4.0 (Superhuman)
        # A: TES > 3.5 (Beats Linear)
        # B: TES > 2.5 (Functional)
        # C: TES > 1.0 (Weak)
        # F: TES < 1.0 or Drift > 10%
        
        if drift > 0.1: return "F (Unstable)"
        if tes > 4.0: return "S (State of the Art)"
        if tes > 3.5: return "A (Excellent)"
        if tes > 2.5: return "B (Good)"
        if tes > 1.0: return "C (Fair)"
        return "F (Poor)"

    def compute_normalized_score(self) -> int:
        """
        Maps TES (0-5) to a 0-100 scale based on Track.
        """
        tes = self.results.get('tes_score', 0.0)
        drift = self.results.get('drift', 1.0)
        
        # Base conversion: TES * 20
        # TES 5.0 -> 100 (Gold Standard)
        # TES 2.5 -> 50 (Passable)
        
        raw_score = tes * 20.0
        
        # Penalties
        if drift > 0.1: # Physics violation
            raw_score *= 0.5 
            
        return min(100, int(raw_score))

    def save(self, output_dir="reports"):
        # Model-specific folder
        model_dir = os.path.join(output_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        grade = self.compute_grade()
        score_100 = self.compute_normalized_score()
        
        # Determine Category Name
        category_map = {
            "TrackA": "Math Sc", 
            "TrackB": "Code Gen Sc", 
            "TrackC": "Research Sc"
        }
        category = category_map.get(self.track, "Score")
        
        # 1. JSON Report (Updated metadata)
        report_data = {
            "metadata": {
                "model": self.model_name,
                "track": self.track,
                "timestamp": self.timestamp,
                "grade": grade,
                "score_100": score_100,
                "category": category,
                "details": self.model_details
            },
            "metrics": self.results
        }
        
        json_path = os.path.join(model_dir, f"{self.track}_score.json")
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # 2. Markdown Certificate (Path per track)
        cert_path = os.path.join(model_dir, f"{self.track}_certificate.md")
        with open(cert_path, 'w') as f:
            f.write(f"# 📜 Trimandala Report Card\n\n")
            f.write(f"**Model**: `{self.model_name}`\n")
            f.write(f"**Track**: `{self.track}`\n")
            f.write(f"**Date**: `{self.timestamp}`\n")
            
            # Big Score Display
            f.write(f"## **{category}**: {score_100}/100\n")
            f.write(f"**Competency Grade**: {grade}\n\n")
            f.write(f"### 📊 Metrics\n")
            f.write(f"| Metric | Value | Unit |\n")
            f.write(f"| :--- | :--- | :--- |\n")
            f.write(f"| **TES Score** | **{self.results.get('tes_score', 0.0):.2f}** | Points |\n")
            f.write(f"| MSE | {self.results.get('mse', 0.0):.2e} | - |\n")
            f.write(f"| Latency | {self.results.get('latency', 0.0):.4f} | s |\n")
            f.write(f"| Speedup | {self.results.get('speedup', 1.0):.1f}x | vs C++ |\n")
            f.write(f"### 🔬 Rigor Rechecks\n")
            f.write(f"- **Physics Drift**: `{self.results.get('drift', 0.0):.2%}`\n")
            f.write(f"- **Lyapunov Info**: `{self.results.get('lyapunov', 0.0):.2e}`\n")
            if 'emissions_kg' in self.results:
                f.write(f"- **Carbon Footprint**: `{self.results['emissions_kg']:.2e} kg CO2`\n")
                
        # 3. Model Details (Shared or per track? Let's assume shared details per model artifact)
        details_path = os.path.join(model_dir, "model_details.md")
        # Only write if details exist
        if self.model_details:
            with open(details_path, 'w') as f:
                f.write(f"# 🧠 Model Architecture: {self.model_name}\n\n")
                
                # General Type
                m_type = self.model_details.get("type", "Unknown")
                f.write(f"**Type**: `{m_type}`\n\n")
                
                # Parameters Table
                f.write(f"## Specifications\n")
                f.write(f"| Property | Value |\n")
                f.write(f"| :--- | :--- |\n")
                for k, v in self.model_details.items():
                    if k != "type":
                         f.write(f"| **{k}** | {v} |\n")
                         
        return cert_path

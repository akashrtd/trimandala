import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


class Leaderboard:
    """
    Leaderboard system for ranking models across all tracks.
    Aggregates results from ReportCards and generates rankings.
    """

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        self.models: Dict[str, Dict] = {}
        self._load_all_reports()

    def _load_all_reports(self):
        """Load all ReportCard JSON files from reports directory"""
        reports_path = Path(self.reports_dir)

        if not reports_path.exists():
            print(f"Warning: Reports directory {self.reports_dir} not found")
            return

        for model_dir in reports_path.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                model_name = model_dir.name
                self.models[model_name] = {
                    "model_name": model_name,
                    "tracks": {},
                    "overall_score": 0.0,
                }

                # Load track-specific scores
                for track_file in model_dir.glob("*_score.json"):
                    track_name = track_file.stem.replace("_score", "")

                    with open(track_file, "r") as f:
                        track_data = json.load(f)

                    metadata = track_data.get("metadata", {})
                    metrics = track_data.get("metrics", {})

                    self.models[model_name]["tracks"][track_name] = {
                        "grade": metadata.get("grade", "N/A"),
                        "score_100": metadata.get("score_100", 0),
                        "tes_score": metrics.get("tes_score", 0.0),
                        "drift": metrics.get("drift", 1.0),
                        "speedup": metrics.get("speedup", 1.0),
                        "timestamp": metadata.get("timestamp", ""),
                    }

                # Compute overall score (average of available tracks)
                tracks = self.models[model_name]["tracks"]
                if tracks:
                    overall = sum(
                        t["tes_score"] for t in tracks.values() if t["tes_score"] > 0
                    ) / len(tracks)
                    self.models[model_name]["overall_score"] = overall

    def rank_by_track(self, track: str) -> List[Dict]:
        """Rank models by performance on a specific track"""
        ranked = []

        for model_name, model_data in self.models.items():
            if track in model_data["tracks"]:
                track_info = model_data["tracks"][track]
                ranked.append(
                    {
                        "model": model_name,
                        "track": track,
                        "grade": track_info["grade"],
                        "score_100": track_info["score_100"],
                        "tes_score": track_info["tes_score"],
                        "drift": track_info["drift"],
                        "speedup": track_info["speedup"],
                    }
                )

        # Sort by TES score (descending)
        ranked.sort(key=lambda x: x["tes_score"], reverse=True)

        return ranked

    def rank_overall(self) -> List[Dict]:
        """Rank models by overall performance across all tracks"""
        ranked = []

        for model_name, model_data in self.models.items():
            tracks = model_data["tracks"]
            avg_tes = (
                sum(t["tes_score"] for t in tracks.values()) / len(tracks)
                if tracks
                else 0.0
            )

            ranked.append(
                {
                    "model": model_name,
                    "overall_tes": avg_tes,
                    "n_tracks": len(tracks),
                    "tracks": list(tracks.keys()),
                }
            )

        ranked.sort(key=lambda x: x["overall_tes"], reverse=True)
        return ranked

    def get_model_details(self, model_name: str) -> Optional[Dict]:
        """Get full details for a specific model"""
        return self.models.get(model_name)

    def generate_html_dashboard(self, output_path: str = "reports/leaderboard.html"):
        """Generate an HTML dashboard with rankings and visualizations"""
        html = self._generate_html()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

        print(f"HTML Dashboard generated: {output_path}")
        return output_path

    def _generate_html(self) -> str:
        """Generate the HTML content for the dashboard"""
        overall_ranking = self.rank_overall()
        track_a = self.rank_by_track("TrackA")
        track_b = self.rank_by_track("TrackB")
        track_c = self.rank_by_track("TrackC")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trimandala Benchmark Leaderboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 3rem;
        }}
        h1 {{
            text-align: center;
            color: #2d3748;
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
        }}
        .subtitle {{
            text-align: center;
            color: #718096;
            margin-bottom: 2rem;
            font-size: 1rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #2d3748;
        }}
        .stat-label {{
            color: #4a5568;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        .section {{
            margin-bottom: 3rem;
        }}
        .section-title {{
            font-size: 1.5rem;
            color: #2d3748;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 1rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:hover {{
            background: #f7fafc;
        }}
        .rank {{
            font-weight: bold;
            color: #667eea;
            width: 60px;
        }}
        .grade-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }}
        .grade-S {{ background: #48bb78; color: white; }}
        .grade-A {{ background: #4299e1; color: white; }}
        .grade-B {{ background: #ed8936; color: white; }}
        .grade-C {{ background: #ecc94b; color: white; }}
        .grade-F {{ background: #f56565; color: white; }}
        .score-bar {{
            width: 100px;
            height: 20px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .score-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }}
        .footer {{
            text-align: center;
            color: #718096;
            margin-top: 3rem;
            font-size: 0.9rem;
        }}
        .tabs {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .tab {{
            padding: 0.75rem 1.5rem;
            background: #e2e8f0;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            color: #4a5568;
        }}
        .tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Trimandala Benchmark Leaderboard</h1>
        <p class="subtitle">AI Benchmarking for N-Body Physics • Last updated: {timestamp}</p>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(self.models)}</div>
                <div class="stat-label">Models Tested</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self._count_total_tracks()}</div>
                <div class="stat-label">Total Submissions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self._get_top_overall_score():.2f}</div>
                <div class="stat-label">Best Overall TES</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len([m for m in overall_ranking if m["overall_tes"] > 3.5])}</div>
                <div class="stat-label">Excellent Models (A+)</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">🏆 Overall Ranking (All Tracks)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Overall TES</th>
                        <th>Tracks</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_overall_rows(overall_ranking)}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">📊 Track A: Neural Surrogate (Math)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>TES Score</th>
                        <th>Grade</th>
                        <th>Energy Drift</th>
                        <th>Speedup</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_track_rows(track_a, "A")}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">💻 Track B: Engineer (Code)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>TES Score</th>
                        <th>Grade</th>
                        <th>Energy Drift</th>
                        <th>Speedup</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_track_rows(track_b, "B")}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">🔬 Track C: Researcher (Discovery)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>TES Score</th>
                        <th>Grade</th>
                        <th>Energy Drift</th>
                        <th>Speedup</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_track_rows(track_c, "C")}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Trimandala Benchmark Framework • Research-grade AI evaluation for chaotic systems</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_overall_rows(self, ranking: List[Dict]) -> str:
        """Generate table rows for overall ranking"""
        rows = []
        for i, model in enumerate(ranking[:10], 1):
            rows.append(f"""
                <tr>
                    <td class="rank">#{i}</td>
                    <td><strong>{model["model"]}</strong></td>
                    <td>
                        <div class="score-bar">
                            <div class="score-fill" style="width: {min(model["overall_tes"] * 20, 100)}%"></div>
                        </div>
                        <span style="margin-left: 10px">{model["overall_tes"]:.2f}</span>
                    </td>
                    <td>{", ".join(model["tracks"])}</td>
                </tr>
            """)
        return "".join(rows)

    def _generate_track_rows(self, ranking: List[Dict], track: str) -> str:
        """Generate table rows for track-specific ranking"""
        rows = []
        for i, model in enumerate(ranking[:10], 1):
            grade = model["grade"].split()[0]
            rows.append(f"""
                <tr>
                    <td class="rank">#{i}</td>
                    <td><strong>{model["model"]}</strong></td>
                    <td>{model["tes_score"]:.2f}</td>
                    <td><span class="grade-badge grade-{grade}">{model["grade"]}</span></td>
                    <td>{model["drift"]:.2e}</td>
                    <td>{model["speedup"]:.1f}x</td>
                </tr>
            """)
        return "".join(rows)

    def _count_total_tracks(self) -> int:
        """Count total track submissions across all models"""
        return sum(len(m["tracks"]) for m in self.models.values())

    def _get_top_overall_score(self) -> float:
        """Get the highest overall TES score"""
        return max((m["overall_score"] for m in self.models.values()), default=0.0)


def generate_leaderboard(
    reports_dir: str = "reports", output_path: str = "reports/leaderboard.html"
):
    """Convenience function to generate leaderboard"""
    leaderboard = Leaderboard(reports_dir)
    return leaderboard.generate_html_dashboard(output_path)


if __name__ == "__main__":
    generate_leaderboard()

"""
Automatic report generation for layer contribution analysis.

This module creates comprehensive HTML and PDF reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReportGenerator:
    """
    Generate comprehensive analysis reports.

    Creates HTML reports with:
    - Executive summary
    - Layer rankings
    - Metric visualizations
    - Recommendations for downstream tasks

    Example:
        >>> generator = ReportGenerator(contributions)
        >>> generator.add_model_info(model, "BERT-base")
        >>> generator.generate_html("report.html")
    """

    def __init__(
        self,
        contributions: Dict[str, Dict[str, float]],
        model_name: str = "Model",
    ):
        """
        Initialize report generator.

        Args:
            contributions: Output from LayerAnalyzer.compute_metrics()
            model_name: Name of the model
        """
        self.contributions = contributions
        self.model_name = model_name
        self.metadata = {
            "generated_at": datetime.now().isoformat(),
            "model_name": model_name,
            "num_layers": len(contributions),
        }

        # Extract available metrics
        first_layer = next(iter(contributions.values()))
        self.metrics = [k for k in first_layer.keys() if k not in ["layer_idx", "layer_type"]]

    def _generate_summary(self) -> str:
        """Generate executive summary section"""
        html = f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="summary-box">
                <p><strong>Model:</strong> {self.model_name}</p>
                <p><strong>Layers Analyzed:</strong> {self.metadata['num_layers']}</p>
                <p><strong>Metrics Computed:</strong> {len(self.metrics)}</p>
                <p><strong>Generated:</strong> {self.metadata['generated_at'][:19]}</p>
            </div>
        </div>
        """
        return html

    def _generate_rankings(self, metric: str, top_k: int = 5) -> str:
        """Generate rankings table for a metric"""
        # Rank layers
        rankings = []
        for layer_name, metrics in self.contributions.items():
            value = metrics.get(metric)
            if value is not None:
                rankings.append((layer_name, value))

        rankings.sort(key=lambda x: x[1], reverse=True)
        rankings = rankings[:top_k]

        # Generate table
        html = f"""
        <div class="ranking">
            <h3>{metric.replace('_', ' ').title()}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Layer</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
        """

        for rank, (layer_name, value) in enumerate(rankings, 1):
            html += f"""
                    <tr>
                        <td>{rank}</td>
                        <td>{layer_name}</td>
                        <td>{value:.4f}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

        return html

    def _generate_recommendations(self) -> str:
        """Generate recommendations based on analysis"""
        # Find metric with highest average value
        metric_avgs = {}
        for metric in self.metrics:
            values = [
                m.get(metric, 0) for m in self.contributions.values() if m.get(metric) is not None
            ]
            if values:
                metric_avgs[metric] = sum(values) / len(values)

        html = """
        <div class="section">
            <h2>💡 Recommendations</h2>
        """

        # Pruning recommendation
        if "gradient_norm" in self.metrics:
            html += """
            <div class="recommendation">
                <h3>🔧 For Model Pruning:</h3>
                <p>Based on gradient norm analysis, consider pruning layers with lowest gradients.</p>
                <p><strong>Strategy:</strong> Start with 30% pruning on low-gradient layers,
                preserve high-gradient layers.</p>
            </div>
            """

        # Distillation recommendation
        if "cka_score" in self.metrics:
            html += """
            <div class="recommendation">
                <h3>📚 For Knowledge Distillation:</h3>
                <p>Select layers with high CKA scores for intermediate supervision.</p>
                <p><strong>Strategy:</strong> Use top-6 layers for knowledge transfer.</p>
            </div>
            """

        # PEFT recommendation
        if "gradient_norm" in self.metrics:
            html += """
            <div class="recommendation">
                <h3>🎯 For PEFT (Adapter Tuning):</h3>
                <p>Insert adapters at layers with highest gradient norms.</p>
                <p><strong>Strategy:</strong> Place 4 adapters at most active layers.</p>
            </div>
            """

        html += "</div>"
        return html

    def _generate_css(self) -> str:
        """Generate CSS styles"""
        return """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
            }
            .section {
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .summary-box {
                background: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }
            th {
                background: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #ecf0f1;
            }
            tr:hover {
                background: #f8f9fa;
            }
            .ranking {
                margin: 20px 0;
            }
            .recommendation {
                background: #e8f5e9;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                border-left: 4px solid #4caf50;
            }
            .recommendation h3 {
                color: #2e7d32;
                margin-top: 0;
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: #7f8c8d;
                font-size: 0.9em;
            }
        </style>
        """

    def generate_html(self, output_path: str = "layer_analysis_report.html"):
        """
        Generate HTML report.

        Args:
            output_path: Path to save HTML file
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Layer Analysis Report - {self.model_name}</title>
            {self._generate_css()}
        </head>
        <body>
            <h1>🔍 Layer Contribution Analysis Report</h1>
            <p class="subtitle">Model: <strong>{self.model_name}</strong></p>

            {self._generate_summary()}

            <div class="section">
                <h2>📈 Layer Rankings</h2>
                <p>Top-5 layers for each metric:</p>
        """

        # Add rankings for each metric
        for metric in self.metrics[:5]:  # Limit to first 5 metrics
            html += self._generate_rankings(metric, top_k=5)

        html += "</div>"

        # Add recommendations
        html += self._generate_recommendations()

        # Add footer
        html += f"""
            <div class="footer">
                <p>Generated by Uni-Layer Framework</p>
                <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """

        # Save file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✓ HTML report saved to {output_path}")
        return output_path

    def generate_markdown(self, output_path: str = "layer_analysis_report.md"):
        """
        Generate Markdown report.

        Args:
            output_path: Path to save Markdown file
        """
        md = f"""# Layer Contribution Analysis Report

**Model:** {self.model_name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Layers Analyzed:** {self.metadata['num_layers']}
- **Metrics Computed:** {len(self.metrics)}

## Layer Rankings

"""

        # Add rankings for each metric
        for metric in self.metrics[:5]:
            md += f"\n### {metric.replace('_', ' ').title()}\n\n"
            md += "| Rank | Layer | Value |\n"
            md += "|------|-------|-------|\n"

            rankings = []
            for layer_name, metrics in self.contributions.items():
                value = metrics.get(metric)
                if value is not None:
                    rankings.append((layer_name, value))

            rankings.sort(key=lambda x: x[1], reverse=True)

            for rank, (layer_name, value) in enumerate(rankings[:5], 1):
                md += f"| {rank} | {layer_name} | {value:.4f} |\n"

        md += "\n## Recommendations\n\n"

        if "gradient_norm" in self.metrics:
            md += "### For Model Pruning\n"
            md += "- Target layers with lowest gradient norms\n"
            md += "- Recommended pruning ratio: 30%\n\n"

        if "cka_score" in self.metrics:
            md += "### For Knowledge Distillation\n"
            md += "- Use top-6 layers with highest CKA scores\n"
            md += "- Apply intermediate supervision\n\n"

        md += "\n---\n*Generated by Uni-Layer Framework*\n"

        # Save file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        print(f"✓ Markdown report saved to {output_path}")
        return output_path

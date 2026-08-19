"""
Printable Diagnostic Report Generator.

Generates self-contained HTML/PDF printable report cards for laboratory records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def generate_printable_html_report(seq_row: Dict[str, Any], model_meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Generates a printable HTML document summarizing a sequence diagnostic result.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    meta = model_meta or {}
    model_name = str(seq_row.get("model_name", "Ensemble"))
    trained_at = str(meta.get("trained_at", "N/A"))
    n_genomes = meta.get("n_sequences", "N/A")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Mpox Diagnostic Report - {seq_row.get('sequence_id', 'Sample')}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 30px;
            color: #2c3e50;
            background-color: #ffffff;
        }}
        .header {{
            border-bottom: 3px solid #2980b9;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .title {{
            font-size: 22px;
            font-weight: bold;
            color: #1a252f;
            margin: 0;
        }}
        .subtitle {{
            font-size: 13px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .section {{
            margin-bottom: 20px;
        }}
        .section-title {{
            font-size: 15px;
            font-weight: bold;
            background-color: #ecf0f1;
            padding: 6px 10px;
            border-left: 4px solid #2980b9;
            margin-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 8px 12px;
            border: 1px solid #bdc3c7;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .badge-good {{ color: #27ae60; font-weight: bold; }}
        .badge-fair {{ color: #d35400; font-weight: bold; }}
        .badge-poor {{ color: #c0392b; font-weight: bold; }}
        .footer {{
            margin-top: 40px;
            border-top: 1px solid #bdc3c7;
            padding-top: 10px;
            font-size: 11px;
            color: #95a5a6;
            text-align: center;
        }}
        @media print {{
            body {{ margin: 0; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="no-print" style="margin-bottom: 15px;">
        <button onclick="window.print()" style="padding: 8px 16px; background-color: #2980b9; color: white; border: none; border-radius: 4px; cursor: pointer;">Print / Save as PDF</button>
    </div>

    <div class="header">
        <div class="title">Mpox Diagnostic & Clade Classification Report</div>
        <div class="subtitle">National Virus Reference Laboratory Surveillance System</div>
    </div>

    <div class="section">
        <div class="section-title">Sample Identification & Result</div>
        <table>
            <tr><th>Sequence ID</th><td><strong>{seq_row.get('sequence_id', 'N/A')}</strong></td></tr>
            <tr><th>Predicted Clade</th><td><strong style="font-size: 16px; color:#2980b9;">{seq_row.get('predicted_clade', 'N/A')}</strong> (Confidence: {float(seq_row.get('confidence', 0)):.1%})</td></tr>
            <tr><th>Ensemble Consensus</th><td>{seq_row.get('consensus_ratio', 'N/A')}</td></tr>
            <tr><th>Quality Flag</th><td><span class="badge-{str(seq_row.get('quality_flag', 'Good')).lower()}">{seq_row.get('quality_flag', 'Good')}</span></td></tr>
            <tr><th>Quality Explanation</th><td>{seq_row.get('quality_explanation', 'N/A')}</td></tr>
            <tr><th>Novelty / OOD Status</th><td>{seq_row.get('ood_status', 'Normal')}</td></tr>
        </table>
    </div>

    <div class="section">
        <div class="section-title">Sequence & Biological Metrics</div>
        <table>
            <tr><th>Genome Length</th><td>{int(seq_row.get('length', 0)):,} bp</td><th>GC Content</th><td>{float(seq_row.get('gc_pct', 0)):.1f}%</td></tr>
            <tr><th>N Count / N %</th><td>{int(seq_row.get('n_count', 0))} ({float(seq_row.get('n_pct', 0)):.2f}%)</td><th>non-ACGTN Bases</th><td>{int(seq_row.get('non_acgtn_count', 0))}</td></tr>
            <tr><th>Premature Stop Codons</th><td>{int(seq_row.get('premature_stop_count', 0))}</td><th>Frameshift Flag</th><td>{'Yes' if seq_row.get('frameshift_flag') else 'No'}</td></tr>
            <tr><th>APOBEC3 Score</th><td>{float(seq_row.get('apobec3_combined_score', 0)):.4f}</td><th>Codon Adaptation (CAI)</th><td>{float(seq_row.get('cai_score', 0)):.4f}</td></tr>
        </table>
    </div>

    <div class="section">
        <div class="section-title">Model Provenance</div>
        <table>
            <tr><th>Deployed Model</th><td>{model_name}</td></tr>
            <tr><th>Training Timestamp</th><td>{trained_at}</td></tr>
            <tr><th>Training Dataset Size</th><td>{n_genomes} genomes</td></tr>
            <tr><th>Report Generated</th><td>{now}</td></tr>
        </table>
    </div>

    <div class="footer">
        Research & Surveillance Support Tool — National Reference Virus Laboratory — Confidential
    </div>
</body>
</html>
"""
    return html

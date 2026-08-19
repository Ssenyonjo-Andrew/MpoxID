"""
Mpox Clade Classifier — Nextclade-style Web UI for Lab Technicians.

Run (from project root):
    streamlit run app/streamlit_app.py

Fully offline once models/deploy_bundle.joblib is present.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Hashable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
import streamlit as st

# Ensure src/, app/, and project root are on path when launched via `streamlit run`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APP = ROOT / "app"
for p in (ROOT, SRC, APP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mpox_clf.inference import MpoxPredictor  # noqa: E402
from mpox_clf.inference.active_learning import log_user_correction  # noqa: E402
from mpox_clf.utils.pdf_report import generate_printable_html_report  # noqa: E402

try:
    from geo_dashboard import render_surveillance_dashboard  # noqa: E402
except ImportError:
    from app.geo_dashboard import render_surveillance_dashboard  # noqa: E402

CLADE_COLORS = {
    "Ia": "#1565c0",
    "Ib": "#7b1fa2",
    "IIa": "#00796b",
    "IIb": "#e65100",
}


@st.cache_resource
def load_predictor() -> MpoxPredictor:
    bundle = ROOT / "models" / "deploy_bundle.joblib"
    return MpoxPredictor(bundle)


def render_nextclade_css():
    st.markdown(
        """
        <style>
        .nextclade-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin-top: 1rem;
            margin-bottom: 2rem;
            overflow-x: auto;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .nextclade-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            background-color: #ffffff;
            color: #212529;
        }
        .nextclade-table th {
            background-color: #2f3640;
            color: #f5f6fa;
            font-weight: 600;
            text-align: center;
            padding: 8px 10px;
            border: 1px solid #404b56;
            white-space: nowrap;
        }
        .nextclade-table td {
            padding: 6px 10px;
            border-bottom: 1px solid #e9ecef;
            vertical-align: middle;
            text-align: center;
        }
        .nextclade-table tr.row-good { background-color: #ffffff; }
        .nextclade-table tr.row-good:hover { background-color: #f1f3f5; }
        .nextclade-table tr.row-fair { background-color: #fff9db; }
        .nextclade-table tr.row-poor { background-color: #ffe3e3; }
        .nextclade-table tr.row-error { background-color: #f8d7da; color: #721c24; }
        
        .qc-pill-box { display: inline-flex; gap: 2px; align-items: center; justify-content: center; }
        .qc-pill {
            display: inline-block; width: 18px; height: 18px; line-height: 18px;
            border-radius: 50%; color: #ffffff; font-weight: 700; font-size: 0.65rem; text-align: center;
        }
        .qc-green { background-color: #2e7d32; }
        .qc-yellow { background-color: #f57c00; }
        .qc-red { background-color: #d32f2f; }
        
        .clade-badge {
            display: inline-block; padding: 2px 8px; border-radius: 12px;
            color: #ffffff; font-weight: 700; font-size: 0.75rem;
        }
        .status-icon { font-size: 0.95rem; font-weight: bold; }
        .status-good { color: #2e7d32; }
        .status-fair { color: #f57c00; }
        .status-poor { color: #d32f2f; }
        .ood-badge {
            background-color: #d32f2f; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.7rem;
        }
        .consensus-badge {
            background-color: #1976d2; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def compute_nextclade_row(row: pd.Series, idx: Hashable | int | str = 0) -> dict:
    n_pct = float(row.get("n_pct", 0.0))
    non_acgtn = int(row.get("non_acgtn_count", 0))
    stops = int(row.get("premature_stop_count", 0))
    length = int(row.get("length", 0))
    frameshift = bool(row.get("frameshift_flag", False))
    quality_flag = str(row.get("quality_flag", "Good"))

    n_color = "qc-green" if n_pct < 1.0 else ("qc-yellow" if n_pct < 5.0 else "qc-red")
    m_color = "qc-green" if non_acgtn == 0 else ("qc-yellow" if non_acgtn < 20 else "qc-red")
    p_color = "qc-green" if stops == 0 else ("qc-yellow" if stops < 3 else "qc-red")
    c_color = "qc-green" if length > 150000 else ("qc-yellow" if length > 50000 else "qc-red")
    f_color = "qc-red" if frameshift else "qc-green"
    s_color = "qc-green" if stops == 0 else "qc-red"

    if quality_flag == "Good":
        icon = ""
        row_cls = "row-good"
    elif quality_flag == "Fair":
        icon = ""
        row_cls = "row-fair"
    else:
        icon = ""
        row_cls = "row-poor"

    clade = str(row.get("predicted_clade", "Unknown"))
    clade_color = CLADE_COLORS.get(clade, "#555555")

    ref_len = 197209
    gaps = max(0, ref_len - length) if length < ref_len else 0
    insertions = max(0, length - ref_len) if length > ref_len else 0
    cov_pct = min(100.0, max(0.0, 100.0 * (length - int(row.get("n_count", 0))) / ref_len))

    consensus = str(row.get("consensus_ratio", "3/3 agree"))
    is_ood = bool(row.get("is_ood", False))
    ood_html = '<span class="ood-badge">Outlier</span>' if is_ood else '<span style="color:#2e7d32; font-weight:bold;">Normal</span>'

    return {
        "index": idx,
        "icon": icon,
        "row_class": row_cls,
        "seq_id": str(row.get("sequence_id", f"Seq_{idx}")),
        "n_color": n_color,
        "m_color": m_color,
        "p_color": p_color,
        "c_color": c_color,
        "f_color": f_color,
        "s_color": s_color,
        "clade": clade,
        "clade_color": clade_color,
        "confidence": float(row.get("confidence", 0.0)),
        "consensus": consensus,
        "ood_html": ood_html,
        "mutations": int(row.get("n_count", 0)) + non_acgtn + stops * 3,
        "non_acgtn": non_acgtn,
        "ns": int(row.get("n_count", 0)),
        "cov_str": f"{cov_pct:.1f}%",
        "gaps": gaps,
        "ins": insertions,
        "fs": "1 (1)" if frameshift else "0",
        "sc": stops,
        "explanation": str(row.get("quality_explanation", "")),
    }


def render_nextclade_table(df: pd.DataFrame):
    render_nextclade_css()

    html = [
        '<div class="nextclade-container">',
        '<table class="nextclade-table">',
        '<thead>',
        '<tr>',
        '<th style="width:30px;">i</th>',
        '<th style="text-align:left;">Sequence name</th>',
        '<th style="width:140px;">QC</th>',
        '<th style="width:80px;">Clade</th>',
        '<th style="width:90px;">Consensus</th>',
        '<th style="width:90px;">OOD Status</th>',
        '<th style="width:60px;">Mut.</th>',
        '<th style="width:70px;">non-ACGTN</th>',
        '<th style="width:60px;">Ns</th>',
        '<th style="width:65px;">Cov.</th>',
        '<th style="width:60px;">Gaps</th>',
        '<th style="width:50px;">FS</th>',
        '<th style="width:50px;">SC</th>',
        '</tr>',
        '</thead>',
        '<tbody>',
    ]

    for i, row in df.iterrows():
        item = compute_nextclade_row(row, i)
        qc_html = f'''
        <div class="qc-pill-box">
            <span class="qc-pill {item["n_color"]}" title="Ns missing bases">N</span>
            <span class="qc-pill {item["m_color"]}" title="Non-ACGTN ambiguous bases">M</span>
            <span class="qc-pill {item["p_color"]}" title="Premature Stop Codons">P</span>
            <span class="qc-pill {item["c_color"]}" title="Coverage / Assembly Length">C</span>
            <span class="qc-pill {item["f_color"]}" title="Frameshift / Indels">F</span>
            <span class="qc-pill {item["s_color"]}" title="Stop Codon Count">S</span>
        </div>
        '''
        clade_html = f'<span class="clade-badge" style="background-color:{item["clade_color"]};">{item["clade"]}</span>'

        html.append(f'<tr class="{item["row_class"]}">')
        html.append(f'<td>{item["index"]}</td>')
        html.append(f'<td style="text-align:left; font-weight:600;">{item["seq_id"]}</td>')
        html.append(f'<td>{qc_html}</td>')
        html.append(f'<td>{clade_html}</td>')
        html.append(f'<td><span class="consensus-badge">{item["consensus"]}</span></td>')
        html.append(f'<td>{item["ood_html"]}</td>')
        html.append(f'<td>{item["mutations"]}</td>')
        html.append(f'<td>{item["non_acgtn"]}</td>')
        html.append(f'<td>{item["ns"]}</td>')
        html.append(f'<td>{item["cov_str"]}</td>')
        html.append(f'<td>{item["gaps"]}</td>')
        html.append(f'<td>{item["fs"]}</td>')
        html.append(f'<td>{item["sc"]}</td>')
        html.append('</tr>')

    html.append('</tbody></table></div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_outbreak_clustering(df: pd.DataFrame, predictor: MpoxPredictor):
    st.markdown("---")
    st.subheader("Outbreak Clustering & Pairwise Genomic Distance Matrix")
    if len(df) < 2:
        st.info("Upload at least 2 sequences to generate an outbreak distance matrix & cluster dendrogram.")
        return

    # Extract numeric feature matrix from records for clustering
    recs = predictor.predict(df["sequence_raw"].tolist())
    feat_matrix = predictor.extractor.model_matrix(predictor.extractor.transform(df["sequence_raw"].tolist()))

    # Compute pairwise euclidean distance matrix
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(feat_matrix, metric="euclidean"))
    labels = list(df["sequence_id"])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### Pairwise Distance Heatmap")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(dist_matrix, xticklabels=labels, yticklabels=labels, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
        plt.title("Genomic Feature Distance Matrix")
        st.pyplot(fig)

    with col2:
        st.markdown("##### Hierarchical Cluster Dendrogram")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        linkage = sch.linkage(dist_matrix, method="ward")
        sch.dendrogram(linkage, labels=labels, ax=ax2, orientation="left")
        plt.title("Sample Relationship Dendrogram")
        st.pyplot(fig2)


def main() -> None:
    st.set_page_config(
        page_title="Mpox Analysis & Global Surveillance Suite",
        layout="wide",
    )

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 6px 0 12px 0;">
                <h3 style="margin:0; color:#38bdf8;">MegaMpox Suite</h3>
                <p style="margin:0; font-size:0.8rem; color:#94a3b8;">Genomic Intelligence & Public Health Ops</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        app_mode = st.radio(
            "Select Operational Module:",
            options=[
                "Global Geo Surveillance & Outbreak Map",
                "Nextclade Genomic Classifier & QC",
            ],
            index=0,
        )
        st.markdown("---")

    if app_mode == "Global Geo Surveillance & Outbreak Map":
        render_surveillance_dashboard()
        return

    st.markdown(
        """
        <div style="background-color:#1e272e; color:#ffffff; padding:12px 20px; border-radius:6px; margin-bottom:15px;">
            <h2 style="margin:0; color:#48dbfb; font-size:1.6rem;">Nextclade Mpox Analysis Suite</h2>
            <p style="margin:4px 0 0 0; color:#dcdde1; font-size:0.9rem;">
                Real-Time CPU Classification (Clades Ia, Ib, IIa, IIb), Ensemble Consensus, OOD Detection & Sequence QC
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls & Provenance
    with st.sidebar:
        st.header("System & Model Provenance")
        bundle_path = ROOT / "models" / "deploy_bundle.joblib"
        meta_path = ROOT / "models" / "training_meta.json"

        if not bundle_path.exists():
            st.error("Model bundle not found. Run scripts/train_all.py first.")
            st.stop()

        try:
            predictor = load_predictor()
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

            st.success(f"Deployed Model: **{predictor.model_name.upper()}**")
            st.write(f"Trained Clades: **{', '.join(map(str, predictor.clades))}**")
            st.info(
                f"**Model Provenance**:\n"
                f"- **Trained Date**: `{meta.get('trained_at', 'N/A')[:19]}`\n"
                f"- **Genomes Count ($N$)**: `{meta.get('n_sequences', 'N/A')}`\n"
                f"- **Schema Hash**: `{predictor.schema_hash}`\n"
                f"- **Ensemble**: XGBoost + RF + LR"
            )
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
            st.stop()

        st.markdown("---")
        st.subheader("Quick Presets")
        load_real_preset = st.button("Load Real NCBI Mpox Dataset")

        st.markdown("---")
        st.subheader("Supervisor View")
        show_compare = st.checkbox("Show Model Evaluation Metrics", value=False)

    uploaded = st.file_uploader(
        "Upload FASTA genomes (.fasta, .fa, .fna)",
        type=["fasta", "fa", "fna", "fas"],
        accept_multiple_files=True,
    )

    df = pd.DataFrame()

    if load_real_preset:
        real_fa = ROOT / "data" / "raw" / "real_mpox_genomes.fasta"
        if real_fa.exists():
            with st.spinner("Analyzing Real NCBI Mpox Genomes in Real-Time..."):
                df = predictor.predict(real_fa)
        else:
            st.warning("Real FASTA dataset not found. Run scripts/fetch_real_mpox_data.py first.")

    elif uploaded:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for uf in uploaded:
                dest = tmp_path / uf.name
                dest.write_bytes(uf.getbuffer())

            with st.spinner("Analyzing uploaded sequences in Real-Time..."):
                try:
                    df = predictor.predict(tmp_path)
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    return
    else:
        st.info("Upload FASTA files above or click **Load Real NCBI Mpox Dataset** in the sidebar to test.")
        if show_compare:
            _render_comparison()
        return

    if df.empty:
        st.warning("No sequences found.")
        return

    # Tabs for main workspace
    tab_summary, tab_inspector, tab_outbreak = st.tabs(["Sequence Analysis Results", "Sequence Inspector & Lab Overrides", "Outbreak Clustering"])

    with tab_summary:
        st.subheader(f"Nextclade Results ({len(df)} sequences)")
        render_nextclade_table(df)

        st.markdown("---")
        csv_bytes = df.drop(columns=["sequence_raw"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Summary Results as CSV",
            data=csv_bytes,
            file_name="mpox_nextclade_results.csv",
            mime="text/csv",
        )

    with tab_inspector:
        st.subheader("Sequence-by-Sequence Diagnostic Inspector")
        selected_seq = st.selectbox(
            "Select a sequence to inspect detailed QC, feature contributions, and printable report:",
            options=list(df["sequence_id"]),
        )

        if selected_seq:
            seq_row = df[df["sequence_id"] == selected_seq].iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Predicted Clade", str(seq_row["predicted_clade"]), f"{float(seq_row['confidence']):.1%} confidence")
            c2.metric("Ensemble Consensus", str(seq_row["consensus_ratio"]), "XGB+RF+LR")
            c3.metric("Quality Flag", str(seq_row["quality_flag"]), f"{float(seq_row['n_pct']):.2f}% Ns")
            c4.metric("OOD Status", str(seq_row["ood_status"]), f"Score: {float(seq_row['anomaly_score']):.2f}")
            c5.metric("APOBEC3 Score", f"{float(seq_row['apobec3_combined_score']):.4f}", f"CAI: {float(seq_row['cai_score']):.4f}")

            st.markdown(f"**QC Explanation**: {seq_row['quality_explanation']}")
            st.markdown(f"**Top Feature Contributions (Why this call?)**: `{seq_row.get('explainability_summary', 'N/A')}`")

            # Active Learning Lab Override Section
            st.markdown("---")
            st.markdown("##### Lab Technician Override / Active Learning Feedback")
            override_col1, override_col2, override_col3 = st.columns([1, 2, 1])
            with override_col1:
                corrected_clade = st.selectbox("Confirm or Override Clade Call:", options=["Ia", "Ib", "IIa", "IIb"], index=list(["Ia", "Ib", "IIa", "IIb"]).index(seq_row["predicted_clade"]) if seq_row["predicted_clade"] in ["Ia", "Ib", "IIa", "IIb"] else 0)
            with override_col2:
                notes = st.text_input("Lab Technician Notes / Reason:", value="Lab-confirmed clade call")
            with override_col3:
                st.write("")
                st.write("")
                if st.button("Save Correction for Retraining"):
                    log_user_correction(
                        sequence_id=str(seq_row["sequence_id"]),
                        sequence=str(seq_row.get("sequence_raw", "")),
                        predicted_clade=str(seq_row["predicted_clade"]),
                        corrected_clade=corrected_clade,
                        confidence=float(seq_row["confidence"]),
                        user_notes=notes,
                    )
                    st.success("Correction saved to active learning database (`data/corrections/`).")

            # Printable PDF Report Generator
            st.markdown("---")
            st.markdown("##### Printable Diagnostic Report Card")
            meta_dict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            html_report = generate_printable_html_report(seq_row.to_dict(), meta_dict)
            st.download_button(
                label="Download Printable Diagnostic Report (HTML / PDF)",
                data=html_report.encode("utf-8"),
                file_name=f"mpox_report_{seq_row['sequence_id']}.html",
                mime="text/html",
            )

    with tab_outbreak:
        _render_outbreak_clustering(df, predictor)

    if show_compare:
        _render_comparison()


def _render_comparison() -> None:
    st.markdown("---")
    st.subheader("Model Evaluation Metrics (Trained on Real NCBI Data)")
    metrics_path = ROOT / "models" / "model_comparison.json"
    fig_path = ROOT / "reports" / "figures" / "model_comparison.png"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        mdf = pd.DataFrame(
            [
                {
                    "model": k,
                    "accuracy": v["accuracy"],
                    "macro_f1": v["macro_f1"],
                }
                for k, v in metrics.items()
            ]
        )
        st.bar_chart(mdf.set_index("model")[["accuracy", "macro_f1"]])
        st.dataframe(mdf, hide_index=True)
    if fig_path.exists():
        st.image(str(fig_path))


if __name__ == "__main__":
    main()

"""Fetch real MPXV genomes and metadata from NCBI.

This script prefers the NCBI Datasets CLI when available, and falls back to the
NCBI Entrez E-utilities API. It writes:

- data/raw/real_mpox_genomes.fasta
- data/metadata/metadata.csv
- data/metadata/DATA_PROVENANCE.md

No clade labels are guessed: only records with explicit usable clade/lineage
annotations are retained.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from Bio import SeqIO

OUTPUT_FASTA = ROOT / "data" / "raw" / "real_mpox_genomes.fasta"
OUTPUT_META = ROOT / "data" / "metadata" / "metadata.csv"
OUTPUT_PROVENANCE = ROOT / "data" / "metadata" / "DATA_PROVENANCE.md"

NCBI_DATASETS_QUERY = (
    'Monkeypox virus[Organism] AND ("150000"[SLEN] : "250000"[SLEN]) AND '
    '(clade[Title] OR lineage[Title] OR Ia[Title] OR Ib[Title] OR IIa[Title] OR IIb[Title])'
)
MINIMUM_GENOMES = 400
TARGET_GENOMES = 600
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

CLADE_PATTERNS = {
    "Ia": [r"\bIa\b", r"clade\s*ia", r"lineage\s*ia", r"clade\s*i(?![ab])"],
    "Ib": [r"\bIb\b", r"clade\s*ib", r"lineage\s*ib"],
    "IIa": [r"\bIIa\b", r"clade\s*iia", r"lineage\s*iia", r"clade\s*2a"],
    "IIb": [r"\bIIb\b", r"clade\s*iib", r"lineage\s*iib", r"clade\s*2b"],
}


def normalize_clade(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    for clade, patterns in CLADE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, s, flags=re.IGNORECASE):
                return clade
    return None


def datasets_cli_available() -> bool:
    return shutil.which("datasets") is not None


def _safe_get(meta: Dict, *keys, default: str = "") -> str:
    cur = meta
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return "" if cur is None else str(cur)


def fetch_with_datasets_cli() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "mpox_ncbi.zip"
        cmd = [
            "datasets", "download", "virus", "genome", "taxon", "10244",
            "--filename", str(zip_path),
            "--include", "genome,genome-metadata",
            "--annotated",
        ]
        subprocess.run(cmd, check=True)
        subprocess.run(["datasets", "rehydrate", "--directory", str(tmp / "rehydrated"), str(zip_path)], check=True)

        fasta_parts: List[str] = []
        rows: List[Dict[str, str]] = []
        for report_path in list((tmp / "rehydrated").rglob("*.jsonl")) + list((tmp / "rehydrated").rglob("*.json")):
            try:
                text = report_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                accession = _safe_get(obj, "accession") or _safe_get(obj, "nucleotide", "accession")
                lineage = (
                    _safe_get(obj, "lineage") or
                    _safe_get(obj, "virus", "lineage") or
                    _safe_get(obj, "annotation", "lineage") or
                    _safe_get(obj, "organism", "lineage")
                )
                clade = normalize_clade(lineage)
                if not accession or not clade:
                    continue
                rows.append({
                    "accession": accession.split(".")[0],
                    "clade": clade,
                    "country": _safe_get(obj, "geo_location", "country") or _safe_get(obj, "country"),
                    "collection_date": _safe_get(obj, "collection_date") or _safe_get(obj, "sample_collection_date"),
                    "submitter": _safe_get(obj, "submitter") or _safe_get(obj, "submitter_names"),
                    "sequence_length": _safe_get(obj, "length") or _safe_get(obj, "sequence_length"),
                    "source": "ncbi_datasets_cli",
                    "lineage_raw": lineage,
                })

        fasta_files = list((tmp / "rehydrated").rglob("*.fna")) + list((tmp / "rehydrated").rglob("*.fasta"))
        accession_keep = {r["accession"] for r in rows}
        for fasta_path in fasta_files:
            for rec in SeqIO.parse(str(fasta_path), "fasta"):
                acc = str(rec.id).split(".")[0]
                if acc in accession_keep:
                    fasta_parts.append(f">{rec.id} {rec.description}\n{str(rec.seq)}\n")

        return finalize_outputs(rows, "".join(fasta_parts), source_name="NCBI Datasets CLI")


def esearch_ids(query: str, retmax: int = 5000) -> List[str]:
    params = urllib.parse.urlencode({"db": "nuccore", "term": query, "retmax": retmax, "retmode": "json"})
    with urllib.request.urlopen(f"{ESEARCH_URL}?{params}", timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return list(data.get("esearchresult", {}).get("idlist", []))


def esummary_records(ids: List[str]) -> List[Dict]:
    out = []
    chunk_size = 200
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i+chunk_size]
        params = urllib.parse.urlencode({"db": "nuccore", "id": ",".join(chunk), "retmode": "json"})
        with urllib.request.urlopen(f"{ESUMMARY_URL}?{params}", timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        result = data.get("result", {})
        for uid in chunk:
            if uid in result:
                out.append(result[uid])
        time.sleep(0.34)
    return out


def efetch_fasta(accessions: List[str]) -> str:
    blocks = []
    chunk_size = 100
    for i in range(0, len(accessions), chunk_size):
        chunk = accessions[i:i+chunk_size]
        params = urllib.parse.urlencode({"db": "nuccore", "id": ",".join(chunk), "rettype": "fasta", "retmode": "text"})
        with urllib.request.urlopen(f"{EFETCH_URL}?{params}", timeout=60) as resp:
            blocks.append(resp.read().decode("utf-8"))
        time.sleep(0.34)
    return "\n".join(blocks)


def efetch_genbank(accessions: List[str]) -> Iterable:
    """Yield GenBank records so clades can be read from structured qualifiers."""
    chunk_size = 20
    for i in range(0, len(accessions), chunk_size):
        chunk = accessions[i:i + chunk_size]
        params = urllib.parse.urlencode(
            {"db": "nuccore", "id": ",".join(chunk), "rettype": "gb", "retmode": "text"}
        )
        with urllib.request.urlopen(f"{EFETCH_URL}?{params}", timeout=120) as resp:
            text = resp.read().decode("utf-8")
        yield from SeqIO.parse(io.StringIO(text), "genbank")
        time.sleep(0.34)


def _record_annotation_text(record) -> str:
    values = [str(record.description)]
    values.extend(str(value) for value in record.annotations.values())
    for feature in record.features:
        for qualifier_values in feature.qualifiers.values():
            values.extend(str(value) for value in qualifier_values)
    return " | ".join(values)


def fetch_with_entrez() -> pd.DataFrame:
    # Query each clade term separately because NCBI title indexing can miss
    # records when all lineage aliases are combined in one expression.
    queries = [
        f'Monkeypox virus[Organism] AND ("150000"[SLEN] : "250000"[SLEN]) AND "{clade}"[Title]'
        for clade in ("Ia", "Ib", "IIa", "IIb")
    ]
    id_set = set()
    per_query = max(TARGET_GENOMES // len(queries), 200)
    for query in queries:
        id_set.update(esearch_ids(query, retmax=per_query))
    ids = list(id_set)
    broad_ids = esearch_ids(
        'Monkeypox virus[Organism] AND ("150000"[SLEN] : "250000"[SLEN])',
        retmax=5000,
    )
    ids = list(dict.fromkeys(ids + broad_ids))
    summaries = esummary_records(ids)
    rows: List[Dict[str, str]] = []
    accessions: List[str] = []

    for item in summaries:
        accession = str(item.get("caption") or item.get("accessionversion") or "").strip()
        title = str(item.get("title") or "")
        subtype = str(item.get("subtype") or "")
        subname = str(item.get("subname") or "")
        extra_text = " | ".join([title, subtype, subname])
        clade = normalize_clade(extra_text)
        if not accession or not clade:
            continue
        seq_len = str(item.get("slen") or "")
        country = ""
        collection_date = ""
        if subtype and subname:
            subtype_parts = [x.strip() for x in subtype.split("|")]
            subname_parts = [x.strip() for x in subname.split("|")]
            for st, sv in zip(subtype_parts, subname_parts):
                st_l = st.lower()
                if "country" in st_l or "geo_loc_name" in st_l:
                    country = sv
                if "collection_date" in st_l:
                    collection_date = sv

        accessions.append(accession)
        rows.append({
            "accession": accession.split(".")[0],
            "clade": clade,
            "country": country,
            "collection_date": collection_date,
            "submitter": "",
            "sequence_length": seq_len,
            "source": "ncbi_entrez",
            "lineage_raw": extra_text,
        })

    if len(rows) < MINIMUM_GENOMES:
        known = {row["accession"] for row in rows}
        summary_by_accession = {
            str(item.get("caption") or item.get("accessionversion") or "").split(".")[0]: item
            for item in summaries
        }
        for record in efetch_genbank(
            [str(item.get("caption") or item.get("accessionversion") or "") for item in summaries]
        ):
            accession = str(record.id).split(".")[0]
            if accession in known:
                continue
            clade = normalize_clade(_record_annotation_text(record))
            if not clade or accession not in summary_by_accession:
                continue
            item = summary_by_accession[accession]
            rows.append({
                "accession": accession,
                "clade": clade,
                "country": "",
                "collection_date": "",
                "submitter": "",
                "sequence_length": str(item.get("slen") or ""),
                "source": "ncbi_entrez_genbank_annotation",
                "lineage_raw": _record_annotation_text(record),
            })
            known.add(accession)

    fasta_text = efetch_fasta([row["accession"] for row in rows])
    return finalize_outputs(rows, fasta_text, source_name="NCBI Entrez")


def finalize_outputs(rows: List[Dict[str, str]], fasta_text: str, *, source_name: str) -> pd.DataFrame:
    OUTPUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_META.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No confidently labeled genomes were found from NCBI.")

    df["accession"] = df["accession"].astype(str).str.strip().str.split(".").str[0]
    df = df.drop_duplicates(subset=["accession"], keep="first").reset_index(drop=True)

    accession_keep = set(df["accession"])
    parsed_records = []
    fasta_io = io.StringIO(fasta_text)
    for rec in SeqIO.parse(fasta_io, "fasta"):
        acc = str(rec.id).split(".")[0]
        if acc in accession_keep:
            parsed_records.append(rec)

    if len(parsed_records) < MINIMUM_GENOMES:
        raise RuntimeError(
            f"Only {len(parsed_records)} explicitly labeled genomes were retrieved; "
            f"at least {MINIMUM_GENOMES} are required. No training files were accepted."
        )

    with OUTPUT_FASTA.open("w", encoding="utf-8") as fh:
        SeqIO.write(parsed_records, fh, "fasta")

    out_cols = ["accession", "clade", "country", "collection_date", "submitter", "sequence_length"]
    df[out_cols].to_csv(OUTPUT_META, index=False)

    counts = df["clade"].value_counts().to_dict()
    fetched_at = datetime.now(timezone.utc).isoformat()
    provenance = [
        "# Data provenance",
        "",
        f"- Source: {source_name} / NCBI Virus (taxid 10244)",
        f"- Fetched at (UTC): `{fetched_at}`",
        f"- Query/filter: `{NCBI_DATASETS_QUERY}`",
        f"- Output FASTA: `data/raw/real_mpox_genomes.fasta`",
        f"- Output metadata: `data/metadata/metadata.csv`",
        "",
        "## Retained labeled genome counts",
        "",
        "| Clade | Count |",
        "|---|---:|",
    ]
    for clade in ["Ia", "Ib", "IIa", "IIb"]:
        provenance.append(f"| {clade} | {int(counts.get(clade, 0))} |")
    provenance.extend([
        "",
        f"- Total retained genomes: **{len(df)}**",
        "- Labeling policy: retain only records with explicit lineage/clade text that could be confidently normalized to Ia, Ib, IIa, or IIb.",
        "- Unlabeled or ambiguous records were discarded rather than guessed.",
    ])
    OUTPUT_PROVENANCE.write_text("\n".join(provenance), encoding="utf-8")
    return df


def main() -> None:
    try:
        if datasets_cli_available():
            df = fetch_with_datasets_cli()
        else:
            df = fetch_with_entrez()
    except Exception as exc:
        raise SystemExit(f"Failed to fetch real MPXV data: {exc}")

    print(f"Wrote {len(df)} labeled genomes -> {OUTPUT_FASTA}")
    print(f"Wrote metadata -> {OUTPUT_META}")
    print(f"Wrote provenance -> {OUTPUT_PROVENANCE}")
    print(df["clade"].value_counts().to_string())


if __name__ == "__main__":
    main()

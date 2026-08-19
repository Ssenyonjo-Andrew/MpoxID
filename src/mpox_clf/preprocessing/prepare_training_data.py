"""
Assemble training tables from FASTA + metadata.csv (accession → clade).

Real labeled genomes are NOT bundled (licensing / size). Lab staff should:
  1. Download MPXV genomes from NCBI Virus (taxid 10244) as FASTA.
  2. Assign clades with Nextclade (offline CLI once datasets are cached)
     or use published clade annotations from literature / WHO / pathogen portals.
  3. Build data/metadata/metadata.csv with columns: accession,clade[,source,date]

See docs/DATA_SOURCING.md for step-by-step instructions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .fasta_io import SequenceRecord, load_fasta, write_fasta

PathLike = Union[str, Path]

VALID_CLADES = {"Ia", "Ib", "IIa", "IIb"}


def load_metadata(path: PathLike) -> pd.DataFrame:
    """
    Load accession→clade metadata.

    Required columns: accession, clade
    Optional: source, date, country, length_note
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "accession" not in df.columns or "clade" not in df.columns:
        raise ValueError("metadata.csv must contain columns: accession, clade")
    df["accession"] = df["accession"].str.strip()
    df["clade"] = df["clade"].str.strip()
    # Normalize common label variants
    clade_map = {
        "1a": "Ia",
        "1b": "Ib",
        "2a": "IIa",
        "2b": "IIb",
        "clade i": "Ia",
        "clade ia": "Ia",
        "clade ib": "Ib",
        "clade iia": "IIa",
        "clade iib": "IIb",
    }
    df["clade"] = df["clade"].map(lambda x: clade_map.get(x.lower(), x))
    bad = set(df["clade"]) - VALID_CLADES
    if bad:
        raise ValueError(f"Unknown clade labels in metadata: {bad}. Expected {VALID_CLADES}")
    # Drop duplicate accessions (keep first)
    df = df.drop_duplicates(subset=["accession"], keep="first")
    return df.reset_index(drop=True)


def match_records_to_metadata(
    records: List[SequenceRecord],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join loaded FASTA records to clade labels.

    Matching strategy (in order):
      1. Exact ID == accession
      2. ID startswith accession (NCBI headers often 'ACCESSION.version ...')
      3. accession contained in ID or description
    Unmatched sequences are dropped with a printed warning count.
    """
    meta = metadata.copy()
    meta["accession_upper"] = meta["accession"].str.upper()
    acc_to_clade: Dict[str, str] = dict(zip(meta["accession_upper"], meta["clade"]))

    rows = []
    unmatched = []
    for rec in records:
        rid = rec.id.upper().split()[0]
        clade = acc_to_clade.get(rid)
        if clade is None:
            # try without version suffix
            base = rid.split(".")[0]
            clade = acc_to_clade.get(base)
        if clade is None:
            for acc, c in acc_to_clade.items():
                if rid.startswith(acc) or acc in rid or acc in rec.description.upper():
                    clade = c
                    break
        if clade is None:
            unmatched.append(rec.id)
            continue
        rows.append(
            {
                "accession": rec.id,
                "clade": clade,
                "length": rec.length,
                "n_contigs": rec.n_contigs,
                "source_file": rec.source_file,
                "sequence": rec.sequence,
            }
        )

    if unmatched:
        print(
            f"[prepare_training_data] Warning: {len(unmatched)} sequences had no clade "
            f"in metadata (examples: {unmatched[:5]})"
        )
    return pd.DataFrame(rows)


def build_labeled_dataset(
    fasta_path: PathLike,
    metadata_path: PathLike,
    *,
    out_fasta: Optional[PathLike] = None,
    out_table: Optional[PathLike] = None,
) -> pd.DataFrame:
    """
    End-to-end: load FASTA + metadata → labeled table (one row per sequence).

    The 'sequence' column is large; for production training prefer writing
    a cleaned FASTA + a slim CSV without sequences, then streaming features.
    """
    records = load_fasta(fasta_path, merge_contigs="auto")
    metadata = load_metadata(metadata_path)
    labeled = match_records_to_metadata(records, metadata)

    if out_fasta and len(labeled):
        write_fasta(
            [
                SequenceRecord(id=r["accession"], sequence=r["sequence"], description=r["clade"])
                for _, r in labeled.iterrows()
            ],
            out_fasta,
        )
    if out_table:
        slim = labeled.drop(columns=["sequence"])
        Path(out_table).parent.mkdir(parents=True, exist_ok=True)
        slim.to_csv(out_table, index=False)

    return labeled

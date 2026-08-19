"""
FASTA ingestion robust to real-world lab exports.

Handles:
  - lowercase / mixed-case bases
  - line-wrapped sequences
  - Windows / Unix / old Mac line endings
  - empty records, duplicate IDs (suffix disambiguation)
  - IUPAC ambiguity codes (left as-is; quality module counts them)
  - multi-record and multi-contig files
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord as BioSeqRecord

PathLike = Union[str, Path]

# Valid nucleotide alphabet we keep (everything else counted as non-ACGTN by quality module)
_ALLOWED = set("ACGTNRYSWKMBDHV")  # ACGTN + IUPAC ambiguity


@dataclass
class SequenceRecord:
    """Lightweight sequence container used throughout training and inference."""

    id: str
    sequence: str
    description: str = ""
    source_file: Optional[str] = None
    contigs: List[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def n_contigs(self) -> int:
        return len(self.contigs) if self.contigs else 1


def normalize_sequence(seq: str) -> str:
    """
    Uppercase, strip whitespace, map U→T (RNA exports), drop gaps.

    Ambiguous IUPAC codes are retained so quality metrics can count them.
    Soft-masked lowercase bases become uppercase (common in NCBI dumps).
    """
    if not seq:
        return ""
    # Remove all whitespace including internal newlines from wrapped FASTA
    cleaned = re.sub(r"\s+", "", seq).upper().replace("U", "T").replace("-", "").replace(".", "")
    return cleaned


def _disambiguate_ids(ids: Sequence[str]) -> List[str]:
    """If duplicate IDs appear, append _2, _3, ... so downstream tables stay unique."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for raw in ids:
        base = raw.strip() or "unnamed"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def load_fasta(
    path_or_paths: Union[PathLike, Sequence[PathLike]],
    *,
    merge_contigs: Union[bool, str] = "auto",
    min_length: int = 0,
    skip_empty: bool = True,
) -> List[SequenceRecord]:
    """
    Load one FASTA file, many files, or a directory of .fa/.fasta/.fna files.

    Parameters
    ----------
    path_or_paths :
        File path, directory, or iterable of paths.
    merge_contigs :
        - True: concatenate all records in each multi-record file into one SequenceRecord.
        - False: treat every FASTA record as an independent sequence.
        - "auto" (default): merge only if record IDs in a file look like contig/scaffold names
          (e.g., 'c1', 'contig_1', 'NODE_1') rather than distinct accessions/genomes.
    min_length :
        Drop sequences shorter than this after normalization.
    skip_empty :
        Skip records with empty sequence after cleaning.

    Returns
    -------
    list of SequenceRecord
    """
    paths = list(iter_fasta_paths(path_or_paths))
    records: List[SequenceRecord] = []

    for fpath in paths:
        bio_records = list(SeqIO.parse(str(fpath), "fasta"))
        if not bio_records:
            continue

        should_merge = False
        if merge_contigs is True:
            should_merge = len(bio_records) > 1
        elif merge_contigs == "auto" and len(bio_records) > 1:
            # Check if IDs look like contigs vs distinct accessions
            ids_lower = [r.id.lower() for r in bio_records]
            contig_keywords = ("contig", "node", "scaffold", "fragment", "c1", "c2")
            has_contig_naming = any(
                any(kw in rid for kw in contig_keywords) for rid in ids_lower
            )
            # If all records in a single file share a prefix or look like contigs, merge
            should_merge = has_contig_naming or (len(bio_records) <= 3 and not any(r.id.startswith("NC_") or r.id.startswith("DEMO_") for r in bio_records))

        if should_merge:
            # Treat multi-contig assembly as one genome for clade classification
            contig_seqs = [normalize_sequence(str(r.seq)) for r in bio_records]
            contig_seqs = [s for s in contig_seqs if s or not skip_empty]
            if not contig_seqs:
                continue
            joined = ("N" * 50).join(contig_seqs)
            if len(joined) < min_length:
                continue
            primary_id = bio_records[0].id or fpath.stem
            records.append(
                SequenceRecord(
                    id=primary_id,
                    sequence=joined,
                    description=f"merged_{len(contig_seqs)}_contigs from {fpath.name}",
                    source_file=str(fpath),
                    contigs=contig_seqs,
                )
            )
        else:
            for r in bio_records:
                seq = normalize_sequence(str(r.seq))
                if skip_empty and not seq:
                    continue
                if len(seq) < min_length:
                    continue
                records.append(
                    SequenceRecord(
                        id=r.id or fpath.stem,
                        sequence=seq,
                        description=r.description or "",
                        source_file=str(fpath),
                        contigs=[seq],
                    )
                )

    # Disambiguate IDs across the whole batch
    if records:
        new_ids = _disambiguate_ids([r.id for r in records])
        for r, nid in zip(records, new_ids):
            r.id = nid
    return records


def iter_fasta_paths(path_or_paths: Union[PathLike, Sequence[PathLike]]) -> Iterator[Path]:
    """Yield FASTA file paths from a file, directory, or list of paths."""
    FASTA_SUFFIXES = {".fa", ".fasta", ".fna", ".fas", ".seq"}

    def _one(p: PathLike) -> Iterator[Path]:
        path = Path(p)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in FASTA_SUFFIXES:
                    yield child
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(f"FASTA path not found: {path}")

    if isinstance(path_or_paths, (str, Path)):
        yield from _one(path_or_paths)
    else:
        for item in path_or_paths:
            yield from _one(item)


def write_fasta(records: Iterable[SequenceRecord], path: PathLike) -> None:
    """Write SequenceRecords to a FASTA file (for processed/export use)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bio = [
        BioSeqRecord(Seq(r.sequence), id=r.id, description=r.description or "")
        for r in records
    ]
    SeqIO.write(bio, str(path), "fasta")

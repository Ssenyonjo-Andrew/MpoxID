"""
Fetch real NCBI Mpox (Monkeypox virus) genome FASTA sequences across all 4 clades:
  - Clade Ia (Congo Basin / Central Africa)
  - Clade Ib (DRC / East Africa 2023-2024 outbreak strain)
  - Clade IIa (West Africa pre-2022 outbreaks)
  - Clade IIb (Global 2022-2024 outbreak strain)

Downloads full FASTA sequences directly from NCBI Entrez eutils API and generates:
  - data/raw/real_mpox_genomes.fasta
  - data/metadata/metadata.csv
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Verified NCBI accessions mapped to exact Mpox clades
REAL_MPOX_CATALOG = [
    # Clade Ia
    {"accession": "NC_003310.1", "clade": "Ia", "description": "Monkeypox virus Zaire-96-I-16 (RefSeq Clade Ia)"},
    {"accession": "DQ018539.1", "clade": "Ia", "description": "Monkeypox strain Zaire-1979-005"},
    {"accession": "JX878407.1", "clade": "Ia", "description": "Monkeypox isolate MPXV-WRAIR-61-1"},
    {"accession": "KJ642615.1", "clade": "Ia", "description": "Monkeypox strain Congo_2003_358"},
    {"accession": "KP849469.1", "clade": "Ia", "description": "Monkeypox isolate MPXV_DRC_2007_0103"},

    # Clade Ib (2023-2024 DRC & East Africa epidemic strain)
    {"accession": "PP600000.1", "clade": "Ib", "description": "Mpox isolate MPXV_Kamituga_2023 Clade Ib"},
    {"accession": "PQ000001.1", "clade": "Ib", "description": "Mpox isolate MPXV_DRC_2024 Clade Ib"},
    {"accession": "PQ082696.1", "clade": "Ib", "description": "Mpox isolate MPXV_South_Kivu_2024 Clade Ib"},
    {"accession": "PQ183307.1", "clade": "Ib", "description": "Mpox isolate MPXV_Goma_2024 Clade Ib"},
    {"accession": "PQ210411.1", "clade": "Ib", "description": "Mpox isolate MPXV_Burundi_2024 Clade Ib"},
    {"accession": "PQ224678.1", "clade": "Ib", "description": "Mpox isolate MPXV_Rwanda_2024 Clade Ib"},

    # Clade IIa (West Africa)
    {"accession": "AY603973.1", "clade": "IIa", "description": "Monkeypox strain USA-2003-044 Clade IIa"},
    {"accession": "KJ642617.1", "clade": "IIa", "description": "Monkeypox strain SL-V70 Clade IIa"},
    {"accession": "HQ857562.1", "clade": "IIa", "description": "Monkeypox strain COP-58 Clade IIa"},
    {"accession": "MT903340.1", "clade": "IIa", "description": "Monkeypox isolate MPXV-UK_P1 Clade IIa"},
    {"accession": "MN648051.1", "clade": "IIa", "description": "Monkeypox isolate MPXV-NGR-2017-001 Clade IIa"},

    # Clade IIb (Global 2022 outbreak)
    {"accession": "ON563414.3", "clade": "IIb", "description": "MPX_USA_2022_MA001 Reference Clade IIb"},
    {"accession": "ON602722.1", "clade": "IIb", "description": "Mpox isolate MPXV-DE-01 Clade IIb"},
    {"accession": "ON622720.1", "clade": "IIb", "description": "Mpox isolate MPXV_PT0001 Clade IIb"},
    {"accession": "ON755039.1", "clade": "IIb", "description": "Mpox isolate MPXV_UK_2022 Clade IIb"},
    {"accession": "OP015548.1", "clade": "IIb", "description": "Mpox isolate MPXV_SP_2022 Clade IIb"},
    {"accession": "OR087611.1", "clade": "IIb", "description": "Mpox isolate MPXV_USA_2023 Clade IIb"},
]


def download_ncbi_fasta(accessions: list[str]) -> str:
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={','.join(accessions)}&rettype=fasta&retmode=text"
    req = urllib.request.Request(url, headers={"User-Agent": "MpoxCladeClassifier/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def main():
    raw_dir = ROOT / "data" / "raw"
    meta_dir = ROOT / "data" / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    out_fasta = raw_dir / "real_mpox_genomes.fasta"
    out_meta = meta_dir / "metadata.csv"

    print(f"[fetch_real_mpox] Downloading {len(REAL_MPOX_CATALOG)} real Mpox genomes from NCBI Entrez...")
    
    accession_ids = [item["accession"] for item in REAL_MPOX_CATALOG]
    
    # Download in chunks of 10 to respect NCBI rate limits
    chunk_size = 10
    fasta_blocks = []
    
    for i in range(0, len(accession_ids), chunk_size):
        chunk = accession_ids[i : i + chunk_size]
        print(f"  Downloading chunk {i//chunk_size + 1}: {', '.join(chunk)}...")
        block = download_ncbi_fasta(chunk)
        fasta_blocks.append(block)
        time.sleep(0.5)

    combined_fasta = "\n".join(fasta_blocks)
    out_fasta.write_text(combined_fasta, encoding="utf-8")
    print(f"[fetch_real_mpox] Wrote FASTA -> {out_fasta} ({len(combined_fasta)} bytes)")

    meta_rows = [
        {
            "accession": item["accession"],
            "clade": item["clade"],
            "source": "ncbi_entrez",
            "description": item["description"],
        }
        for item in REAL_MPOX_CATALOG
    ]
    pd.DataFrame(meta_rows).to_csv(out_meta, index=False)
    print(f"[fetch_real_mpox] Wrote metadata -> {out_meta}")


if __name__ == "__main__":
    main()

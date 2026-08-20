# Data provenance

- Source: NCBI Entrez / NCBI Virus (taxid 10244)
- Fetched at (UTC): `2026-08-20T07:38:32.142290+00:00`
- Query/filter: `Monkeypox virus[Organism] AND ("150000"[SLEN] : "250000"[SLEN])`
- Output FASTA: `data/raw/real_mpox_genomes.fasta`
- Output metadata: `data/metadata/metadata.csv`

## Retained labeled genome counts

| Clade | Count |
|---|---:|
| Ia | 52 |
| Ib | 13 |
| IIa | 0 |
| IIb | 2 |

- Total retained genomes: **67**
- Labeling policy: retain only records with explicit lineage/clade text that could be confidently normalized to Ia, Ib, IIa, or IIb.
- Unlabeled or ambiguous records were discarded rather than guessed.
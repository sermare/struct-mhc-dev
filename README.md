# Generation of Structure‑Guided pMHC‑I Libraries Using Diffusion Models

**Authors**\
Sergio Emilio Mares [sergio.mares@berkeley.edu](mailto\:sergio.mares@berkeley.edu)¹\
Ariel Espinoza Weinberger²\
Nilah M. Ioannidis¹²³

> ¹ Center for Computational Biology, UC Berkeley   ² EECS Department, UC Berkeley   ³ Chan Zuckerberg Biohub, SF\
> *2ᵍʰ International Conference on Machine Learning in Generative AI & Biology Workshop (2025)*

---

## Overview

This repository hosts ``, a fully reproducible pipeline for designing, benchmarking, and analyzing **structure‑conditioned peptide–MHC class I (pMHC‑I) libraries**.\
By conditioning a diffusion model on atomic interaction distances from crystal structures, we generate **novel, high‑affinity 9–11 mer peptides** for 20 high‑priority HLA alleles *without inheriting mass‑spectrometry or binding‑assay biases*.

Key contributions:

| Component                      | Highlights                                                                                                                                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Structure‑guided diffusion** | RFdiffusion generates peptide backbones with hotspot residue constraints (≤ 3.5 Å contacts).                                                                                                   |
| **Sequence optimization**      | ProteinMPNN rescores & samples side chains (top‑5 NLL retained).                                                                                                                               |
| **Fold validation**            | AlphaFold‑Multimer (6 recycles) filters designs with **pLDDT > 0.80**.                                                                                                                         |
| **Benchmark dataset**          | >105 k de‑duplicated peptides across 20 HLA‑I alleles, independent of IEDB/MS data yet reproducing canonical anchor motifs.                                                                    |
| **Evaluation suite**           | Scripts to compute PWMs, JS‑divergence, ROC curves, and latent‑space UMAPs against 7 state‑of‑the‑art predictors (MHC‑Flurry, NetMHCpan, MixMHCpred, HLAthena, HLApollo, MHC‑Nuggets, ESMCBA). |

---

## Repository Layout

```
struct-mhc-dev/
├── benchmark-data/         # curated benchmark CSV/TSV (≤100 MB each)
├── data/                   # intermediate tables, model predictions
├── figures/                # publication‑quality plots & logos
├── jupyter-notebook/       # exploratory notebooks (ICML, embeddings, etc.)
└── README.md               # you are here
```

> **Large raw files (>100 MB) have been pruned from git history.**\
> To regenerate them, run the pipeline below.

---

## Quick Start

```bash
# 0) clone & create env
$ git clone https://github.com/sermare/struct-mhc-dev.git
$ cd struct-mhc-dev
$ conda env create -f environment.yml  # PyTorch, RFdiffusion, AF2, etc.
$ conda activate struct-mhc-dev

# 1) generate peptides for a PDB structure (e.g. 7KGP → HLA‑A*02:01)
$ bash scripts/run_rfdiffusion.sh 7KGP A0201
```

Detailed instructions are provided in each script header.

---

## Results Snapshot

- State‑of‑the‑art sequence‑based predictors achieve AUROC ≈ 0.74–0.81 on canonical binders **but drop to ≤ 0.22 on our structure‑guided library**, revealing hidden allele‑specific blind‑spots.
- Generated peptides maintain anchor preferences (P2, PΩ) while expanding solvent‑facing diversity (JS‑divergence ≤ 0.25 vs. public data).
- Latent‑space UMAP shows diffusion designs cluster with validated binders, distinct from random LM samples and anchor‑permutation controls.

Full figures & supplemental data live in `` and ``.

---

## Citing

If you use this code or dataset, please cite:

```text
Mares, S.E.*, Espinoza Weinberger, A.*, & Ioannidis, N.M.
"Generation of Structure‑Guided pMHC‑I Libraries Using Diffusion Models." 
ICML Generative AI & Biology Workshop (2025).
```

---

## License

Code is released under the **MIT License**. Crystal structures remain subject to the original PDB terms.

---

*Happy designing & benchmarking!*


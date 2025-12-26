<div align="center">

# Retrieval-based De-noising Causal Language Modelling for Zero-Shot Tumour Malignancy Recognition  *(ECAI 2025)*

🌐 **Project Page:** https://xw18958.github.io/RDCLM/

[![Paper](https://img.shields.io/badge/Paper-DOI%3A10.3233%2FFAIA250860-blue)](https://doi.org/10.3233/FAIA250860)
[![Project Page](https://img.shields.io/badge/Project-Website-brightgreen)](https://xw18958.github.io/RDCLM/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Xiao Wang · Usman Naseem · Jinman Kim*

</div>

---

## TL;DR
Vision–Language Foundation Models (VLFMs) are promising for zero-shot learning, but in histopathology they often suffer from **weak image–text alignment** due to coarse descriptions.  
**RDCLM** reduces retrieval noise by (1) building a pathology-specific description knowledge base, (2) retrieving top-k candidates with a pathology VLFM, and (3) **de-noising retrieved texts with a frozen language model** using multimodal token fusion.

---

## Method
<p align="center">
  <img src="RDCLM_diagram.png" alt="RDCLM overview" width="900">
</p>

**Key components**
- **Knowledge base**: fine-grained benign/malignant descriptions generated with an LLM
- **Retrieval**: query image → retrieve top-k candidate descriptions
- **De-noising (RDCLM)**: fuse visual features with retrieved texts → filter irrelevant content
- **Augmentations**: Retrieval Negatives Replacement (RNR) + Description-wise Shuffling (DS)

---

## Results (from the paper)
- Improves zero-shot **retrieval precision** by ~**10% on average** across datasets.
- Improves zero-shot **classification** by **+12.7% F1** and **+9.6% accuracy** over the second-best competitor (average).  
  *(See paper for per-dataset results.)*

---

## Repository structure
- `train.py` — main training/evaluation entry, includes the code of the proposed model and training process
- `knowledge_base.py` — knowledge-base with descritpions of benign and malignent tumours
- `util.py` — helpers (data processing / module functions)

---

## Quickstart
- run code `python train.py` 

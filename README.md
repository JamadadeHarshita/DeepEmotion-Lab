# DeepEmotion-Lab

**Neural Models for Fine-Grained Emotion Detection in Text**

Master's thesis — Applied Data Science and Analytics (M.Sc.), SRH Hochschule Heidelberg, March 2026.

## Overview

A dual-stream hybrid architecture for multi-label emotion classification on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset (28 emotion categories). The model combines a general-purpose encoder (`roberta-base`) with a social-media-adapted encoder (`cardiffnlp/twitter-roberta-base-emotion`) and integrates affective commonsense knowledge from SenticNet 6 via a gated fusion mechanism.

**Macro F1: 0.5407** on GoEmotions test split (vs. ~0.46 BERT baseline).

## Architecture

- **Dual-stream encoders** — RoBERTa-base (mean pooling) + Twitter-RoBERTa with DES self-attention ([CLS] pooling)
- **Learned fusion projection** — concatenation → 1536d → 384d bottleneck
- **SenticNet integration** — BFS-based label priors (depth=3) + affective residual from Hourglass dimensions
- **Adaptive label embeddings** — 28 learnable emotion nodes refined via sparse self-attention (top-k annealed 28→5)
- **Auxiliary Ekman objective** — 7-class coarse supervision with warmup (training only)

## Requirements

- Python 3.11+
- PyTorch
- HuggingFace Transformers & Datasets
- scikit-learn
- senticnet
- tqdm, matplotlib

## Quick Start

```bash
pip install torch transformers datasets scikit-learn senticnet tqdm matplotlib
```

Training and evaluation are implemented in Jupyter notebooks. Load the GoEmotions simplified split via HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("google-research-datasets/go_emotions", "simplified")
```

## Results

| Model | Macro F1 | Micro F1 |
|---|---|---|
| BERT-base baseline | 0.5057 | 0.5877 |
| RoBERTa-base baseline | 0.5157 | 0.5918 |
| Twitter-RoBERTa single-stream | 0.5277 | 0.6018 |
| **Proposed dual-stream hybrid** | **0.5407** | **0.6050** |

Notable per-class results: gratitude (0.92), amusement (0.82), love (0.81), grief (0.63 with only 77 training examples).

## License

This repository contains thesis research code. See individual files for details.

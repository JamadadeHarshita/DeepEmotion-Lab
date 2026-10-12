
# Fine-Grained Emotion Detection with Dual-Stream Transformers

Research implementation accompanying the paper **[Fusing General and Domain-Specific Transformers for Fine-Grained Emotion Detection](https://doi.org/10.1049/sfw2/5612036)**, published in *IET Software (2026)*.

## Overview

This project investigates multi-label emotion classification using general-domain and social-media-adapted transformers, combined with affective commonsense knowledge.

The model is evaluated on [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), containing 27 emotion categories and one neutral label.

The research focuses on three challenges:

- Recognizing fine-grained emotions in informal text.
- Handling severe class imbalance and rare emotions.
- Measuring the contribution of individual model components under consistent evaluation conditions.

## Architecture

The proposed model combines:

- **Dual-stream encoders:** RoBERTa-base and Twitter-RoBERTa for general and domain-specific text representations.
- **Representation fusion:** Learned projection of both encoder outputs into a 384-dimensional representation.
- **SenticNet integration:** Bounded breadth-first search (depth 3) for knowledge-based label priors, alongside an affective residual.
- **Adaptive label attention:** Learnable emotion embeddings with sparse self-attention to model label dependencies.
- **Auxiliary supervision:** Seven-class Ekman-style classification as a training-time regularizer.
- **Class-weighted learning:** Damped positive class weighting to address long-tail imbalance.

## Evaluation

Experiments use the official GoEmotions simplified split.

| Split | Samples |
|---|---:|
| Training | 43,410 |
| Validation | 5,426 |
| Test | 5,427 |

Evaluation includes:

- Per-class decision thresholds calibrated on validation data.
- A shared evaluation procedure across baselines and ablations.
- Ten-seed replication for principal comparisons.
- Paired statistical significance tests.
- Component-wise ablation studies.
- Inference cost and memory measurements.

**Macro F1** is the primary metric, with micro F1 reported as a secondary measure.

## Results

### GoEmotions

Results below use per-class threshold calibration.

| Model | Macro F1 |
|---|---:|
| BERT-base | 0.5262 |
| RoBERTa-base | 0.5195 |
| Twitter-RoBERTa | 0.5290 |
| RoBERTa + inverse-frequency weighting | 0.5312 ± 0.0049 |
| Dual-stream without SenticNet | 0.5300 ± 0.0087 |
| **Proposed model** | **0.5368 ± 0.0040** |

Results with ± denote mean and standard deviation across 10 seeds. The unweighted single-stream results are individual runs.

The proposed model improved macro F1 by **0.0056** over the strongest calibrated single-stream baseline, at approximately twice the inference cost.

### Key findings

1. **Threshold calibration matters.** Per-class calibration improved macro F1 by approximately 0.095 for the inverse-frequency-weighted RoBERTa baseline, changing the interpretation of its effectiveness.

2. **Class weighting was the strongest intervention.** Removing positive class weighting reduced macro F1 by 0.0574 on average across 10 seeds. The largest improvements occurred in rare categories such as pride, grief, and relief.

3. **Commonsense knowledge provided a modest improvement.** SenticNet integration contributed an average gain of 0.0068 macro F1 over the knowledge-ablated model.

4. **Dual-stream fusion alone was insufficient.** Without knowledge integration, the additional encoder did not meaningfully outperform the strongest calibrated single-stream baseline.

5. **Accuracy comes with computational cost.** The full architecture uses approximately twice the parameters and inference latency of a single-stream RoBERTa model.

### Additional benchmark

The architecture was also trained and evaluated independently on **SemEval-2018 Task 1 E-c**, achieving a **Jaccard index of 0.613**.

This is a second-dataset evaluation, not zero-shot transfer.


Training and evaluation are implemented in Jupyter notebooks.


## Publication

**Binh Vu, Harshita Jamadade, Swati Chandna, Mehrdad Jalali.**

*Fusing General and Domain-Specific Transformers for Fine-Grained Emotion Detection.*

IET Software, 2026, Article ID 5612036.

**DOI:** https://doi.org/10.1049/sfw2/5612036

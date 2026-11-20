# Pure-Acoustic Non-Native (L2) Phoneme Recognition: Hybrid WavLM + ConBiMamba + Frame-Level LoRA-MoE

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/Transformers-WavLM--Base--Plus-yellow?style=flat-square)](https://huggingface.co/microsoft/wavlm-base-plus)
[![Task](https://img.shields.io/badge/Task-L2%20Phoneme%20Recognition%20%2F%20MDD-blue?style=flat-square)]()
[![Full Test Strict PER](https://img.shields.io/badge/Full%20Test%20Strict%20PER-13.15%25-success?style=flat-square)]()
[![Allophone Equivalence PER](https://img.shields.io/badge/Allophone%20Equivalence%20PER-9.20%25-brightgreen?style=flat-square)]()

---

## 📌 1. Project Overview & Aim

This project develops a **pure-acoustic phoneme recognition system** as the first stage of a larger **Mispronunciation Detection and Diagnosis (MDD)** pipeline. The recognizer aims to represent the phonemes perceived in non-native (L2) speech rather than reconstructing words through an external language model.

### Core Objectives

1. **Acoustic Phoneme Recognition:** Decode phoneme sequences from acoustic model outputs without an external language model or pronunciation dictionary during inference.
2. **Perceived-Phone Supervision:** Use L2-Arctic perceived IPA annotations for L2 adaptation, converted to the 39-class ARPAbet inventory through a custom IPA-to-ARPAbet parser.
3. **Parameter-Efficient Adaptation:** Combine WavLM-Base-Plus, learnable hidden-layer fusion, six-expert frame-level LoRA-MoE modules, and custom bidirectional acoustic sequence blocks.
4. **Acoustic Decoding:** Use CTC prefix beam search with pre-emphasis test-time augmentation (TTA), logit fusion, and Stage 2 stochastic weight averaging (SWA).

---

## 🏗️ 2. System Workflow & Pipeline

```text
+--------------------------------------------------------------------------+
| 1. DATA                                                                   |
| LibriSpeech train.100 + SpeechOcean762 train -> Stage 1 general training  |
| L2-Arctic scripted -> 80% train / 10% validation / 10% test (seed 42)     |
| L2-Arctic targets: perceived IPA -> custom ARPAbet mapping                |
+--------------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------------+
| 2. ACOUSTIC ENCODER                                                       |
| 16 kHz raw waveform -> WavLM-Base-Plus                                    |
| Learnable fusion of 13 hidden representations                             |
| Six-expert frame-level LoRA-MoE in top six Transformer layers              |
| MoE adapters in intermediate_dense and output_dense projections           |
+--------------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------------+
| 3. SEQUENCE MODEL                                                         |
| Four custom bidirectional gated/convolutional acoustic blocks              |
| Intermediate CTC head after Block 2; final CTC head after Block 4          |
| 41 output classes: 39 ARPAbet phones + blank + unknown                     |
+--------------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------------+
| 4. TWO-STAGE TRAINING                                                     |
| Stage 1: 800 updates on interleaved LibriSpeech + SpeechOcean762           |
| Stage 2: 700 updates adapting to L2-Arctic; SWA over 21 checkpoints        |
+--------------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------------+
| 5. INFERENCE                                                              |
| Normalized + pre-emphasized waveform passes                               |
| 0.65 / 0.35 acoustic logit blend; 0.92 / 0.08 final/intermediate blend     |
| Temperature 0.88; blank adjustment 0.25                                   |
| CTC prefix beam search: beam 12, top-8 tokens; no external LM              |
+--------------------------------------------------------------------------+
```

### Model Components

- **Backbone:** `microsoft/wavlm-base-plus` with learnable fusion of its 13 hidden representations.
- **Frame-level LoRA-MoE:** Six experts per router, rank 8 and alpha 16, inserted into `intermediate_dense` and `output_dense` in the top six WavLM Transformer layers (12 routers total).
- **Sequence encoder:** Four custom bidirectional gated/convolutional acoustic blocks (`d_model=256`). These are Mamba-inspired custom blocks, not a canonical Mamba selective state-space implementation.
- **CTC heads:** Intermediate supervision after the second block and final prediction after the fourth block.

### Training

- **Stage 1:** 800 updates on interleaved LibriSpeech and SpeechOcean762. WavLM parameters are frozen while the downstream projection, sequence encoder and CTC heads are trained.
- **Stage 2:** 700 updates on shuffled L2-Arctic training utterances, with differential learning rates for acoustic-backbone and task-specific parameters. SWA averages 21 checkpoints sampled every 10 updates over the final 200 updates.

---

## 📊 Results & Analysis

### Full L2-Arctic Test Set

| Metric | Result |
|---|---:|
| Test utterances | 360 |
| Speakers / L1 groups | 24 / 6 |
| Reference phonemes | 11,505 |
| **Strict 39-class PER** | **13.1508%** |
| Allophone-equivalence PER | 9.1952% |

| Error type | Count |
|---|---:|
| Substitutions | 1,110 |
| Deletions | 193 |
| Insertions | 210 |
| Hits | 10,202 |

The allophone-equivalence score applies the predefined mapping `DH→D`, `IY→IH`, `Z→S`, `AO→AA`, `AH→AA`, `ER→AH`, and `G→K`, and collapses adjacent repeated labels. It is a more permissive supplementary metric; strict PER is the primary result.

### L1-Wise Strict PER

| L1 group | Pooled PER |
|---|---:|
| Chinese | 14.77% |
| Hindi | 14.07% |
| Vietnamese | 13.62% |
| Spanish | 13.23% |
| Arabic | 12.04% |
| Korean | 11.37% |

Each group contains four speakers. These differences are descriptive and are not evidence of statistically significant L1 effects.

### Phoneme-Level Error Analysis

- Highest reference error rates among frequently observed phones include **DH (42.53%)**, **AA (36.02%)**, **Z (28.97%)**, and **AO (26.44%)**.
- Prominent reciprocal substitution pairs: **DH↔D (106)**, **Z↔S (101)**, and **IY↔IH (98)**.
- Most frequent deletions: AH (36) and R (26).
- Most frequent insertions: D (38) and T (28).

### LoRA-MoE Routing Analysis

Routing probabilities were collected for 12 routers (six layers × two projections), each containing six experts, across 360 test utterances and 24 speakers.

The exploratory between-L1/within-L1 variability analysis highlighted these combinations based on both variability ratio and absolute L1-mean range:

| Router–expert | Between/within ratio | L1 mean range |
|---|---:|---:|
| L7 intermediate_dense – E6 | 3.395 | 0.1696 |
| L6 intermediate_dense – E5 | 3.224 | 0.1324 |
| L7 intermediate_dense – E2 | 3.163 | 0.0929 |
| L8 intermediate_dense – E3 | 2.892 | 0.0983 |

These are descriptive candidates for further investigation, not proof of accent-specific expert specialization. There are four speakers per L1 group, and no statistical significance or causal L1 effect is claimed.

### Baseline Comparison Note

An earlier Conformer/G2P baseline was reported at 33.30% test PER. It is not a controlled, directly comparable ablation: the model setup and L2 target-generation method differ from the current perceived-IPA-supervised system. The 13.1508% result above is the verified full-test result for the current model and evaluation pipeline.

---

## Repository Structure

The project is organized into the following parts:

1. **Environment & Authentication:** Dependencies, Hugging Face authentication and checkpoint access.
2. **Vocabulary & Data:** 41-class ARPAbet mapping, LibriSpeech, SpeechOcean762 and L2-Arctic preparation.
3. **Model Architecture:** WavLM, learnable layer fusion, frame-level LoRA-MoE and custom acoustic sequence blocks.
4. **Training:** Stage 1 general-speech training and Stage 2 L2-Arctic adaptation with SWA.
5. **Inference & Evaluation:** Acoustic TTA, CTC prefix beam decoding and full-test PER.
6. **Post-Training Analysis:** Expert routing, speaker/L1 performance and phoneme-level error analysis.

Model checkpoint: [Hugging Face model repository](https://huggingface.co/hsuyip05/wavlm-conbimamba-lora-moe)

Analysis artifacts are maintained separately in the private Hugging Face Dataset repository:
`hsuyip05/mdd-analysis-artifacts`.

---

## Interactive Inference

The original development notebook contains an experimental Gradio interface for audio upload/microphone input, ARPAbet sequence prediction, optional reference PER, and an acoustic posterior heatmap. This interface is a phoneme-recognition demonstration; it does not implement complete phone-level mispronunciation localization or articulatory diagnosis.

---

## Known Limitations

- The L2-Arctic split is utterance-level, not speaker-disjoint; results should not be described as evaluation on entirely unseen speakers.
- Each L1 group contains four speakers, so L1 comparisons and routing variability are exploratory.
- The custom IPA-to-ARPAbet parser is many-to-one for some symbols and skips unmatched IPA characters; perceived-phone distinctions may therefore be lost.
- The custom bidirectional sequence block is Mamba-inspired but is not a canonical selective state-space Mamba implementation.
- The current system recognizes acoustic phoneme sequences. Full expected-versus-perceived phone alignment, error localization, and articulatory diagnosis remain future MDD stages.
- As an acoustic-only recognizer, performance may be affected by noise, recording quality and channel mismatch.

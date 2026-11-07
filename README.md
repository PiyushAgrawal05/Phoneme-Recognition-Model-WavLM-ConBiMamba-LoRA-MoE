# Pure-Acoustic Non-Native (L2) Phoneme Recognition: Hybrid WavLM + ConBiMamba + Frame-Level LoRA-MoE

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-WavLM--Base--Plus-yellow?style=flat-square)](https://huggingface.co/microsoft/wavlm-base-plus)
[![Task](https://img.shields.io/badge/Task-L2%20Phoneme%20Recognition%20%2F%20CAPT-blue?style=flat-square)]()
[![Test Strict PER](https://img.shields.io/badge/Unseen%20Test%20Strict%20PER-12.91%25-success?style=flat-square)]()
[![Test Equivalence PER](https://img.shields.io/badge/L2%20Equivalence%20PER-8.56%25-brightgreen?style=flat-square)]()

---

## 📌 1. Project Overview & Aim

In **Computer-Assisted Pronunciation Training (CAPT)** and **Mispronunciation Detection and Diagnosis (MDD)**, a phoneme recognizer must transcribe *what the speaker actually pronounced*, not what they were supposed to say. 

Standard Automatic Speech Recognition (ASR) pipelines rely heavily on external **Language Models (LMs)** or top-layer Transformer representations that carry strong lexical bias. When a non-native (L2) speaker mispronounces a word (e.g., pronouncing *"think"* as *"sink"* or *"three"* as *"tree"*), LM-biased models autocorrect the phoneme sequence back to the canonical dictionary pronunciation, rendering them useless for pronunciation assessment. Furthermore, training models on Grapheme-to-Phoneme (`G2P`) dictionary targets forces the acoustic encoder to hallucinate unpronounced phonemes on accented speech.

### Core Objectives
1. **100% Pure-Acoustic Modeling (Zero Language Model):** Build an end-to-end acoustic phoneme recognition system that decodes purely from acoustic frame probabilities without any lexical or language model prior.
2. **Perceived Acoustic Ground-Truth Alignment:** Replace canonical dictionary prompts (`G2P`) on non-native datasets with a custom longest-first greedy **IPA-to-ARPAbet parser** over human-annotated perceived phonetic transcriptions, ensuring all L2 allophones (taps, glottal stops, non-rhotic vowels) are preserved.
3. **Dynamic Frame-Level Accent Adaptation:** Design a parameter-efficient **6-Expert Frame-Level LoRA Mixture-of-Experts (LoRA-MoE)** injected into `WavLM-Base-Plus`, paired with a **13-Layer Learnable Acoustic Fusion** mechanism and a **ConBiMamba (Conformer + Bidirectional Selective State-Space)** sequence encoder.
4. **High-Precision Acoustic Decoding:** Implement a pure-acoustic **CTC Prefix Beam Search Decoder** enhanced with **Stochastic Weight Averaging (SWA)** and **Pre-Emphasis Test-Time Augmentation (TTA)** to drive unseen L2-Arctic Phoneme Error Rate (PER) down from a **33.30% baseline to 12.91% (Strict 39-Class PER)** and **8.56% (L2 Allophone-Equivalence PER)**.

---

## 🏗️ 2. System Workflow & Pipeline

```text
+-----------------------------------------------------------------------------------+
|                        1. MULTI-CORPUS DATA INGESTION                             |
|  • LibriSpeech (train.100)  --> Native English Baseline (G2P Targets)             |
|  • SpeechOcean762 (train)   --> Non-Native Expansion (Human Phone Annotations)    |
|  • L2-Arctic (scripted)     --> Target L2 Domain (80% Train / 10% Val / 10% Test) |
|                                 Custom Perceived IPA-to-ARPAbet Greedy Parser     |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                     2. AUDIO NORMALIZATION & FRONT-END                            |
|  • 16kHz Raw Waveform (<= 10s–15s) -> Per-Utterance Zero-Mean Unit-Variance Norm  |
|  • Pre-trained WavLM-Base-Plus CNN Encoder (320x downsampling -> 20ms stride)     |
|  • 6-Expert Frame-Level LoRA-MoE injected into Top 6 Transformer FFN Blocks       |
|  • LayerNorm-Stabilized Learnable Weighted Sum across all 13 Hidden States        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                     3. CONBIMAMBA SEQUENCE ENCODER (4 BLOCKS)                     |
|  • Linear Projection (768 -> 256) + Spatial Dropout1d (p = 0.10)                  |
|  • Block 1 & 2: Macaron FFN + Bi-Directional Selective Scan (Leaky Coarticulation)|
|                 + Depthwise Separable Conv1d (K = 31) + Macaron FFN               |
|  • [Intermediate CTC Head @ Block 2] --> 41 Classes (Aux Supervision)             |
|  • Block 3 & 4: ConBiMamba Processing                                             |
|  • [Final CTC Head @ Block 4]        --> 41 Classes (Primary Supervision)         |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                 4. TWO-STAGE CURRICULUM TRAINING & SWA                            |
|  • Stage 1 (Generalization): 800 Updates on Interleaved LibriSpeech + SO762       |
|    (WavLM & LoRA Frozen; Train ConBiMamba + Heads; Joint CTC: 0.7 Final/0.3 Inter)|
|  • Stage 2 (L2 Adaptation): 700 Updates on In-Memory Speaker-Balanced L2-Arctic   |
|    (Unfreeze Feature Proj, 12 WavLM Layers, LoRA-MoE, Layer Weights + SWA)        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|               5. ACOUSTIC TTA ENSEMBLE & PREFIX BEAM DECODING                     |
|  • Pass 1: Standard Normalized Waveform  | Pass 2: Pre-Emphasized Waveform (0.95) |
|  • Logit Blend (0.65 Pass1 + 0.35 Pass2) + Inter-CTC Blend (0.92 Final/0.08 Inter)|
|  • Temperature Scaling (T = 0.88) + Blank Penalty (-0.25)                         |
|  • Pure-Acoustic CTC Prefix Beam Search (K = 12, Top-8 Pruning) -> 39 ARPAbet     |
+-----------------------------------------------------------------------------------+

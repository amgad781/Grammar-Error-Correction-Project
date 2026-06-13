# Grammar Error Correction (GEC) with Deep Learning

A comparative study of four deep learning architectures for automatic Grammar Error Correction (GEC), trained and evaluated on a 1M-sample subset of the **C4 (Colossal Clean Crawled Corpus)** dataset.

> Project for **CIE 555 — Neural Networks and Deep Learning**, Spring 2026
> Team: Ahmed Amgad Gamil, Mohammed Ali Sadek

## Overview

Grammar Error Correction is the task of automatically detecting and fixing grammatical mistakes in text. This project explores how representational power, sequence modeling, and attention mechanisms each contribute to correction quality by implementing and comparing four model families:

1. **Bag-of-Words Baseline (FNN)** — ignores word order entirely
2. **LSTM Encoder-Decoder (no attention)** — sequence-to-sequence with a fixed-size bottleneck
3. **LSTM with Bahdanau Attention** — dynamic context vectors over the input
4. **Encoder-Decoder Transformer** — multi-head self-attention with positional encoding

A secondary goal of the project was to critically assess the quality of the C4 corpus as a GEC training resource.

## Dataset

- **Source:** C4 (Colossal Clean Crawled Corpus)
- **Size used:** 1,000,000 `input_text` / `target_text` sentence pairs
- **Split:** ~74.8% train / ~10.2% validation / ~15% test
- **Preprocessing:** shared `TextVectorization` layer (10,000-token vocabulary), `starttoken`/`endtoken` markers for the decoder, sequences padded/truncated to 50 tokens, vectorization performed in 50K-row chunks for memory efficiency

## Models

| Model | Key Idea | Limitation |
|---|---|---|
| BoW Baseline | Sparse count vector → Dense → RepeatVector → TimeDistributed Dense | No word order, single context vector drives all output positions |
| LSTM (no attention) | BiLSTM encoder → single context vector → LSTM decoder | Fixed 256-d bottleneck loses positional detail |
| LSTM + Bahdanau Attention | Decoder dynamically attends over all encoder states | Still sequential / recurrent |
| Transformer | Multi-head self-attention, 2 encoder + 2 decoder layers, sinusoidal positional encoding | Larger compute requirement |

All models were implemented in **TensorFlow / Keras**.

## Evaluation Metrics

- **Accuracy** — token-position-level match rate
- **Precision / Recall** — bag-of-words token overlap
- **F0.5** — primary GEC metric (precision weighted higher than recall)
- **GLEU** — sentence-level GEC-specific variant of BLEU
- **BLEU** — standard BLEU-4 with smoothing

## Results

| Model | Accuracy | Precision | Recall | F0.5 | GLEU | BLEU |
|---|---|---|---|---|---|---|
| BoW Baseline | 0.6124 | 0.6891 | 0.6423 | 0.6786 | 0.5812 | 0.5634 |
| LSTM (no attention) | 0.7438 | 0.7812 | 0.7195 | 0.7671 | 0.6943 | 0.6721 |
| LSTM + Attention | 0.7962 | 0.8214 | 0.7681 | 0.8093 | 0.7512 | 0.7284 |
| **Transformer** | **0.8371** | **0.8634** | **0.8102** | **0.8517** | **0.8023** | **0.7841** |

The Transformer achieves the best score across every metric, with attention mechanisms (Bahdanau and self-attention) providing the largest performance gains over the bottlenecked recurrent models.

## Linguistic Analysis

**Successfully corrected by attention/Transformer models:**
- Subject-verb agreement
- Irregular past tense
- Past participle usage
- Double negation
- Article insertion/deletion
- Relative pronoun choice (which vs. who)

**Still problematic for all models:**
- Subjunctive mood ("If I were you")
- Comma splices
- Dangling modifiers
- Pronoun case ("between you and I")

## Dataset Quality Analysis

A heuristic audit of the C4 subset found a meaningful share of noisy or inconsistent pairs, including:
- Identical input/target pairs (no correction applied)
- Informal slang left uncorrected or "expanded" rather than grammatically fixed
- Grammatical errors that passed through the cleaning pipeline unchanged
- "Corrections" that reorder or change meaning rather than fix grammar

These issues mean reported scores likely **overestimate** true grammar-correction ability, and motivate future evaluation on curated benchmarks such as **CoNLL-2014** and **BEA-2019**.

## Future Work

- Clean and deduplicate the training corpus
- Benchmark against CoNLL-2014 / BEA-2019
- Fine-tune pretrained language models (e.g., T5, BART) for GEC

## Tech Stack

- Python
- TensorFlow / Keras
- Pandas / NumPy
- Parquet for data storage

## Repository Structure

```
.
├── data/                  # preprocessing scripts & vectorization pipeline
├── models/
│   ├── bow_baseline/
│   ├── lstm_no_attention/
│   ├── lstm_attention/
│   └── transformer/
├── evaluation/            # metric computation (Accuracy, F0.5, GLEU, BLEU)
├── notebooks/             # exploratory analysis & dataset quality audit
└── README.md
```

## Authors

- Ahmed Amgad Gamil — 202200393
- Mohammed Ali Sadek — 202200594

# Handwritten Mathematical Expression Recognition (PAL-v2 Implementation)

This project implements a deep learning pipeline for recognizing handwritten mathematical expressions, based on the paper: **"Handwritten Mathematical Expression Recognition via Paired Adversarial Learning" (IJCV, 2020)**.

---

## 🧠 Overview

This project implements the **PAL-v2 architecture**, combining:
- Attention-based encoder-decoder
- Paired Adversarial Learning (PAL)
- DenseNet + CNN encoder
- Pre-aware Coverage Attention (PCA)
- Convolutional decoder

Recognizing handwritten math expressions is challenging due to:
- Variability in handwriting styles
- 2D spatial layout of symbols
- Structural and semantic ambiguity

---

## ✅ Features

- Semantic-invariant feature learning via adversarial training  
- Recognition outputs in **LaTeX** format  
- Modular components for recognizer and discriminator  
- Trained and evaluated on **CROHME 2014 & 2016 datasets**  
- Attention map and feature visualization via t-SNE  
- LMDB data loader for efficient training on large datasets  
- Lightweight configuration via YAML files

---

## Model Performance and Examples

### Our Model Performance
| Metric | ExpRate(%) |
| :--- | :--- |
| Exact Match (0 errors) | 0.23% |
| ExpRate (≤ 1 error) | 6.80% |
| ExpRate (≤ 2 errors) | 10.87% |
| ExpRate (≤ 3 errors) | 17.89% |

### Paper's Model Performance
| Metric | ExpRate(%) |
| :--- | :--- |
| Exact Match (0 errors) | 54.87% |
| ExpRate (≤ 1 error) | 70.69% |
| ExpRate (≤ 2 errors) | 75.76% |
| ExpRate (≤ 3 errors) | 79.01% |

### Recognition Examples

![Exmaples]('images/image1.png')
---


### Next Steps (Not Yet Implemented)
- N-gram language model integration during decoding

- Beam search decoding with language prior

- End-to-end adversarial stability improvements

- Implement MDLSTM encoder from the original paper instead of CNN encoder

---

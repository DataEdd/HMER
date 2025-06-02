# Handwritten Mathematical Expression Recognition (PAL-v2 Implementation)

This project implements a deep learning pipeline for recognizing handwritten mathematical expressions, based on the paper **"Handwritten Mathematical Expression Recognition via Paired Adversarial Learning" (IJCV, 2020)**.

## Overview

Handwritten math expressions are challenging to interpret due to style variance, 2D layout, and symbol ambiguity. This implementation adopts the **PAL-v2 architecture**, combining:
- **Attention-based encoder-decoder**
- **Paired adversarial learning**
- **DenseNet and MDLSTM encoders**
- **Pre-aware coverage attention (PCA)**
- **Convolutional decoder**
- Optional **N-gram language model** for decoding

## Features

- Semantic-invariant feature learning via adversarial training
- Recognition output in LaTeX format
- Trained and evaluated on **CROHME** datasets
- Visualization of attention maps and symbol features using t-SNE
- Modular training of recognizer and discriminator

## Dependencies

- Python 3.8+
- PyTorch
- NumPy, SciPy, scikit-learn
- Tesseract 
- Matplotlib 

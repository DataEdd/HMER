# Handwritten Mathematical Expression Recognition (PAL-v2 Implementation)

This project implements a deep learning pipeline for recognizing handwritten mathematical expressions, based on the paper:  
**"Handwritten Mathematical Expression Recognition via Paired Adversarial Learning" (IJCV, 2020)**.

---

## 🧠 Overview

Recognizing handwritten math expressions is challenging due to:
- Variability in handwriting styles
- 2D spatial layout of symbols
- Structural and semantic ambiguity

This repo implements the **PAL-v2 architecture**, combining:
- Attention-based encoder-decoder
- Paired Adversarial Learning (PAL)
- DenseNet + CNN encoder
- Pre-aware Coverage Attention (PCA)
- Convolutional decoder

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

### Next Steps (Not Yet Implemented)
- N-gram language model integration during decoding

- Beam search decoding with language prior

- End-to-end adversarial stability improvements

- Implement MDLSTM encoder from the original paper instead of CNN encoder

## ⚠️ Data & Model File Handling

Due to **GitHub’s 100MB file size limit**, we do **not include** large files in this repository.  
This includes:
- `data/`: contains CROHME dataset, LMDB format
- `checkpoints/`: saved `.pt` model weights
- `palv2.pt`: final trained model

You must manually **recreate these folders** following the steps below.

## 🔁 How to Reproduce Results

### 1. Clone the Repository

```bash
git clone https://github.com/DataEdd/HMER.git
cd HMER
``` 
### 2. Setup environment and install dependencies

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare the Data

Download CROHME Dataset - https://www.isical.ac.in/~crohme/
Place the raw .inkml files into a folder like:
```bash
data/crohme2013/raw/
```

Preprocess to LMDB

```bash
mkdir -p data/crohme2013/processed/

python tools/create_lmdb.py \
  --input data/crohme2013/raw/ \
  --output data/crohme2013/processed/ \
  --split train,val,test
```
This will generate train.lmdb, val.lmdb, and test.lmdb.

### 4. Train the Model
```bash
python train.py --config configs/train.yaml
```


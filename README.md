
# Mini‑Xception‑Emo

A **lightweight convolutional network for real‑time facial‑expression recognition** based on the *Mini‑Xception* backbone enriched with residual depth‑wise separable blocks and *Squeeze‑and‑Excitation* attention.

---

## What’s inside?

| Component | Script | Highlights |
|-----------|--------|------------|
| **Training** | `code/train.py` | data‑augmentation (crop, rotation, flip, MixUp α = 0.2); **Focal Loss** + AdamW + One‑Cycle‑LR; mixed‑precision; early‑stopping (30 epochs patience). |
| **Model** | `code/model.py` | five residual DW‑Separable blocks with SE; global average pooling → dropout → FC; ≈ 1.1 M params, input **1 × 128 × 128**. |
| **Evaluation** | `code/evaluate.py` | accuracy with/without test‑time augmentation (horizontal flip); automatically restores the best checkpoint. |
| **GUI demo** | `code/app.py` | PyQt5 + OpenCV frontend; Haar‑cascade face detection + real‑time emotion prediction (7 classes); runs on CPU or GPU. |

---

## Repository structure

```text
.
├─ code/
│  ├─ app.py
│  ├─ evaluate.py
│  ├─ model.py
│  ├─ train.py
│  └─ utils.py
├─ data/
│  ├─ train/
│  └─ test/
├─ results/
│  ├─ best_model.pt   # auto‑generated
│  └─ README.md       # experiment logs
└─ README.md          # you’re reading it
```

---

## Installation

```bash
git clone https://github.com/romansafranko/Mini-Xception-Emo.git
cd Mini-Xception-Emo
pip install -r requirements.txt
```

---

## Training

The script expects images under `data/train/<label>/*.png` (same for `data/test/`).  
Run:

```bash
python code/train.py [output_dir]
```

*If* `output_dir` is **omitted** the script will fall back to the **default folder `results/`** and will save the best model as `results/best_model.pt` (see constant `DEFAULT_SAVE_DIR` in `train.py`).

---

## Evaluation

```bash
python code/evaluate.py [model_path]
```

If `model_path` is **not provided** **`results/best_model.pt`** is used automatically (`DEFAULT_MODEL_PATH` inside `evaluate.py`).

---

## GUI demo

```bash
python code/app.py [model_path]
```

No argument? The application again loads **`results/best_model.pt`** by default and immediately starts a 640 × 480 webcam preview with the predicted emotion rendered above each detected face.

---

## Results

On FER‑2013 the model reached **69.0 7%** top‑1 accuracy (without TTA) and **69.20 %** (with TTA).  
See `results/README.md` for full logs, confusion matrices, and training curves.

---

## License

MIT license – see `LICENSE`.

---

## Reference

*O. Arriaga, M. Valdenegro‑Toro, P. Plöger. “Real‑time Convolutional Neural Networks for Emotion and Gender Classification”, arXiv:1710.07557 (2017).*


# FER‑2013 dataset (already included)

This **`data/`** directory already contains the full **FER‑2013** dataset converted to images
and split into the canonical *train* / *test* folders
expected by the training and evaluation scripts.

```
data/
├── train/
│   ├── angry/      # 3 995 images
│   ├── disgust/    #   436 images
│   ├── fear/       # 4 106 images
│   ├── happy/      # 7 133 images
│   ├── neutral/    # 4 976 images
│   ├── sad/        # 4 078 images
│   └── surprise/   # 2 947 images
└── test/
    ├── angry/      #   958 images
    ├── disgust/    #   111 images
    ├── fear/       # 1 025 images
    ├── happy/      # 1 777 images
    ├── neutral/    # 1 248 images
    ├── sad/        # 1 067 images
    └── surprise/   #   831 images
```

*All images are grayscale `48 × 48 px` PNGs identical to the originals in the
Kaggle CSV; they have simply been extracted for faster loading.*

---

## Usage

The scripts in `code/` pick up this dataset **automatically** because their
default data root is set to `data/`.

- No additional download or conversion steps are required.

---

## Dataset details

| Split      | Images | Classes |
|------------|--------|---------|
| Train      | 28 709 | 7 |
| Test       |  3 589 | 7 |
| **Total**  | 32 298 | 7 |

Class labels: *angry, disgust, fear, happy, neutral, sad, surprise*.

---

## License

FER‑2013 is distributed under **CC BY‑NC‑SA 4.0**  
(non‑commercial, share alike, attribution required).

If you publish models trained on this data, please cite the original
Kaggle challenge accordingly.

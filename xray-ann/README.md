# Normal vs. Pneumonia: an ANN built from first principles

This project is a small educational image-classification experiment with a
fully-connected ANN/MLP written using NumPy.

The network's cost function, forward propagation, and backpropagation are
implemented in this project. SciPy supplies the conjugate-gradient optimizer,
and OpenCV loads and preprocesses the images.

## What it does

- Classifies the dataset's `NORMAL` and `PNEUMONIA` labels.
- Converts images to grayscale, resizes them to 10×10, flattens them to 100
  inputs, and scales pixels to [0, 1].
- Trains a two-hidden-layer network, with 10 units per hidden layer by
  default.
- Reports train, validation, and test accuracy in the terminal.

The default regularization coefficient is λ = 1.02, the optimizer is capped
at 50 iterations, and the random seed defaults to 42. A small, dense network
can still overfit; results are for learning and experimentation only.

## Dataset and split handling

The local archive is from Kaggle's
[*Chest X-Ray Pneumonia — Balanced Dataset*](https://www.kaggle.com/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset).
It contains 8,530 images: 4,265 per class. See
[DATASET_CITATION.md](DATASET_CITATION.md) for attribution and a note about
the dataset page's conflicting license statements.

The supplied train/validation/test folders contain augmented image variants
and images with the same `person…` identifier across folders. The supplied
test folder also contains only 15 images per class. To avoid source-group
leakage and rely on more than that small test subset, the loader pools the
folders and creates deterministic, approximately 80/10/10 splits by source
group. All variants of a normal image and all images sharing a pneumonia
`person…` identifier stay in one split. Training uses every image in its
groups; validation and test use one representative image per group, preferring
a non-augmented file when available. The archive itself is left untouched, and
reported metrics use these newly generated splits—not the original Kaggle
test folder.

Expected local layout:

```text
data/
└── archive/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

Image files are excluded from Git. This is an educational experiment, not a
medical device or diagnostic tool.

## Run on Windows

Open PowerShell in this folder and use Python 3.9 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe train_pneumonia_ann.py
```

To try a different configuration, pass options such as:

```powershell
.\.venv\Scripts\python.exe train_pneumonia_ann.py --hidden-size 20 --l2 0.5 --max-iterations 50 --seed 42
```

Use validation accuracy to compare settings; keep test accuracy for a final
check rather than repeatedly tuning against it. Training may take several
minutes. The script reports when the optimizer reaches its iteration limit
without confirmed convergence.

## Project files

- [train_pneumonia_ann.py](train_pneumonia_ann.py): ANN training and evaluation.
- [dataset_loader.py](dataset_loader.py): group-wise splitting and image loading.
- [tests/](tests/): loader and gradient-check tests.
- [ERRATA.md](ERRATA.md): current implementation caveats.
- [LESSONS_LEARNED.md](LESSONS_LEARNED.md): project takeaways.
- [analysis/historical_experiment_log.xlsx](analysis/historical_experiment_log.xlsx):
  supplementary experiment notes.

The project code is licensed under the MIT License in the repository root
([LICENSE](../LICENSE)). The license does not apply to third-party X-ray
images; see [DATASET_CITATION.md](DATASET_CITATION.md) before redistribution.

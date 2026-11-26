# Neural Network Image Classifiers

Two small image-classification projects I built to explore neural-network
fundamentals, implementing the models myself in NumPy from first principles.

I wrote the code in July–August 2022, using only my notes from Andrew Ng's
six-month Machine Learning course from Stanford on Coursera. I typed the
implementations myself without AI assistance, external tutorials, or copied
code.

| Project | Task | Trainer |
| --- | --- | --- |
| [X-ray ANN](xray-ann/README.md) | `NORMAL` vs. `PNEUMONIA` | `xray-ann/train_pneumonia_ann.py` |
| [Simpsons ANN](simpsons-ann/README.md) | Bart vs. Homer | `simpsons-ann/train_bart_homer.py` |

Each project has its own dependencies, dataset instructions, tests, citation,
and notes about its current limitations. Start with that project's README.

## Data and attribution

The image datasets are not included in this repository. Download them
separately, follow the instructions in the project READMEs, and review the
dataset citation and rights notes before reusing or redistributing any images.
The local data directories are excluded by `.gitignore`.

The X-ray project uses Kaggle's
[*Chest X-Ray Pneumonia — Balanced Dataset*](https://www.kaggle.com/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset).
See the project citation files for full dataset attribution and usage notes.

The `analysis/` folders contain historical experiment workbooks. Treat their
reported results as exploratory notes, not as independently reproduced
benchmarks.

## Usage

Run each project from its own directory and install only its listed
requirements. For example, in PowerShell:

```powershell
Set-Location .\xray-ann
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe train_pneumonia_ann.py
```

For Bart vs. Homer, use the matching instructions in
[simpsons-ann/README.md](simpsons-ann/README.md).

## Important

The X-ray classifier is an educational experiment, not a medical device or
diagnostic tool. The project code is licensed under the MIT License in
[LICENSE](LICENSE). That license does not apply to third-party datasets or
images; dataset and image rights are separate.

## Repository layout

```text
neural-network-image-classifiers/
├── xray-ann/
│   ├── analysis/
│   ├── data/
│   ├── tests/
│   ├── DATASET_CITATION.md
│   ├── dataset_loader.py
│   ├── train_pneumonia_ann.py
│   └── requirements.txt
└── simpsons-ann/
    ├── analysis/
    ├── data/
    ├── tests/
    ├── DATASET_CITATION.md
    ├── dataset_loader.py
    ├── train_bart_homer.py
    └── requirements.txt
```

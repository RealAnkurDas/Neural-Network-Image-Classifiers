# Bart vs. Homer: a handwritten ANN

After finishing an Andrew Ng machine-learning course, I had some time before
starting my BTech in computer science. I wanted to see if I could take the
ideas I had just studied and build a neural network myself, rather than only
use a ready-made model. Bart and Homer made a fun first image-classification
problem.

This project contains a fully-connected neural network (ANN/MLP) for
classifying Bart and Homer. Its cost function and backpropagation are
implemented with NumPy; SciPy performs conjugate-gradient optimization, and
OpenCV reads and preprocesses the images.

## Current model

- Input: 35×35 grayscale image, flattened to 1,225 normalized values.
- Architecture: two hidden layers of 100 units, with two output units.
- Regularization: L2 coefficient λ = 1.6.
- Optimizer: SciPy conjugate gradient, capped at 50 iterations.
- Output: optimizer status, cost, training accuracy, and test accuracy in the
  terminal.

Weights are initialized randomly on each run, so results vary. The optimizer
may stop at its iteration limit without converging. The trainer does not save
weights or use a validation split, so use validation data for parameter
selection and reserve the supplied test split for final evaluation.

## Dataset

The images come from Kaggle's
[*The Simpsons (Bart and Homer) Data*](https://www.kaggle.com/datasets/williamu32/dataset-bart-or-homer).
The dataset contains 196 training images (118 Bart, 78 Homer) and 73 test
images (42 Bart, 31 Homer). Place the extracted images in this layout:

```text
data/
└── archive/
    └── dataset_personagens/
        └── dataset_personagens/
            ├── training_set/
            │   ├── bart/
            │   └── homer/
            └── test_set/
                ├── bart/
                └── homer/
```

See [DATASET_CITATION.md](DATASET_CITATION.md) for the full citation and
licensing and redistribution cautions. Image files are excluded from Git.

## Run it on Windows

Open PowerShell in this folder. With Python 3.9 or newer available, create an
environment, install dependencies, run the tests, and start training:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe train_bart_homer.py
```

Training can take several minutes. Results vary because of random weight
initialization; the optimizer status indicates whether it converged or reached
its iteration limit.

## Project files

- [train_bart_homer.py](train_bart_homer.py): training and evaluation.
- [dataset_loader.py](dataset_loader.py): dataset loading and preprocessing.
- [tests/](tests/): loader tests.
- [ERRATA.md](ERRATA.md): current implementation caveats.
- [LESSONS_LEARNED.md](LESSONS_LEARNED.md): personal reflections.
- [analysis/historical_experiment_log.xlsx](analysis/historical_experiment_log.xlsx):
  supplementary experiment notes.

The project code is licensed under the MIT License in the repository root
([LICENSE](../LICENSE)). The license does not apply to third-party Simpsons
images; see [DATASET_CITATION.md](DATASET_CITATION.md) before redistribution.

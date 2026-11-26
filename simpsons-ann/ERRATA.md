# Errata and current limitations

These notes describe the code and data handling currently in this folder.

- The model is a fully-connected ANN/MLP, not a convolutional neural network.
- The trainer uses 35×35 grayscale images, two hidden layers of 100 units,
  L2 regularization λ = 1.6, and a 50-iteration maximum.
- Random initialization is not seeded, so results can differ between runs.
- The training loop can fit the training set more closely than the test set.
  The trainer has no validation split or early stopping; do not repeatedly
  tune parameters against the test set.
- The optimizer may reach its iteration limit without converging. Check the
  reported optimizer status along with the accuracy values.
- The trainer prints metrics but does not save model weights, predictions, or
  plots.
- The image files are not included in the repository. Kaggle's license
  metadata does not by itself establish rights to the underlying Simpsons
  episode imagery; see [DATASET_CITATION.md](DATASET_CITATION.md).

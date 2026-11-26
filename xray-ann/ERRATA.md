# Errata and current limitations

These notes describe the code and data handling currently in this folder.

- The classifier is a fully-connected ANN/MLP, not a convolutional network.
- Its classes are `NORMAL` and `PNEUMONIA`; it does not classify COVID-19.
- The default input is 10×10 grayscale. This is intentionally small for the
  handwritten experiment and discards substantial image detail.
- Supplied split folders contain related images across splits. The current
  loader therefore ignores those assignments and makes reproducible group-wise
  splits using normal-image augmentation suffixes and pneumonia `person…`
  identifiers. This is a heuristic based on the available filenames.
- The supplied test folder has only 15 images per class. Reported test metrics
  use the newly generated group-wise test split, not that original folder.
- The current trainer reports train, validation, and test accuracy but does
  not save model weights, predictions, or plots.
- The default optimizer cap is 50 iterations; a completed run can reach that
  cap without confirmed convergence.
- The random seed makes one run reproducible, but does not make the measured
  accuracy representative of other datasets or clinical settings.
- The underlying images are pediatric chest X-rays. This project is for
  education only and must not be used for diagnosis.
- Kaggle's license metadata and its description disagree. The image files are
  excluded from Git; see [DATASET_CITATION.md](DATASET_CITATION.md) before
  redistributing any data.

# What I learned

This project taught me that getting an image classifier to run is not the same
as evaluating it reliably.

## The data pipeline is part of the model

Image decoding, grayscale conversion, resizing, normalization, class order,
and source grouping all affect what the network learns. These assumptions
need to be explicit and tested.

## Related images must stay together

Augmented versions of one X-ray, or multiple images associated with one
patient, should not be divided across training and evaluation sets. Otherwise
the model can appear to generalize by recognizing familiar source images.

## Metrics need context

Training accuracy alone does not measure generalization. Validation data is
for selecting settings; test data should be reserved for a final evaluation.
Even then, a score on one small dataset is not evidence of clinical
performance.

## Implementing the math is a useful first step

Writing forward propagation, cross-entropy, regularization, and
backpropagation helped connect the equations to code. Gradient checks and
focused data-loader tests make that implementation easier to trust.

# What I learned

I started this after finishing an Andrew Ng machine-learning course, while I
was waiting for my undergraduate program to begin. I had learned the ideas in
the course and wanted to find out what it took to make them work on real
images.

## The math is only one part of the project

Writing the forward pass, cost function, and backpropagation helped me connect
the equations to actual code. I also learned that image loading, resizing,
normalization, and consistent labels are part of the model pipeline—not
unimportant setup.

## Data contracts matter

The network's two output units only make sense if every split uses the same
two class labels. A mismatch between labels and output indices can invalidate
training even when the network code itself runs. Explicit class order and
focused loader tests make that assumption visible.

## Evaluation needs discipline

Training accuracy is not a measure of how well a model generalizes. Test data
should stay separate from training, and an extra folder of images should not
be treated as a meaningful benchmark unless its provenance and relationship to
the official splits are known.

## A reported number needs a reproducible run

An accuracy figure is much more useful when the dataset split, preprocessing,
architecture, optimizer settings, random seed, and saved model are known.
This project preserves historical results as notes, but does not present them
as verified because the original run cannot be reconstructed reliably.

## Learning implementation versus building a tool

Implementing the network by hand was a good way to learn. It also exposed what
I would improve next: vectorize the calculations, make runs reproducible,
validate gradients, save the trained model, and separate training from
evaluation. Those are follow-up goals, not features this project claims to
have.

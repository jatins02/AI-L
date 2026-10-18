# AI-L

A collection of machine-learning experiments for exploring automatic differentiation, neural networks, optimization, and prediction with Python.

## Experiments

- `Neural Nets/Model1 (simple NN)/fair/`: scalar autodiff engine, MLP, optimizer demonstrations, and an MNIST experiment.
- `Convolutional Neural Net/`: a six-class scene classifier using PyTorch and the Intel image dataset.
- `Linear Regression Model/`: a wine-quality classification experiment despite its folder name; includes a WineQT CSV.
- `Makemore/Bigram model - makemore1/`: character-pair counting and an unfinished notebook.
- Decision Trees and Tensorboard folders are placeholders.

## Try the optimizer demo

From the repository root, using an activated Python environment:

```sh
python -m pip install numpy
cd "Neural Nets/Model1 (simple NN)/fair"
python optimise_surfaces.py
```

This prints optimization progress for a simple mathematical surface. Other experiments use PyTorch/torchvision, pandas, scikit-learn, TensorFlow/Keras, or Jupyter and have separate dataset requirements. Dataset source URLs are recorded in the relevant scripts/notes.

## Status

Learning work in progress: MNIST evaluation is broken, wine preprocessing needs a train/test separation fix, and the bigram notebook has an incomplete cell. Historical accuracy notes are not reproducible benchmark results. Training scripts may load data and train immediately when run or imported.

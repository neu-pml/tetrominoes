# Tetrominoes dataset

This repository contains the Tetrominoes dataset, used to assess the generalization performance of variational autodencoders. 

If you use this dataset in your work, please cite it as follows:

## Bibtex

```
@misc{tetrominoes19,
author = {Alican Bozkurt and Babak Esmaeili and Jennifer Dy and Dana Brooks and Jan-Willem van de Meent},
title = {Tetrominoes dataset},
howpublished= {https://github.com/neu-pml/tetrominoes/},
year = "2019",
}
```

## Description

Tetrominoes is a dataset of 2D shapes procedurally generated from 6 ground truth
independent latent factors. These factors are *rotation*, *color*, *scale*, *x* and *y* positions, and *shape*.

## Generating the Tetromino dataset

To generate and save the Tetromino dataset, run:

```
python generate_tetromino.py <data_path>
```

Where `data_path` is the path of the directory where the data will be downloaded. This will create two files: `id_tetrominos.pkl` and `ood_tetrominos.pkl` corresponding to in-domain and out-of-domain settings with 50% split.

You can pass training ratio values as additional arguments in order to generate in-domain Tetrominos with different ratios of training-test splits.  

```
python generate_tetromino.py <data_path> 0.1 0.3 0.6
```

where `[0.1, 0.3, 0.6]` are ![\frac{N_{train}}{N_{train} + N_{test}}](https://latex.codecogs.com/svg.latex?\frac{N_{train}}{N_{train}&space;&plus;&space;N_{test}}) ratios.

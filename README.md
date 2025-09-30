# D-LIM (Direct-Latent Interpretable Model)

# Overview

D-LIM (Direct-Latent Interpretable Model) is a neural network that enhances
genotype-fitness mapping by combining interpretability with predictive accuracy.
It assumes independent phenotypic influences of genes on fitness, leading to
advanced accuracy and insights into phenotype analysis and epistasis. The model
includes an extrapolation method for better understanding of genetic
interactions and integrates multiple data sources to improve performance in
low-data biological research.


# System Requirements

## Hardware requirements
   **D-LIM** requires only a standard computer with enough RAM to
   support the in-memory operations. 

## Software requirements
   This package is supported for Linux. The package has been tested on the following systems:
   - Linux: Ubuntu 20.04

## Python Dependencies
   **D-LIM** depends primarily on **pytorch**, as well as
   the components of the Python scientific stack:
   - ~pandas~
   - ~numpy~

## Installation guide
- Install the package from Pypi:
```
pip install dlim
```

Or install it from the sources:
```
pip install .
```

   
# Manuscript reproduction
  Source code to reproduce the analysis of the **D-LIM** manuscript are
  available at [Reproducibility for figures](https://github.com/LBiophyEvo/D-LIM-model.git).
  
# License
  This project is covered under the *MIT License*
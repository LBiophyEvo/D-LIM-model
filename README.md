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
   - `pandas`
   - `numpy`

## Installation guide
- Install the package from Pypi:
```
pip install dlim
```

Or install it from the sources:
```
git clone https://github.com/LBiophyEvo/D-LIM-model.git
cd D-LIM-model 
pip install -e .
```

## Documentation 
The official documentation with usage is available at: https://d-lim.readthedocs.io/en/latest/. Documentation covers instructions for running D-LIM on simulated and experimental data, including a some demonstrations on some simple datasets. 

# Data avaibility 
- Simulated data: see `src_simulate_data\`
- Experimental data:
   - Data from [Kemble et al.](https://www.science.org/doi/10.1126/sciadv.abb2236) can be found in `data\`. 
   - Data of protein-protein interaction is from [Diss et al.](https://elifesciences.org/articles/32472).
   - Data of yeast with different mutations and enviroments is from [Kinsler et al.](https://elifesciences.org/articles/61271). 
You can download data from our Google driver link, [Data used in D-LIM](https://drive.google.com/drive/folders/1hwixojm3thyYpf8X6qPG7NIvxQseFDKz?usp=sharing).

# Manuscript reproduction
  Source code to reproduce the analysis of the **D-LIM** manuscript are
  available at `reproducibility\` folder in [D-LIM GitHub](https://github.com/LBiophyEvo/D-LIM-model.git).
  
# License
  This project is covered under the *MIT License*
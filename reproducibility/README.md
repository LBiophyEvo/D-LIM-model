# Reproducibility Instructions

This folder contains Jupyter notebooks and scripts to reproduce the main figures and supplementary analyses for the D-LIM manuscript.

## Contents

### Jupyter files 
All the figures will be save in `\img`. 
- **figure2.ipynb**  
  Prediction on three simulated data: 
    - linear regulation of two genes (see [Kemble et al.](https://www.science.org/doi/10.1126/sciadv.abb2236) )
    - geometric model with Gaussian (see [Tenaillon et al.](https://www.annualreviews.org/content/journals/10.1146/annurev-ecolsys-120213-091846))
    - regulatory cascade model (see [Nghe et al.](https://www.nature.com/articles/s41467-018-03644-8))

- **figure3&SI3.ipynb**  
  Extrapolation with unseen mutations and measured phenotypes.  
  Tests on biological and exponential models.

- **figure4&6.ipynb**  
  Application to experimental data from Kemble et al. (2020).  
  Prediction of fitness and epistasis.

- **figure5.ipynb**  
  Visualize the performance of different models on Kemble et al. data. 

- **figure7.ipynb**  
  Application to protein-protein interaction data (Lehner et al., 2018).

- **figure8.ipynb**  
  Additional analyses and visualizations for yeast with different mutations in different environment datasets.


### models_comp
This folder is used for benchmarking D-LIM and other state-of-the-art models such as ALM, LANTERN, MAVE-NN. 

You can run all the model using:
   ```bash
   bash run_all.sh
   ```
Then visualize the results using `figure5.ipynb`. 

### pretrain_models
This folder is used for saving the trained model for plotting the figures. 


### stability_test Folder
This folder is used for the robustness analysis on simulated and experimental datasets. It is for reproducing `Fig. 9`, `Fig SI. 8` and `Fig SI. 9` in the paper.  These scripts allow: 
1. Trains multiple instances of DLIM with and without spectral regularization. Saved pretrained models in: ./pretrained_model/<data_flag>/
2. Computes embedding similarities across runs.
3. Plots the cosine similarity distribution of infered phenotype.


- `get_similarity_realdata.py`
   This script is used to get embedding similarity, allowing for data from Kemble et al. and Kinsler et al. 

- `get_similarity_simulated.py`
   This script is used to get embedding similarity, allowing for simulated data: cascade, bio, or tgaus. 

- `stability_realdata.py`
   This script is used to get performance of D-LIM with and without spectral initialization, allowing for data from Kemble et al. and Kinsler et al. 

- `stability_simulated.py`
   This script is used to get performance of D-LIM with and without spectral initialization, allowing for simulated data: cascade, bio, or tgaus. 

- Figures in: ./figures/
    * convergence_<data_flag>.png
    * conv_hist_<data_flag>.png
    * boxplot_<data_flag>_mse.png
    * boxplot_<data_flag>_pearson.png



## How to Use

1. **Install dependencies**  
   Make sure you have installed the D-LIM package and all required dependencies:
   ```bash
   pip install -e ..
   pip install -r ../requirements.txt
   ```

2. **Run notebooks**  
   Open each notebook in Jupyter or VS Code and run the cells sequentially to reproduce the figures.

3. **Data files**  
   The notebooks expect input data files in the `../data/` directory.  
   Please ensure all required CSV and Excel files are present.

4. **Output**  
   Figures and results will be saved in the `img/` subfolder.

## References

- Kemble, Harry, et al. "Flux, toxicity, and expression costs generate complex genetic interactions in a metabolic pathway." *Science Advances* 6.23 (2020): eabb2236.
- Diss, Guillaume, and Ben Lehner. "The genetic landscape of a physical interaction." *Elife* 7 (2018): e32472.

- Tonner, Peter D., Abe Pressman, and David Ross. "Interpretable modeling of genotype–phenotype landscapes with state-of-the-art predictive power." *Proceedings of the National Academy of Sciences* 119.26 (2022): e2114021119.

- Tareen, A., Kinney, J.B. "MAVE-NN: learning genotype–phenotype maps from multiplex assays of variant effect." Genome Biology 23, 248 (2022). https://doi.org/10.1186/s13059-022-02661-7

## Contact

For questions or issues, please contact the repository maintainer or open an issue on GitHub.

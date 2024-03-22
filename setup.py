"""

D-LIM (Direct Latent Interpretable Model): An interpretable neural network for
mapping genotype to fitness.

D-LIM employs a constrained latent space to map genes to single-value
dimensions, enabling the extrapolation to new genotypes and capturing the
non-linearity in genetic data. Its design facilitates a deeper understanding of
genetic mutations and their impact on fitness, making it highly applicable in
molecular adaptations.

The model's strengths include its interpretability, ability to handle
real-world genetic datasets, qualitative assessment of gene-gene interactions,
and incorporation of diverse data sources for improved performance in
data-scarce scenarios.
"""

from setuptools import setup, find_packages

setup(
    name='dlim',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "pandas",
        "torch",
        "numpy",
    ],
    # Other metadata
    author='Shuhui Wang, Alexandre Allauzen, Philippe Nghe, Vaitea Opuu',
    author_email='vaiteaopuu@gmail.com',
    description='Model genotype to fitness map',
    long_description=open('README.org').read(),
    license="MIT",
    url="https://github.com/LBiophyEvo/D-LIM-model"
)

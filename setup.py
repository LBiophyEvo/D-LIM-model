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
    name="dlim",
    version="1.1.0",
    description="Direct Latent Interpretable Model (D-LIM): An interpretable neural network for mapping genotype to fitness.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Shuhui Wang, Alexandre Allauzen, Philippe Nghe, Vaitea Opuu',
    author_email='vaiteaopuu@gmail.com',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here, e.g.:
        "numpy>=1.21.0",
        "torch>=1.10.0",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

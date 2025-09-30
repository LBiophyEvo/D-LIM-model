# Reproducibility Review

Below is a seven point reproducibility review prescribed by [Improving reproducibility and reusability in the
Journal of Cheminformatics](https://doi.org/10.1186/s13321-023-00730-y) of the `main` branch of
repository [https://github.com/LBiophyEvo/d-lim-model](https://github.com/LBiophyEvo/d-lim-model) (commit [`a04d01aa`](https://github.com/LBiophyEvo/d-lim-model/commit/a04d01aad7d69a476d0f916d0a920950e3a5bde1)),
accessed on 2025-09-30.

## 1. Does the repository contain a LICENSE file in its root?


Yes, MIT.


## 2. Does the repository contain a README file in its root?


No, a minimal viable README file contains:

- A short, one line description of the project
- Information on how to download, install, and run the code locally
- Brief documentation describing the single most important use case for the repository. For scientific code, this is
  ideally a one-liner in Python code, a shell script, or a command line interface (CLI) that can be used to reproduce
  the results of the analysis presented in a corresponding manuscript, use the tool presented in the manuscript, etc.
- Link to an archive on an external system like Zenodo, FigShare, or an equivalent.
- Citation information, e.g., for a pre-print then later for a peer reviewed manuscript

GitHub can be used to create a README file with
[https://github.com/LBiophyEvo/d-lim-model/new/main?filename=README.md](https://github.com/LBiophyEvo/d-lim-model/new/main?filename=README.md).
Repositories typically use the Markdown format, which is
explained [here](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).


## 3. Does the repository contain an associated public issue tracker?

Yes.

## 4. Has the repository been externally archived on Zenodo, FigShare, or equivalent that is referenced in the README?

No,  this repository does not have a README, and therefore it is not possible for a reader to tell if it is archived.

## 5. Does the README contain installation documentation?

No, this repository does not have a README, and therefore it is not possible for a reader to easily find installation
documentation.

## 6. Is the code from the repository installable in a straight-forward manner?

Yes.

### Packaging Metadata

[`pyroma`](https://github.com/regebro/pyroma) rating: 5/10

1. Your project does not have a pyproject.toml file, which is highly recommended.
You probably want to create one with the following configuration:

    [build-system]
    requires = [&#34;setuptools&gt;=42&#34;]
    build-backend = &#34;setuptools.build_meta&#34;

1. Your package does not have classifier data.
1. The classifiers should specify what Python versions you support.
1. You should specify what Python versions you support with the &#39;Requires-Python&#39; metadata.
1. Your package does not have keywords data.
1. The license &#39;MIT&#39; specified is not listed in your classifiers.
1. Your Description is not valid ReST: 
&lt;string&gt;:114: (WARNING/2) Bullet list ends without a blank line; unexpected unindent.
&lt;string&gt;:117: (WARNING/2) Bullet list ends without a blank line; unexpected unindent.
1. Specifying a development status in the classifiers gives users a hint of how stable your software is.

These results can be regenerated locally using the following shell commands:

```shell
git clone https://github.com/LBiophyEvo/d-lim-model
cd d-lim-model
python -m pip install pyroma
pyroma .
```


## 7. Does the code conform to an external linter (e.g., `black` for Python)?

No, the repository does not conform to an external linter. This is important because there is a large
cognitive burden for reading code that does not conform to community standards. Linters take care
of formatting code to reduce burden on readers, therefore better communicating your work to readers.

For example, [`black`](https://github.com/psf/black)
can be applied to auto-format Python code with the following:

```shell
git clone https://github.com/LBiophyEvo/d-lim-model
cd d-lim-model
python -m pip install black
black .
git commit -m "Blacken code"
git push
```


# Summary


Scientific integrity depends on enabling others to understand the methodology (written as computer code) and reproduce
the results generated from it. This reproducibility review reflects steps towards this goal that may be new for some
researchers, but will ultimately raise standards across our community and lead to better science.

Because the repository does not pass all seven criteria of the reproducibility review, I
recommend rejecting the associated article and inviting later resubmission after the criteria have all been
satisfied.



# Colophon

This review was automatically generated with the following commands:

```shell
python -m pip install autoreviewer
python -m autoreviewer LBiophyEvo/d-lim-model
```

Please leave any feedback about the completeness and/or correctness of this review on the issue tracker for
[cthoyt/autoreviewer](https://github.com/cthoyt/autoreviewer).
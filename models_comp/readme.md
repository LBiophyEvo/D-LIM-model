# Benchmark on D-LIM and the other state-of-the-art methods 
We compared `D-LIM` with:
- `LR` (linear regression)
- `ALM` (additive latent model)
- `LANTERN` (see [Tonner et al.](https://www.pnas.org/doi/10.1073/pnas.2114021119) )
- `MAVE-NN` (see [Tareen et al.](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02661-7))

## install mevv environment 
```
- go to https://github.com/jbkinney/mavenn 
- pip install mavenn

```
## install lantern 
- [LANTERN](https://github.com/usnistgov/lantern)
```
git clone https://github.com/usnistgov/lantern.git
cd lantern
python setup.py install
```

## Get comparison results

```
cd models_comp 
bash run_all.sh 
```
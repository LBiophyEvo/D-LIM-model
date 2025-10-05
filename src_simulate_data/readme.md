# Data Simulation

This module provides tools for simulating genotype-phenotype landscapes using various mathematical and biological models. The main class is `Simulated`, which generates synthetic datasets for testing and benchmarking genotype-fitness mapping algorithms.

## Landscape Types (`cor` argument)

The `cor` argument in the `Simulated` class specifies the type of landscape to simulate. Each type corresponds to a different mathematical or biological function:

- **"exp"**:  
  Generates a 2D exponential landscape centered at a specified point. Useful for smooth, unimodal fitness landscapes.

  $F(X,Y) = exp(-\frac{((X_0 - X)^2 + (Y_0 - Y)**2)}{t}),$

  where $X_0$ and $Y_0$ are the center of the exponential landscape. $t$ is a constant variable. 


- **"tgaus"**:  
  Produces a tilted Gaussian (tgaus) landscape, allowing simulation of anisotropic or rotated fitness peaks. See [Tenaillon et al.](https://www.annualreviews.org/content/journals/10.1146/annurev-ecolsys-120213-091846).



- **"cascade"**:  
  Simulates a genetic cascade using nested Hill functions and optimization, modeling complex gene interactions. See [Nghe et al.](https://www.nature.com/articles/s41467-018-03644-8)



- **"bio"**:  
  Implements a mechanistic biological model (Kemble et al. 2020) for realistic genotype-phenotype relationships.

 $F(X,Y) = \left ( w + \mu \varphi - \frac{\nu }{1/\eta - \varphi } \right )\left ( 1- \theta_X X - \theta_Y Y\right ),$

where $\varphi = \frac{1}{1/X + 1/Y + \eta }$ denotes for flux, $\eta$ is the
inverse of the maximal flux $\varphi$, $\theta_X$ and $\theta_Y$ represent the
cost of increasing cellular enzyme activity, $w$ describes the growth rate,
$\mu$ and $\nu$ are two variables related to downstream enzyme properties.

- **"add"**:  
  Simple additive landscape: fitness is the sum of two variables.

  $F(X,Y) = X + Y.$

- **"quad"**:  
  Quadratic landscape: fitness is the product of two variables.

  $F(X,Y) = X \times Y.$


- **"comp"**:  
  Composite landscape: combines additive and multiplicative effects.

  $F(X,Y) = X + Y - X \times Y.$


- **"saddle"**:  
  Saddle-shaped landscape: difference of squares, useful for simulating antagonistic effects.
  $F(X,Y) = X^2 - Y^2.$

- **"hat"**:  
  Hat-shaped landscape: uses a sine function for periodic or oscillatory fitness effects.
  $F(X,Y) = sin(X^2 + Y^2).$

## Usage Example

```python
from sim_data import Simulated

# Create a simulated dataset with an exponential landscape
dataset = Simulated(nb_var=20, cor="exp")

# Access data points
sample = dataset.data 
```

## Functions

- `hill_function`: Implements the Hill equation for gene regulation.
- `get_cascade`: Models a genetic cascade using nested Hill functions.
- `Simulated`: Main class for generating datasets with different landscape types.

See the source code for more details on each function and landscape type.
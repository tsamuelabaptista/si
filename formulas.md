# Formulas 

## Dense Layer

### Forward Propagation

$\frac{\partial E}{\partial X}=\left[\frac{\partial E}{\partial x_{1}}\quad\frac{\partial E}{\partial x_{2}}\quad\ ...\quad\frac{\partial E}{\partial x_{i}}\right]$

Using the chain rule:

${\frac{\partial E}{\partial x_{i}}}={\frac{\partial E}{\partial y_{1}}}{\frac{\partial y_{1}}{\partial x_{i}}}+\dots+{\frac{\partial E}{\partial y_{j}}}{\frac{\partial y_{j}}{\partial x_{i}}}$

$=\frac{\partial E}{\partial y_{1}}w_{i1}+\ldots+\frac{\partial E}{\partial y_{j}}w_{i j}$




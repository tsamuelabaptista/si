# Formulas 

## Dense Layer

### Forward Propagation

$$
\frac{\partial E}{\partial X}=\left[\frac{\partial E}{\partial x_{1}}\quad\frac{\partial E}{\partial x_{2}}\quad\ ...\quad\frac{\partial E}{\partial x_{i}}\right]
$$

Using the chain rule:

$$
{\frac{\partial E}{\partial x_{i}}}={\frac{\partial E}{\partial y_{1}}}{\frac{\partial y_{1}}{\partial x_{i}}}+\dots+{\frac{\partial E}{\partial y_{j}}}{\frac{\partial y_{j}}{\partial x_{i}}} 
$$

$$
=\frac{\partial E}{\partial y_{1}}w_{i1}+\ldots+\frac{\partial E}{\partial y_{j}}w_{i j}
$$

We can then write the whole matrix:

$$
\frac{\partial E}{\partial X}=\left[(\frac{\partial E}{\partial y_{1}}w_{11}+\dots+\frac{\partial E}{\partial y_{j}}w_{1j})\right.\ \dots\ \ \left.(\frac{\partial E}{\partial y_{1}}w_{i1}+\dots+\frac{\partial E}{\partial y_{j}}w_{i j})\right] 
$$

$$
=\left[{\frac{\partial E}{\partial y_{1}}}\quad ... \quad{\frac{\partial E}{\partial y_{j}}}\right] =\left[\frac{\partial E}{\partial y_{1}} ... \frac{\partial E}{\partial y_{j}}\right]
\begin{bmatrix}
w_{11} & \cdots & w_{1i} \\
\vdots & \ddots & \vdots \\
w_{j1} & \cdots & w_{ji}
\end{bmatrix}
$$

$$
={\frac{\partial E}{\partial Y}}W^{i}
$$

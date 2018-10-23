# doatools.py

**doatools.py** is the Python version of my [doa-tools](https://github.com/morriswmz/doa-tools) in MATLAB. It provides basic tools for theoretical research on direction-of-arrival (DOA) estimation, including basic array designs, various DOA estimators, plus tools to compute performance bounds. The MATLAB version served as a small toolbox for [my research](http://research.wmz.ninja/research.html) related to array signal processing. I made this Python version because I will no longer have access to MATLAB.

I made some notebooks that produce figures similar to those in my papers (may not be exactly the same due to the randomness of Monte Carlo simulations). You can browse them [here](examples/paper). These examples are not as complete as those in the MATLAB version.

You can also view the full documentation [here](https://morriswmz.github.io/doatools.py/).

## Features

* Several array design and difference coarray related functions.
* Commonly used DOA estimators including MVDR beamformer, MUSIC, root-MUSIC, ESPRIT.
* Maximum-likelihood estimators including AML, CML, and WSF.
* Sparsity-based DOA estimator.
* Functions to compute the [Cramér-Rao bounds](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound).
* Functions to compute the asymptotic covariance matrix of the estimation errors of MUSIC estimators (including difference coarray based).
* Functions to visualize the estimation results.

>**Note:** **doatools.py** is designed to facilitate my theoretical research on array signal processing. It is not designed for real-world applications. Nevertheless, the implementations of various DOA estimators in this repository provide good references on understanding these estimation algorithms.

## Differences from the MATLAB version

The Python version retains most of the functionalities of the MATLAB version, while adding some new ones. For instance, the Python version supports 2D (azimuth and elevation) DOA estimation and includes more ML-based estimators.

The Python version is also more modular, providing more flexibility over the MATLAB version. Many spectrum-based estimators (e.g., MUSIC) are now reusable, and work for both 1D and 2D sources.

The Python version is better documented.

## Requirements

**doatools.py** requires [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy) and [Matplotlib](https://github.com/matplotlib/matplotlib). It also requires [CVXPY](https://github.com/cvxgrp/cvxpy) to solve sparse recovery problems. To run the examples, you also need to install [tqdm](https://github.com/tqdm/tqdm).

## Examples

You can view the examples [here](examples/).

## License

The source code is released under the [MIT](LICENSE.md) license.

## Citation

If you find my code helpful. You are welcomed to cite my papers [here](http://research.wmz.ninja/research.html).

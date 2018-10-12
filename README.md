# doatools.py

**doatools.py** is the Python version of my [doa-tools](https://github.com/morriswmz/doa-tools) in MATLAB. It provides basic tools for direction-of-arrival (DOA) estimation related applications, including basic array designs, various DOA estimators, plus tools to compute performance bounds. The MATLAB version served as a small toolbox for [my research](http://research.wmz.ninja/research.html) related to array signal processing. I made this Python version because I will no longer have access to MATLAB.

I made some notebooks that produce figures similar to those in my papers (may not be exactly the same due to the randomness of Monte Carlo simulations). You can browse them [here](examples/paper). These examples are not as complete as those in the MATLAB version.

## Features

* Several array design and difference coarray related functions.
* Commonly used DOA estimators including MVDR beamformer, MUSIC, root-MUSIC, ESPRIT, etc.
* Sparsity-based DOA estimator.
* Functions to compute the [Cramér-Rao bounds](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound).
* Functions to compute the asymptotic covariance matrix of the estimation errors of MUSIC estimators (including difference coarray based).
* Functions to visualize the estimation results.

## Requirements

**doatools.py** requires **NumPy**, **SciPy** and **matplotlib**. It also requires **cvxpy** to solve sparse recovery problems.

## Examples

You can view the examples [here](examples/).

## License

The source code is released under the [MIT](LICENSE.md) license.

## Citation

If you find my code helpful. You are welcomed to cite my papers
[here](http://research.wmz.ninja/research.html).

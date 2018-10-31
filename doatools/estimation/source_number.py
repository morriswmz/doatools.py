import numpy as np
from math import log

def ld_stat(l, n_sources, n_snapshots):
    """Computes the sufficient statistic for source number detection in MDL/AIC.

    Args:
        l (~numpy.ndarray): A 1D vector of the eigenvalues of the covariance
            matrix in ascending order.
        n_sources (int): Number of sources.
        n_snapshots (int): Number of snapshots.
    
    References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    """
    if l.ndim != 1:
        raise ValueError('A 1D numpy vector expected.')
    n_sensors = l.size
    diff = n_sensors - n_sources
    l = l[:diff]
    return n_snapshots * diff * log(np.sum(l) / diff / (np.prod(l)**(1./diff)))

def aic(x, n_snapshots):
    """Detects source numbers using AIC.

    Args:
        x (~numpy.ndarray): A 1D vector of the eigenvalues of the covariance
            matrix in ascending order, or the covariance matrix itself.
        n_snapshots (int): Number of snapshots.

    Notes:
        AIC is inconsistent, and tends to asymptotically overestimate the
        number of sources. However, it tends to give a higher probability of a
        correct decision.
    
    References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    """
    if x.ndim > 1:
        x = np.linalg.eigvalsh(x)
    n_sensors = x.size
    ld = np.zeros((n_sensors, 1))
    for i in range(n_sensors):
        ld[i] = ld_stat(x, i, n_snapshots) + i * (2 * n_sensors - i)
    return np.argmin(ld)

def mdl(x, n_snapshots):
    """Detects source number using MDL.
    
    Args:
        x (~numpy.ndarray): A 1D vector of the eigenvalues of the covariance
            matrix in ascending order, or the covariance matrix itself.
        n_snapshots (int): Number of snapshots.
    
    Notes:
        MDL is consistent.
    
    References:
        [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    """
    if x.ndim > 1:
        x = np.linalg.eigvalsh(x)
    n_sensors = x.size
    ld = np.zeros((n_sensors, 1))
    for i in range(n_sensors):
        ld[i] = ld_stat(x, i, n_snapshots) + 0.5 * (i * (2 * n_sensors - i) + 1) * log(n_snapshots)
    return np.argmin(ld)

def sorte(x):
    r"""Detects source number using SORTE.

    Arg:
        x: (~numpy.ndarray): A 1D vector of the eigenvalues of the covariance
            matrix in ascending order, or the covariance matrix itself.
    
    Refereces:
        [1] Z. He, A. Cichocke, S. Xie, and K. Choi, "Detecting the number of
        clusters in n-way probabilistic clustering," IEEE Trans. Pattern
        Anal. Mach. Intell., vol. 32, pp. 2006-2021, Nov. 2010.
    """
    if x.ndim > 1:
        x = np.linalg.eigvalsh(x)
    # SORTE requires descending order.
    x = x[::-1]
    n = x.size
    if n < 4:
        raise ValueError('At least four eigenvalues required.')
    diffs = x[:-1] - x[1:]
    sigmas = [np.var(diffs[k:]) for k in range(n - 1)]
    s = np.zeros((n - 2,))
    for k in range(n - 2):
        if np.abs(sigmas[k]) < np.finfo(np.float_).eps:
            s[k] = np.inf
        else:
            s[k] = sigmas[k + 1] / sigmas[k]
    return np.argmin(s[:-1]) + 1

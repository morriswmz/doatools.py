import numpy as np
from ..model.sources import FarField1DSourcePlacement
from .core import ensure_n_resolvable_sources

def get_default_row_weights(m):
    """Gets the default row weights for the ESPRIT estimator.
    
    Args:
        m (int): Number of rows.

    Returns:
        A ndarray vector of weights.
    """
    w = np.zeros((m,))
    for i in range(m // 2):
        w[i] = i + 1
        w[m - i - 1] = i + 1
    if m % 2 == 1:
        w[m // 2] = (m + 1) / 2
    return np.sqrt(w)

class Esprit1D:
    """Creates an ESPRIT estimator for 1D uniform linear arrays.
    
    Args:
        wavelength (float): Wavelength of the carrier wave.

    References:
        [1] R. Roy and T. Kailath, "ESPRIT-estimation of signal parameters via
        rotational invariance techniques," IEEE Transactions on Acoustics,
        Speech and Signal Processing, vol. 37, no. 7, pp. 984–995,
        Jul. 1989.
        
        [2] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    """

    def __init__(self, wavelength):
        self._wavelength = wavelength

    def estimate(self, R, k, d0=None, displacement=1, formulation='ls',
                 row_weights='default', unit='rad'):
        r"""Estimate the direction-of-arrivals (DOAs) using ESPRIT.

        Args:
            R (~numpy.ndarray): Covariance matrix input. This covariance matrix
                must be obtained using a uniform linear array.
            
            k (int): Expected number of sources.

            d0 (float): Inter-element spacing of the uniform linear array used
                to obtain ``R``. If not specified, it will be set to one half
                of the ``wavelength`` used when creating this estimator.
                Default value is ``None``.
            
            displacement (int): The displacement between the two overlapping 
                subarrays measured in number of minimal inter-element spacings.
                Default value is 1.

                Increasing this value will lead to **smaller** unambiguous
                range and number of resolvable sources. Make sure your DOAs
                falls within the unambiguous range.
            
            formulation (str): Method used to estimate the rotation matrix.
                Either ``'tls'`` (Total Lease Squares) or ``'ls'``
                (Least Squares). Default value is ``'tls'``.
            
            row_weights (str or ~numpy.ndarray): Specifies the row weights
                with a vector or a string. Default value is ``'default'``, which
                generates the following weight vector:

                .. math::

                    \lbrack
                    1\ \sqrt{2}\ \sqrt{3}\ \cdots\ \sqrt{3}\ \sqrt{2}\ 1
                    \rbrack
                
                You can disable row weighting by passing in ``'none'``, or
                specify your own row weights with a 1D :class:`~numpy.ndarray`.
            
            unit (str): Unit of the estimates. Default value is ``'rad'``. See
                :class:`~doatools.model.sources.FarField1DSourcePlacement` for
                more details on valid units.

        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): ``True`` only if the rooting algorithm
              successfully finds ``k`` roots inside the unit circle. This flag
              does **not** guarantee that the estimated source locations are
              correct. The estimated source locations may be completely wrong!
              If resolved is False, ``estimates`` will be ``None``.
            * estimates (:class:`~doatools.model.sources.FarField1DSourcePlacement`):
              A :class:`~doatools.model.sources.FarField1DSourcePlacement`
              recording the estimated source locations. Will be ``None`` if
              resolved is ``False``.
        """
        m = R.shape[0]
        if displacement < 1:
            raise ValueError('Displacement must be a non-negative integer.')
        m_reduced = m - displacement
        ensure_n_resolvable_sources(k, m_reduced)
        if d0 is None:
            d0 = self._wavelength / 2.0
        if isinstance(row_weights, str):
            if row_weights == 'none':
                row_weights = None
            elif row_weights == 'default':
                row_weights = get_default_row_weights(m_reduced)
            else:
                raise ValueError("When specified using a string, row weights must be either 'none' or 'default'.")
        elif isinstance(row_weights, np.ndarray):
            if row_weights.ndim != 1 or row_weights.size != m_reduced:
                raise ValueError('Row weights must be a vector of length {0}.'.format(m_reduced))
        else:
            raise ValueError("Row weights must be 'default', 'none', or a compatible numpy vector.")
        # Extract the signal subspace.
        _, E = np.linalg.eigh(R)
        Es = E[:, -k:]
        # Separation
        Es1 = Es[:-displacement, :]
        Es2 = Es[displacement:, :]
        # Apply row weights.
        if row_weights is not None:
            Es1 *= row_weights[:, np.newaxis]
            Es2 *= row_weights[:, np.newaxis]
        # Estimate the rotation matrix.
        if formulation == 'tls':
            # Total least-squares
            C = np.hstack((Es1, Es2))
            C = C.conj().T @ C
            _, V = np.linalg.eigh(C)
            V = np.fliplr(V) # Now in descending order
            V12 = V[:k, k:]
            V22 = V[k:, k:]
            # Phi = -V12 V22^{-1}
            Phi = -np.linalg.solve(V22.T, V12.T).T
        elif formulation == 'ls':
            # Least-squares
            Es1_H = Es1.conj().T
            Phi = np.linalg.solve(Es1_H @ Es1, Es1_H @ Es2)
        else:
            raise ValueError("Formulation must be either 'ls' or 'tls'.")
        # Recover the DOAs.
        z = np.linalg.eigvals(Phi)
        return True, FarField1DSourcePlacement.from_z(z, self._wavelength, d0 * displacement, unit)

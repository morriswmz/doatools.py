from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from ..model.sources import FarField1DSourcePlacement
from ..utils.math import projm, vec
from .core import ensure_covariance_size, ensure_n_resolvable_sources

def f_nll_stouc(R, array, sources, wavelength, p, sigma):
    # log|S| + tr(S^{-1} R)
    # S = A P A^H + sigma * I
    A = array.steering_matrix(sources, wavelength)
    S = (A * p) @ A.conj().T + sigma * np.eye(array.size)
    sgn, logdet = np.linalg.slogdet(S)
    if sgn < 0 or logdet < 0:
        return np.Inf
    return logdet + np.trace(np.linalg.solve(S, R))

class CovarianceBasedMLEstimator(ABC):
    '''Encapsulates a covariance based maximum-likelihood problem.'''

    def __init__(self, array, wavelength):
        self._array = array
        self._wavelength = wavelength
        self._estimates = None

    def get_last_estimates(self):
        '''Retrieves the last estimates of source locations.'''
        if self._estimates is None:
            raise RuntimeError('No estimation has been performed yet.')
        # Returns a copy.
        return self._estimates[:]

    def get_max_resolvable_sources(self):
        '''Returns the maximum number of sources resolvable.
        
        This default implementation returns (array size - 1), which is suitable
        for most ML based estimators because the projection matrix of the
        steering matrix is not well-defined when the number of sources is
        greater than or equal to the number of sensors.
        '''
        return self._array.size - 1

    @abstractmethod
    def eval_nll(self, x, R, k):
        '''Evaluates the negative log-likelihood function for the given input.
        
        Implementation notice: `self._estimates` serves as a working instance
        for computing the steering matrix. It is reused and modified in-placed
        during the optimization process.

        Args:
            x: A vector of variables being optimized (e.g. DOAs, source powers,
                noise variance, etc.). The first k*d elements of x are always
                for source locations, where d=1 for 1D sources and d=2 for 2D
                sources. The remaining elements of x correspond to other unknown
                parameters.
            R: The sample covariance matrix.
            k: The number of sources.

        Returns:
            A real number, the value of the negative log-likelihood function for
            the given input.
        '''
        raise NotImplementedError()

    def prepare_opt_prob(self, sources0, R):
        '''Prepares the optimization problem.

        More specifically, this method
        1. constructs the objective function.
        2. creates the starting point x0.
        3. determines the bounds for the variable.

        Implementation notice: the first k*d elements of x0 should correspond to
        the source location estimates, where d=1 for 1D sources and d=2 for 2D
        sources. The default implementation assumes that x0 consists of only
        the source locations. This is not necessarily true for every ML-based
        optimization problems.

        Args:
            sources0: A SourcePlacement instance containing the initial guess of
                the source locations.
            R: The sample covariance matrix.

        Returns:
            f: The objective function.
            x0: A vector whose size is equal to the number of variables in the
                ML-based optimization problem.
            bounds: A list of 2-element tuples representing the bounds for the
                variables.
        '''
        k = sources0.size
        # Delegate the call to self.eval_nll
        f = lambda x : self.eval_nll(x, R, k)
        # Simply flatten the source location array:
        # x0 = [\theta_{11} \theta_{12} ... \theta_{1d} \theta{21} ...]
        # For instance, for far-field 2D sources
        # x0 = [az1 el1 az2 el2 ...]
        x0 = sources0.locations.copy().flatten()
        # Bounds for the source locations.
        # Must correspond to the ordering of the elements in x0.
        bounds = list(sources0.valid_ranges) * k
        return f, x0, bounds

    def update_estimates_from_x(self, x):
        '''Updates the current source location estimates from x.'''
        n = self._estimates.locations.size
        np.copyto(
            self._estimates.locations,
            x[:n].reshape(self._estimates.locations.shape)
        )

    def eval_steering_matrix_from_x(self, x):
        '''Evaluates the steering matrix from x.
        
        `self._estimates` is first updated from x and then used to evaluate
        the steering matrix.
        '''
        self.update_estimates_from_x(x)
        return self._array.steering_matrix(
            self._estimates, self._wavelength,
            perturbations='known'
        )
    
    def estimate(self, R, sources0, **kwargs):
        r'''Solves the ML problem for the given inputs.

        Note: in general, ML estimates are computationally expensive to obtain
        and sensitive to initialization. They are generally used in theoretical
        performance analyses.

        Args:
            R: The sample covariance matrix.
            sources0: A SourcePlacement instance representing the initial guess.
                Its type determines the source type and its size determines the
                number of sources.
                Note: because the log-likelihood function is highly non-convex,
                the initial guess of source locations will greatly affect the
                final estimates. It is recommended to use the output of another
                estimator (e.g. conventional beamformer) as the initial guess.
            **kwargs: Additional arguments for the solver.

        Returns:
            resolved (bool): A boolean indicating if the desired number of
                sources are found. This flag does not guarantee that the
                estimated source locations are correct. The estimated source
                locations may be completely wrong!
                This flag is always True for ML-based estimators.
            estimates (SourcePlacement): A SourcePlacement instance of the same
                type as the one used in the search grid, represeting the
                estimated DOAs. Will be `None` if resolved is False.
        '''
        ensure_n_resolvable_sources(sources0.size, self.get_max_resolvable_sources())
        ensure_covariance_size(R, self._array)
        # Make a copy of the initial guess as a working variable for the
        # optimization process.
        # This is reused and modified in-place during the optimization process
        # to avoid repeatedly creating new SourcePlacement instances.
        self._estimates = sources0[:]
        # Use scipy for numerical optimization.
        # Subclasses should override this implementation if there exists faster
        # optimization approaches.
        obj_func, x0, bounds = self.prepare_opt_prob(sources0, R)
        res = minimize(
            obj_func, x0,
            method='L-BFGS-B',
            bounds=bounds,
            **kwargs
        )
        self.update_estimates_from_x(res.x)
        return True, self.get_last_estimates()

class AMLEstimator(CovarianceBasedMLEstimator):
    r'''Asymptotic maximum-likelihood (AML) estimator.
    
    The AML estimator maximizes the following log-likelihood function:
        - logdet S - tr(S^{-1} R)
    where S = APA^H + sI, A is the steering matrix, P is the source covariance
    matrix, s is the noise variance, and R is the sample covariance matrix.
    Here the unknown parameters include the source locations, P, and s.
    We can obtain the MLE of P and s in terms of the source locations. The
    final optimization problem only involves the source locations as unknown
    variables.

    References:
    [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    '''

    def eval_nll(self, x, R, k):
        # See 8.6.1 of the following:
        # * H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
        m = self._array.size
        A = self.eval_steering_matrix_from_x(x)
        # Projection matrix of A
        PA = projm(A, True)
        # Null projection matrix of A
        PPA = np.eye(m) - PA
        # Computing NLL
        H = PA @ R @ PA + np.trace(PPA @ R) / (m - k) * PPA
        # Force Hermitian
        H += H.conj().T
        H *= 0.5
        sgn, nll_val = np.linalg.slogdet(H)
        if sgn <= 0:
            return np.Inf
        else:
            return nll_val

class CMLEstimator(CovarianceBasedMLEstimator):
    r'''Conditional maximum-likelihood (CML) estimator.
    
    Given the conditional observation model (the source signals are assumed to
    be deterministic unknown):
        y(t) = Ax(t) + n(t), t = 1,2,...,T,
    the CML estimator maximizes the following log-likelihood function:
        - TM\log s - s^{-1}\sum_{t=1}^T |y(t) - Ax(t)|^2
    where M is the number of sensors, T is the number of snapshots, A is the
    steering matrix, matrix, s is the noise variance.
    Here the unknown parameters include the source locations, x(t), and s.

    References:
    [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    '''

    def eval_nll(self, x, R, k):
        # See 8.5.2 of the following:
        # * H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
        # tr(P^\perp_A R)
        A = self.eval_steering_matrix_from_x(x)
        PPA = np.eye(self._array.size) - projm(A, True)
        return np.real(np.trace(PPA @ R))

class WSFEstimator(CovarianceBasedMLEstimator):
    r'''Weighted subspace fitting (WSF) estimator.

    WSF is based on the CML estimator. This class uses the asymptotically
    optimal weights.

    References:
    [1] M. Viberg and B. Ottersten, "Sensor array processing based on subspace
        fitting," IEEE Transactions on Signal Processing, vol. 39, no. 5,
        pp. 1110-1121, May 1991.
    [2] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    [3] P. Stoica and K. Sharman, "Maximum likelihood methods for direction-of-
        arrival estimation," IEEE Trans. Acoust., Speech, Signal Process.,
        vol. 38, pp. 1132-1143, July 1990.
    '''

    def prepare_m(self, sources0, R):
        '''Prepare the M matrix used in the optimization.'''
        k = sources0.size
        # Pre-calculate optimal weights
        v, E = np.linalg.eigh(R)
        # Signal subspace
        Es = E[:,-k:]
        vs = v[-k:]
        # Noise variance estimate
        sigma_est = np.sum(v[:-k]) / (self._array.size - k)
        # Asymptotically optimal weights
        vt = vs - sigma_est
        w = vt * vt / vs
        # M = Es diag(w) Es^H
        self._M = (Es * w) @ Es.conj().T

    def prepare_opt_prob(self, sources0, R):
        self.prepare_m(sources0, R)
        return super().prepare_opt_prob(sources0, R)

    def eval_nll(self, x, R, k):
        # See 8.5.3 of the following:
        # * H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
        A = self.eval_steering_matrix_from_x(x)
        PPA = np.eye(self._array.size) - projm(A, True)
        # tr(P^\perp_A M)
        return np.real(np.trace(PPA @ self._M))

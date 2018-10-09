import numpy as np
from ..model.arrays import UniformLinearArray
from ..model.coarray import WeightFunction1D
from ..utils.math import vec

class ACMTransformer1D:

    def __init__(self, design):
        '''
        Based on the array design, creates an callable object that transforms
        covariance matrices obtained from the physical array model into
        augmented covariance matrices under the difference coarray model.

        Args:
            design - A 1D grid-based array design. 
        '''
        self._design = design
        self._w = WeightFunction1D(design)
    
    def __call__(self, R, method='ss'):
        '''
        Transforms the input covariance matrix into the augmented covariance
        matrix under the difference coarray model.

        Args:
            R: Covariance matrix.
            method: 'ss' for spatial-smoothing based construction, and 'da'
                for direct-augmentation based construction. It should be noted
                that direct-augmentation does not guarantee the
                positive-definiteness of `Ra`, which may lead to unexpected
                results when using beamforming-based estimators.
          
        Returns:
            Ra: Augmented covariance matrix.
            vula: The virtual ULA associated with the augmented covariance
                matrix, to be used together with `Ra` for DOA estimation.

        References:
        [1] M. Wang and A. Nehorai, "Coarrays, MUSIC, and the Cramér-Rao Bound,"
            IEEE Transactions on Signal Processing, vol. 65, no. 4, pp. 933-946,
            Feb. 2017.
        [2] P. Pal and P. P. Vaidyanathan, "Nested arrays: A novel approach to
            array processing with enhanced degrees of freedom," IEEE
            Transactions on Signal Processing, vol. 58, no. 8, pp. 4167-4181,
            Aug. 2010.
        '''
        if R.shape[0] != self._design.size or R.shape[1] != self._design.size:
            raise ValueError('The dimension of the covariance matrix does not match the array size.')
        if method not in ['ss', 'da']:
            raise ValueError('Method can only be one of the following: ss, da.')
        mc = self._w.get_central_ula_size()
        mv = (mc + 1) // 2
        z = np.zeros((mc,), dtype=np.complex_)
        r = vec(R)
        for i in range(mc):
            diff = i - mv + 1
            z[i] = np.mean(r[self._w.indices_of(diff)])
        Ra = np.zeros((mv, mv), dtype=np.complex_)
        if method == 'ss':
            # Spatial smoothing
            for i in range(mv):
                Ra += np.outer(z[i:i+mv], z[i:i+mv].conj())
            Ra /= mv
        else:
            # Direct augmentation
            for i in range(mv):
                Ra[:,-(i+1)] = z[i:i+mv]
        return Ra, UniformLinearArray(mv, self._design.d0)

def get_narrowband_snapshots(array, sources, wavelength, source_signal,
                             noise_signal=None, n_snapshots=1,
                             return_covariance=False):
    '''Generates snapshots based on the narrowband snapshot model (see
    Chapter 8.1 of [1]).

    Let A be the steering matrix, s(t) be the source signal vector, and n(t) be
    the noise signal matrix. Then the snapshots received at the array is given
    by y(t) = A s(t) + n(t), t = 1, 2, ..., N, where N denotes the number of
    snapshots.

    Args:
        array: The array receiving the snapshots.
        sources: An instance of SourcePlacement.
        wavelength: Wavelength of the carrier wave.
        source_signal: Source signal generator.
        noise_signal: Noise signal generator. Default value is None, meaning no
            additive noise.
        n_snapshots: Number of snapshots. Default value is 1.
        return_covariance: If set to true, also returns the sample covariance
            matrix. Default value is False.

    Returns:
        Y: A matrix where each column represents a snapshot.
        R: The sample covariance matrix. R = (Y Y^H) / n_snapshots. 

    References:
    [1] H. L. Van Trees, Optimum array processing. New York: Wiley, 2002.
    '''
    A = array.steering_matrix(sources, wavelength)
    S = source_signal.emit(n_snapshots)
    Y = A @ S
    if noise_signal is not None:
        N = noise_signal.emit(n_snapshots)
        Y += N
    if return_covariance:
        R = (Y @ Y.conj().T) / n_snapshots
        return Y, R
    else:
        return Y

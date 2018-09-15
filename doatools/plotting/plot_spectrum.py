import numpy as np
import matplotlib.pyplot as plt

def _plot_spectrum_1d(sp, grid, estimates, ground_truth, use_log_scale, discrete):
    x = grid.axes[0]
    # Normalize
    max_y = sp.max()
    y = sp / max_y if max_y > 0 else sp
    if discrete:
        if use_log_scale:
            plt.yscale('log')
        container_sp = plt.stem(x, y, 'C0-', markerfmt=' ', basefmt=' ')
    else:
        if use_log_scale:
            container_sp = plt.semilogy(x, y, 'C0-')
        else:
            container_sp = plt.plot(x, y, 'C0-')
    # Set up x-axis
    x_presets = {
        'rad': ('\\theta/rad', -np.pi/2, np.pi/2),
        'deg': ('\\theta/deg', -90, 90),
        'sin': ('\\sin\\theta', -1, 1)
    }
    x_label, x_min, x_max = x_presets[grid.unit]
    plt.xlim((x_min, x_max))
    plt.xlabel(x_label)
    # Set up y-axis
    if not use_log_scale:
        plt.ylim((0, plt.ylim()[1]))
    plt.ylabel('Normalized spectrum')
    # Plot estimates
    if estimates is not None:
        if estimates.unit != grid.unit:
            raise ValueError('The unit of estimates does not match that of the search grid.')
        x_est = estimates.locations
        y_est = np.ones(x_est.shape)
        container_est = plt.stem(x_est, y_est, 'C3--', markerfmt='C3x',
                                 basefmt=' ', label='Estimates')
    else:
        container_est = None
    # Plot ground truth
    if ground_truth is not None:
        if ground_truth.unit != grid.unit:
            raise ValueError('The unit of ground truth does not match that of the search grid.')
        x_truth = ground_truth.locations
        y_truth = np.ones(x_truth.shape)
        container_truth = plt.stem(x_truth, y_truth, 'C1--', markerfmt='C1o',
                                   basefmt=' ', label='Ground truth')
    else:
        container_truth = None
    plt.legend()
    return container_sp, container_est, container_truth

def plot_spectrum(sp, grid, estimates=None, ground_truth=None,
                  use_log_scale=False, discrete=False):
    '''
    Provides a convenient way to plot the given spectrum.

    Args:
        sp: A numpy array representing the spectrum. Usually this is the output
            of a spectrum-based estimator.
        grid: The search grid used to generate the spectrum.
        estimates: A SourcePlacement instance containing the estimated source
            locations. Will be plotted if supplied. Default value is None.
        ground_truth: A SourcePlacement instance containing the true source
            locations. Will be plotted if supplied. Default value is None.
        use_log_scale: Sets whether the spectrum should be plotted in log scale.
            Default value is False.
        discrete: Sets whether the spectrum should be plotted with `stem`.
            Default value is False.
    
    Returns:
        *containers: A tuple of plot containers.
    '''
    if sp.shape != grid.shape:
        raise ValueError('The shape of the spectrum does not match the search grid.')
    if sp.ndim == 1:
        return _plot_spectrum_1d(sp, grid, estimates, ground_truth, use_log_scale, discrete)
    else:
        raise NotImplementedError()

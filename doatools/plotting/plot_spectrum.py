import numpy as np
import matplotlib.pyplot as plt

def _normalize_by_maximum(x):
    max_x = x.max()
    if max_x > 0:
        return x / max_x
    else:
        return x

def _preprocess_spectrum_input(sp, grid):
    '''
    Preprocesses the spectrum input and convert it into a list of tuples, where
    the first element in the tuple is the numpy array and the second element is
    the label. Also outputs the dimension of the spectrum.
    '''
    if isinstance(sp, np.ndarray):
        sp_list = [(sp, '')]
    elif isinstance(sp, list) or isinstance(sp, tuple):
        sp_list = [(s, '') for s in sp]
    elif isinstance(sp, dict):
        sp_list = [(v, k) for k, v in sp.items()]
    else:
        raise ValueError('Unsupported spectrum data input.')
    if len(sp_list) == 0:
        raise ValueError('Expecting at least on spectrum.')
    for t in sp_list:
        if t[0].shape != grid.shape:
            raise ValueError('The shape of the spectrum does not match the search grid.')
    return sp_list, sp_list[0][0].ndim

def _plot_spectrum_1d(sp_list, grid, estimates, ground_truth, use_log_scale, discrete):
    x = grid.axes[0]
    has_legend = False
    # 'C3' and 'C2' are reserved for estimates and ground truth
    color_estimates = 'C3'
    color_truth = 'C2'
    color_cycle = ['C0', 'C1', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    plot_containers = []
    # Plot every spectrum
    if discrete and use_log_scale:
        plt.yscale('log')
    for i, (spectrum, label) in enumerate(sp_list):
        color = color_cycle[i % len(color_cycle)]
        y = _normalize_by_maximum(spectrum)
        if len(label) > 0:
            has_legend = True
        if discrete:
            container_sp = plt.stem(x, y, '-', markerfmt=' ', basefmt=' ',
                                    label=label)
            plt.setp(container_sp, color=color)
            plot_containers.append(container_sp)
        else:
            if use_log_scale:
                plot_containers.append(plt.semilogy(x, y, '-', label=label, color=color))
            else:
                plot_containers.append(plt.plot(x, y, '-', label=label, color=color))
    # Set up x-axis
    x_presets = {
        'rad': ('DOA/rad', -np.pi/2, np.pi/2),
        'deg': ('DOA/deg', -90, 90),
        'sin': ('DOA/sin', -1, 1)
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
        container_est = plt.stem(x_est, y_est, '--', markerfmt='x',
                                 basefmt=' ', label='Estimates')
        plt.setp(container_est, color=color_estimates)
        plot_containers.append(container_est)
        has_legend = True
    # Plot ground truth
    if ground_truth is not None:
        if ground_truth.unit != grid.unit:
            raise ValueError('The unit of ground truth does not match that of the search grid.')
        x_truth = ground_truth.locations
        y_truth = np.ones(x_truth.shape)
        container_truth = plt.stem(x_truth, y_truth, '--', markerfmt='o',
                                   basefmt=' ', label='Ground truth')
        plt.setp(container_truth, color=color_truth)
        plot_containers.append(container_truth)
        has_legend = True
    # Show the legend.
    if has_legend:
        plt.legend()
    return plot_containers

def plot_spectrum(sp, grid, estimates=None, ground_truth=None,
                  use_log_scale=False, discrete=False):
    '''
    Provides a convenient way to plot the given spectrum.

    Args:
        sp: Can be one of the following:
            1. A numpy array representing the spectrum. Usually this is the
               output of a spectrum-based estimator.
            2. A list or tuple of numpy arrays of the same shape. This will
               draw multiple spectra in the same plot.
            3. A dictionary of numpy arrays of the same shape, where the keys
               represent the labels. This will draw multiple spectra in the same
               plot with labels.
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
        *containers: A list of plot containers.
    '''
    sp_list, sp_dim = _preprocess_spectrum_input(sp, grid)
    if sp_dim == 1:
        return _plot_spectrum_1d(sp_list, grid, estimates, ground_truth,
                                 use_log_scale, discrete)
    else:
        raise NotImplementedError()

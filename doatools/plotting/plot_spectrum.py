import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def _normalize_by_maximum(x):
    max_x = x.max()
    if max_x > 0:
        return x / max_x
    else:
        return x

def _build_spectrum_list(sp, grid):
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
            raise ValueError('The shape of the spectrum, {0}, does not match the search grid, {1}.'.format(t[0].shape, grid.shape))
    return sp_list

def plot_spectrum_1d(sp, grid, ax, estimates=None, ground_truth=None,
                     use_log_scale=False, discrete=False):
    '''Plots an 1D spectrum or multiple 1D spectra.

    '''
    # Preprocess the sp input
    sp_list = _build_spectrum_list(sp, grid)
    x = grid.axes[0]
    has_legend = False
    # 'C3' and 'C2' are reserved for estimates and ground truth
    color_estimates = 'C3'
    color_truth = 'C2'
    color_cycle = ['C0', 'C1', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    plot_containers = []
    # Plot every spectrum
    if discrete and use_log_scale:
        ax.set_yscale('log')
    for i, (spectrum, label) in enumerate(sp_list):
        color = color_cycle[i % len(color_cycle)]
        y = _normalize_by_maximum(spectrum)
        if len(label) > 0:
            has_legend = True
        if discrete:
            container_sp = ax.stem(x, y, '-', markerfmt=' ', basefmt=' ',
                                   label=label)
            plt.setp(container_sp, color=color)
            plot_containers.append(container_sp)
        else:
            if use_log_scale:
                plot_containers.append(plt.semilogy(x, y, '-', label=label, color=color))
            else:
                plot_containers.append(plt.plot(x, y, '-', label=label, color=color))
    # Set up x-axis
    ax.set_xlabel('{0}/{1}'.format(grid.axis_names[0], grid.units[0]))
    # Set up y-axis
    if not use_log_scale:
        ax.set_ylim((0, plt.ylim()[1]))
    ax.set_ylabel('Normalized spectrum')
    # Plot estimates
    if estimates is not None:
        if estimates.units != grid.units:
            raise ValueError('The unit of estimates does not match that of the search grid.')
        x_est = estimates.locations
        y_est = np.ones(x_est.shape)
        container_est = ax.stem(x_est, y_est, '--', markerfmt='x',
                                basefmt=' ', label='Estimates')
        plt.setp(container_est, color=color_estimates)
        plot_containers.append(container_est)
        has_legend = True
    # Plot ground truth
    if ground_truth is not None:
        if ground_truth.units != grid.units:
            raise ValueError('The unit of ground truth does not match that of the search grid.')
        x_truth = ground_truth.locations
        y_truth = np.ones(x_truth.shape)
        container_truth = ax.stem(x_truth, y_truth, '--', markerfmt='o',
                                  basefmt=' ', label='Ground truth')
        plt.setp(container_truth, color=color_truth)
        plot_containers.append(container_truth)
        has_legend = True
    # Show the legend.
    if has_legend:
        ax.legend()
    return plot_containers

def plot_spectrum_2d(sp, grid, ax, estimates=None, ground_truth=None,
                     use_log_scale=False, swap_axes=False, color_map='jet'):
    if sp.shape != grid.shape:
        raise ValueError('The shape of the spectrum, {0}, does not match the search grid, {1}.'.format(sp.shape, grid.shape))
    # Note that columns -> x, rows -> y by default
    if swap_axes:
        ind_x, ind_y = 0, 1
        sp = sp.T
    else:
        ind_x, ind_y = 1, 0
    axes = grid.axes
    axis_names = grid.axis_names
    units = grid.units
    has_legend = False
    x, y = axes[ind_x], axes[ind_y]
    z = _normalize_by_maximum(sp)
    x_label = '{0}/{1}'.format(axis_names[ind_x], units[ind_x])
    y_label = '{0}/{1}'.format(axis_names[ind_y], units[ind_y])
    plot_args = {
        'extent': (x.min(), x.max(), y.min(), y.max()),
        'origin': 'lower',
        'cmap': color_map,
        'aspect': 'auto'
    }
    containers = []
    if use_log_scale:
        plot_args['norm'] = LogNorm()
    containers.append(ax.imshow(sp, **plot_args))
    plt.colorbar(containers[0], ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if estimates is not None:
        if estimates.units != grid.units:
            raise ValueError('The unit of estimates does not match that of the search grid.')
        containers.append(ax.scatter(
            estimates.locations[:, ind_x],
            estimates.locations[:, ind_y],
            marker='o',
            edgecolors='k',
            facecolors='none',
            label='Estimates'))
        has_legend = True
    if ground_truth is not None:
        if ground_truth.units != grid.units:
            raise ValueError('The unit of ground truth does not match that of the search grid.')
        containers.append(ax.scatter(
            ground_truth.locations[:, ind_x],
            ground_truth.locations[:, ind_y],
            marker='+',
            c='k',
            label='Ground truth'))
        has_legend = True
    if has_legend:
        ax.legend()
    return containers

def plot_spectrum(sp, grid, ax=None, estimates=None, ground_truth=None,
                  use_log_scale=False, **kwargs):
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
        ax: The axes object containing the plot.
        containers: A list of plot containers.
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        new_plot = True
    else:
        new_plot = False
    if grid.ndim == 1:
        ret = plot_spectrum_1d(sp, grid, ax, estimates, ground_truth,
                               use_log_scale, **kwargs)
    elif grid.ndim == 2:
        ret = plot_spectrum_2d(sp, grid, ax, estimates, ground_truth,
                               use_log_scale, **kwargs)
    else:
        raise NotImplementedError()
    if new_plot:
        plt.show()
    return ax, ret

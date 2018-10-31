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
    """
    Preprocesses the spectrum input and convert it into a list of tuples, where
    the first element in the tuple is the numpy array and the second element is
    the label. Also outputs the dimension of the spectrum.
    """
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
    """Plots a 1D spectrum or multiple 1D spectra.

    Args:
        sp: Can be one of the following:

            1. An :class:`~numpy.ndarray` representing the spectrum. Usually
               this is the output of a spectrum-based estimator. This function
               will draw a single spectrum.
            2. A :class:`~list` or :class:`~tuple` of :class:`~numpy.ndarray` of
               the same shape. This function will draw multiple spectra in the
               same plot without labels.
            3. A dictionary that maps labels to numpy arrays of the same shape.
               This function will draw multiple spectra in the same plot with
               labels.
        
        grid (~doatools.estimation.grid.SearchGrid): The search grid used to
            generate the spectrum/spectra. Its shape must match that of the
            spectrum/spectra.
        ax (~matplotlib.axes.Axes): The matplotlib axes that will be used for
            plotting.
        estimates (~doatools.model.sources.SourcePlacement): Estimated source
            locations. Will be plotted if supplied. Default value is ``None``.
        ground_truth (~doatools.model.sources.SourcePlacement): True source
            locations. Will be plotted if supplied. Default value is ``None``.
        use_log_scale (bool): Sets whether the spectrum should be plotted in
            logarithmic scale. Default value is ``False``.
        discrete (bool): Sets whether the spectrum should be visualized using
            stem plots instead of line plots. Default value is ``False``.

    Returns:
        list: A list of plot containers with the following structure:
        ``[sp1, sp2, ..., spN, est, truth]``, where ``sp1``, ``sp2``, ...,
        ``spN`` are the plot containers of the spectra, ``est`` is the plot
        container of the estimates, and ``truth`` is the plot container of
        the ground truth.
    """
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
    # Set up x-axis
    ax.set_xlabel('{0}/{1}'.format(grid.axis_names[0], grid.units[0]))
    ax.margins(x=0)
    # Set up y-axis
    if not use_log_scale:
        ax.set_ylim((0, plt.ylim()[1]))
    ax.set_ylabel('Normalized spectrum')
    # Show the legend.
    if has_legend:
        ax.legend()
    return plot_containers

def plot_spectrum_2d(sp, grid, ax, estimates=None, ground_truth=None,
                     use_log_scale=False, swap_axes=False, color_map='jet'):
    """Plots a 2D spectrum.

    Args:
        sp (~numpy.ndarray): A 2D ndarray representing the spectrum.
        grid (~doatools.estimation.grid.SearchGrid): The search grid used to
            generate the spectrum. Its shape must match the shape of ``sp``.
        ax (~matplotlib.axes.Axes): The matplotlib axes that will be used for
            plotting.
        estimates (~doatools.model.sources.SourcePlacement): Estimated source
            locations. Will be plotted if supplied. Default value is ``None``.
        ground_truth (~doatools.model.sources.SourcePlacement): True source
            locations. Will be plotted if supplied. Default value is ``None``.
        use_log_scale (bool): Sets whether the spectrum should be plotted in
            logarithmic scale. Default value is ``False``.
        swap_axes (bool): Set to ``True`` to swap the x and y axis when
            plotting. Default value is ``False``.
        color_map: Specifies the color map. Default value is ``'jet'``.

    Returns:
        list: A list of plot containers with the following structure:
        ``[sp, est, truth]``, where ``sp1`` is the plot containers of the
        spectrum, ``est`` is the plot container of the estimates, and ``truth``
        is the plot container of the ground truth.
    """
    if sp.shape != grid.shape:
        raise ValueError('The shape of the spectrum, {0}, does not match the search grid, {1}.'.format(sp.shape, grid.shape))
    # Note that columns -> x, rows -> y by default
    if swap_axes:
        ind_x, ind_y = 1, 0
    else:
        ind_x, ind_y = 0, 1
        sp = sp.T
    axes = grid.axes
    axis_names = grid.axis_names
    units = grid.units
    has_legend = False
    x, y = axes[ind_x], axes[ind_y]
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

def plot_spectrum(sp, grid, ax=None, figsize=None, estimates=None,
                  ground_truth=None, use_log_scale=False, **kwargs):
    """Plots the given spectrum/spectra.
    
    Provides a convenient way to plot the given spectrum/spectra. Automatically
    selects the plot function based on input grid's number of dimensions.

    Args:
        sp: Compatible spectrum (or spectra collection) input.
        grid (~doatools.estimation.grid.SearchGrid): The search grid used to
            generate the spectrum/spectra. Its shape must match that of the
            spectrum/spectra supplied in ``sp``.
        ax (~matplotlib.axes.Axes): The matplotlib axes used for plotting. If
            not specified, a new figure will be created and shown. Default value
            is ``None``.
        figsize (tuple): If ``ax`` is ``None``, specifies the new figure's size.
        estimates (~doatools.model.sources.SourcePlacement): Estimated source
            locations. Will be plotted if supplied. Default value is ``None``.
        ground_truth (~doatools.model.sources.SourcePlacement): True source
            locations. Will be plotted if supplied. Default value is ``None``.
        use_log_scale (bool): Sets whether the spectrum should be plotted in
            logarithmic scale. Default value is ``False``.
        **kwargs: Other compatible options depending on the number of dimensions
            of the input grid. See :meth:`plot_spectrum_1d` and
            :meth:`plot_spectrum_2d` for more details.
    
    Returns:
        tuple: A tuple consists of the following elements:

        * ax (:class:`~matplotlib.axes.Axes`): The axes object containing the
          plot.
        * containers (:class:`~list`): A list of plot containers. See
          :meth:`plot_spectrum_1d` and :meth:`plot_spectrum_2d` for more
          details.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
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

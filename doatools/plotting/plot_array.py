import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from ..model.coarray import compute_unique_location_differences

def _auto_scatter(ax, x, *args, **kwargs):
    """Scatter plots the input points (1D, 2D, or 3D).

    This function automatically calls `scatter` with the correct signature based
    on the number of columns of the input.
    """
    if x.shape[1] == 1:
        ax.scatter(x[:,0], np.zeros((x.shape[0])), *args, **kwargs)
    elif x.shape[1] == 2:
        ax.scatter(x[:,0], x[:,1], *args, **kwargs)
    elif x.shape[1] == 3:
        ax.scatter(x[:,0], x[:,1], x[:,2], *args, **kwargs)
    else:
        raise ValueError('To many columns.')

def _fix_3d_aspect(ax):
    # `set_aspect` is broken for 3d projections:
    # https://github.com/matplotlib/matplotlib/issues/1077
    # We make it slightly better by adjusting the data limits.
    limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    ranges = [abs(l[1] - l[0]) for l in limits]
    max_range = max(ranges)
    # Need to check for pathological cases.
    if max_range == 0:
        return
    # Force similar ranges and re-scales the view.
    for i in range(3):
        if ranges[i] == 0:
            limits[i] = [-max_range / 2.0, max_range / 2.0]
        else:
            limits[i] = [limits[i][0] * max_range / ranges[i], limits[i][1] * max_range / ranges[i]]
    # This method is not documented but judging from its source it should do
    # the job for us.
    ax.auto_scale_xyz(limits[0], limits[1], limits[2])

def _plot_array_impl(array, ax=None, coarray=False, show_location_errors=False):
    """Internal implementation for plotting arrays."""
    if not array.has_perturbation('location_errors') and show_location_errors:
        warnings.warn(
            'The input array does not have location errors.'
            'Visualization of location errors is disabled.'
        )
        show_location_errors = False
    # Create a new axes if necessary.
    if show_location_errors:
        plt_dim = array.actual_ndim
    else:
        plt_dim = array.ndim
    if ax is None:
        new_plot = True
        fig = plt.figure()
        if plt_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    else:
        new_plot = False
    # Plot the nominal array.
    element_locations = array.element_locations
    if coarray:
        element_locations = compute_unique_location_differences(element_locations)        
    _auto_scatter(ax, element_locations, marker='o', label='Nominal locations')
    # Plot the perturbed array.
    if show_location_errors:
        element_locations = array.actual_element_locations
        if coarray:
            element_locations = compute_unique_location_differences(element_locations)
        _auto_scatter(ax, element_locations, marker='x', label='Actual locations')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if plt_dim < 3:
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True)
        ax.set_axisbelow(True) # Move grid lines behind.
    else:
        ax.set_zlabel('z')
        _fix_3d_aspect(ax)
    ax.legend()
    if new_plot:
        plt.show()
    return ax

def plot_array(array, ax=None, show_location_errors=False):
    """Visualizes the input array.

    Args:
        array (~doatools.model.arrays.ArrayDesign): A sensor array.
        ax (~matplotlib.axes.Axes): Matplotlib axes used for the plot. If not
            specified, a new figure will be created. Default value is ``None``.
        show_location_errors (bool): If set to ``True``, will visualized the
            perturbed array if the input array has location errors.
    
    Returns:
        The axes object containing the plot.
    """
    return _plot_array_impl(array, ax, False, show_location_errors)

def plot_coarray(array, ax=None, show_location_errors=False):
    """Visualizes the difference coarray of the input array.

    Args:
        array (~doatools.model.arrays.ArrayDesign): A sensor array.
        ax (~matplotlib.axes.Axes): Matplotlib axes used for the plot. If not
            specified, a new figure will be created. Default value is ``None``.
        show_location_errors (bool): If set to ``True``, will visualized the
            perturbed array if the input array has location errors.
    
    Returns:
        The axes object containing the plot.
    """
    return _plot_array_impl(array, ax, True, show_location_errors)

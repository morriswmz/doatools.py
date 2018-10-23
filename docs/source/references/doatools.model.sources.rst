Modeling source placements
==========================

In common direction-of-arrival estimation problems, the sources are assumed to
be far-field. We provide two basic types of far-field source placements:

* :class:`~doatools.model.sources.FarField1DSourcePlacement` - Each sources
  location is represented by a broadside angle and assumed to be within the
  xy-plane. The broadside angles are measured with respect to linear arrays
  along the x-axis.
* :class:`~doatools.model.sources.FarField2DSourcePlacement` - Each sources
  location is represented by a pair of azimuth and elevation angles. The azimuth
  angles are measured counter-clockwise starting from the x-axis. The elevation
  angles are measured with respect to the xy-plane.

.. code-block:: python

    import numpy as np
    import doatools.model as model
    # Uniformly place 5 far-field sources within (-pi/3, pi/3)
    # Default unit is 'rad'.
    sources_1d = model.FarField1DSourcePlacement(
        np.linspace(-np.pi/3, np.pi/3, 5)
    )
    # Place 3 2D sources by specifying azimuth-elevation pairs.
    sources_2d = model.FarField2DSourcePlacement(
        [[-10, 30], [30, 50], [100, 45]],
        unit='deg'
    )

API references
~~~~~~~~~~~~~~

.. automodule:: doatools.model.sources
    :members:
    :special-members: __getitem__

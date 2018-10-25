Modeling arrays
===============

In array signal processing, sensor arrays consists a group of sensors (acoustic,
electromagnetic, etc.) arranged in special geometry patterns. They find wide
applications in radar, sonar, audio and speech processing, geophysics, and
communications. In ``doatools``, we focus on their application in
direction-of-arrival (DOA) estimation.

.. code-block:: python

    import doatools.model as model
    # Create a 10-element ULA with inter-element spacing 0.5.
    ula = model.UniformLinearArray(10, 0.5)

API references
~~~~~~~~~~~~~~

.. automodule:: doatools.model.arrays
    :members:

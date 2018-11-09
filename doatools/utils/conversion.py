import numpy as np

def ae2broad(doas):
    """Converts azimuth-elevation pairs to broadside angles.

    The array is assumed to be placed along the x-axis.

    When the elevation angle is zero, the broadside angle is measured with
    respect to the y-axis and ranges from -pi/2 (left side of the y-axis) and
    pi/2 (right side of the y-axis. When the elevation angle is not zero, we
    need to consider the plane that intersects with the xy-plane along the
    x-axis.
    
    The azimuth angle is measured counter-clockwise starting from the positive
    x-axis.

    The elevation angle is measured with respect to the xy-plane and ranges
    from -pi/2 (below the xy-plane, z-neg) and pi/2 (above the xy-plane, z-pos).
    
          y | broadside angle
            | v /
            |  /
            | /
            |/  < azimuth angle 
    --------+-------->x
    """
    return np.arcsin(np.cos(doas[:,0]) * np.cos(doas[:,1])).reshape((-1, 1))

ANGULAR_COV_MAT = {
    'rad': {
        'deg': lambda x: np.rad2deg(x),
        'sin': lambda x: np.sin(x)
    },
    'deg': {
        'rad': lambda x: np.deg2rad(x),
        'sin': lambda x: np.sin(np.deg2rad(x))
    },
    'sin': {
        'rad': lambda x: np.arcsin(x),
        'deg': lambda x: np.rad2deg(np.arcsin(x))
    }
}

def convert_angles(x, from_unit, to_unit):
    """Converts input angular values to a new unit.
    
    If `from_unit` and `to_unit` are the same, a copied will be made.

    Args:
        x: An ndarray of angular values. Must be within the following ranges to
            ensure invertible conversion:
                'rad': (-pi, pi)
                'deg': (-180, 180)
                'sin': (-1, 1)
        from_unit: The original unit.
        to_unit: The target unit.
    """
    if from_unit == to_unit:
        return x.copy()
    return ANGULAR_COV_MAT[from_unit][to_unit](x)
    
def cart2spherical(coords):
    """Converts Cartesian coordinates to spherical coordinates.
    
    Args:
        x (~numpy.ndarray): A matrix where each row represents a single
            Cartesian coordinate.
    
    Returns:
        ~numpy.ndarray: A matrix where each row is a triplet of the range, the
        azimuth angle and the elevation angle. The azimuth angle is measured
        counter-clockwise from the positive x-axis, and the elevation angle is
        measured from the xy-plane.
    """
    s = np.zeros((coords.shape[0], 3))
    if coords.shape[1] == 1:
        # x coordinates only.
        s[:, 0] = np.abs(coords[:, 0])
        s[:, 1] = np.arctan2(0, coords[:, 0])
    elif coords.shape[1] <= 3:
        xy2 = coords[:, 0]**2 + coords[:, 1]**2
        s[:, 1] = np.arctan2(coords[:, 1], coords[:, 0])
        if coords.shape[1] == 2:
            # x, y coordinates only
            s[:, 0] = np.sqrt(xy2)
        else:
            # x, y, z coordinates are all present
            s[:, 0] = np.sqrt(xy2 + coords[:, 2]**2)
            s[:, 2] = np.arctan2(coords[:, 2], np.sqrt(xy2))
    return s


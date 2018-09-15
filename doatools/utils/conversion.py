import numpy as np

def ae2broad(doas):
    '''
    Converts azimuth-elevation pairs to broadside angles.
    The array is assumed to be placed along the x-axis.
    The broadside angle is measured with respect to the y-axis and ranges
    from -pi/2 (left side of the y-axis) and pi/2 (right side of the y-axis.
    The azimuth angle is measured counter-clockwise starting from the x-axis.
    The elevation angle is measured with respect to the xy-plane and ranges
    from -pi/2 (below the xy-plane, z-neg) and pi/2 (above the xy-plane, z-pos).
    
          y | broadside angle
            | v /
            |  /
            | /
            |/  < azimuth angle 
    --------+-------->x
    '''
    return np.arcsin(np.cos(doas[:,0]) * np.cos(doas[:,1])).reshape((-1, 1))

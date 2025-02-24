import numpy as np

import numpy as np

def regplane_to_polar2(x_coords, y_coords, regplane_size=10000, fix_rotation=True):
    """
    Convert regression plane coordinates to polar coordinates.
    
    The Euclidean space is used for determining the rotation,
    i.e., the rotation direction is counterclockwise and rotated by -90 degrees.
    
    Parameters:
        x_coords (array-like): x coordinates from the regression plane.
        y_coords (array-like): y coordinates from the regression plane.
        regplane_size (int, optional): Size of the square regression plane in pixels (default: 10000).
        fix_rotation (bool, optional): Whether to apply regression plane conception rotation and direction (default: True).
    
    Returns:
        theta (numpy.ndarray): Rotation in degrees.
        radius (numpy.ndarray): Radius in pixels.
    """
    x_coords = np.asarray(x_coords, dtype=np.float64).flatten()
    y_coords = np.asarray(y_coords, dtype=np.float64).flatten()
    
    # Center the coordinates around (regplane_size / 2, regplane_size / 2)
    p = np.column_stack((x_coords, y_coords)) - (regplane_size / 2)
    
    # Reference x-axis vector
    X = np.array([regplane_size, 0], dtype=np.float64)
    
    # Compute theta using acos (equivalent to MATLAB's acosd)
    dot_products = np.sum(X * p, axis=1)
    norms_X = np.linalg.norm(X)
    norms_p = np.linalg.norm(p, axis=1)
    
    theta = np.degrees(np.arccos(dot_products / (norms_X * norms_p)))
    
    # Adjust for points below the x-axis (y < 0)
    theta[p[:, 1] < 0] = 360 - theta[p[:, 1] < 0]
    
    # Compute radius
    radius = np.sqrt(np.sum(p**2, axis=1))

    if fix_rotation:
        theta = theta + 90
        theta = -theta
        theta = theta % 360

    return theta, radius

def regplane_to_polar(x, y, center=(5000, 5000)):
    """Convert Cartesian coordinates to polar (theta in degrees), considering a center offset and rotation."""
    x_shifted = x - center[0]
    y_shifted = y - center[1]

    # Convert to polar coordinates and apply the shift (rotation)
    theta = (np.degrees(np.arctan2(y_shifted, x_shifted)))
    r = np.sqrt(x_shifted ** 2 + y_shifted ** 2)

    return theta, r

def regression_to_class(coords, center=(5000, 5000), shift=0):
    """
    Convert a list of regression plane coordinates to categorical classes (1-41),
    with reversed class ordering. The bottom 15-15 degrees are classified as 'interphase'.

    Args:
        coords (list of tuples): List of (x, y) coordinates.
        center (tuple): The center point for polar conversion.
        shift (float): Shift applied to angle calculation.

    Returns:
        list: List of class categories.
    """
    split_deg = (360 - 30) / 40  # Divide the remaining 330 degrees into 40 bins
    class_cats = []

    for coord in coords:
        theta, r = regplane_to_polar2(coord[0], coord[1])  # Apply 90° rotation
        
        if 15 <= theta <= 345:  # Mitotic classes
            class_cat = int((theta - 15) / split_deg+1)  # Reverse the class order
        else:  # Interphase class
            class_cat = 0  # Interphase

        class_cats.append(class_cat)

    return class_cats


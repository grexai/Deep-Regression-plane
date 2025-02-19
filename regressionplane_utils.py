import numpy as np


def regplane_to_polar(x, y, center=(5000, 5000), shift=0):
    """Convert Cartesian coordinates to polar (theta in degrees), considering a center offset and rotation."""
    x_shifted = x - center[0]
    y_shifted = y - center[1]

    # Convert to polar coordinates and apply the shift (rotation)
    theta = (np.degrees(np.arctan2(y_shifted, x_shifted)) + shift) % 360
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
        theta, r = regplane_to_polar(coord[0], coord[1], center, shift=shift)  # Apply 90° rotation

        if 15 <= theta <= 345:  # Mitotic classes
            class_cat = 40 - int((theta - 15) / split_deg)  # Reverse the class order
        else:  # Interphase class
            class_cat = 0  # Interphase

        class_cats.append(class_cat)

    return class_cats


import numpy as np

def regplane_to_polar(x, y, center=(5000, 5000)):
    """Convert Cartesian coordinates to polar (theta in degrees), considering a center offset."""
    x_shifted = x - center[0]
    y_shifted = y - center[1]
    theta = (np.degrees(np.arctan2(y_shifted, x_shifted))) % 360  # Rotate CCW by 90°
    r = np.sqrt(x_shifted ** 2 + y_shifted ** 2)
    return theta, r

def regression_to_class(coord, center=(5000, 5000)):
    """
    Convert regression plane coordinates to a categorical class (1-41), with reversed class ordering.
    The bottom 15-15 degrees are classified as 'interphase'.
    """
    split_deg = (360 - 30) / 40  # Divide the remaining 330 degrees into 40 bins
    theta, r = regplane_to_polar(coord[0], coord[1], center)
    print(f"Theta: {theta}, Radius: {r}")

    if 15 <= theta <= 345:  # Mitotic classes
        class_cat = 40 - int((theta - 15) / split_deg)  # Reverse the class order
    else:  # Interphase class
        class_cat = 0  # Interphase

    return class_cat

import numpy as np

def radius_valley_vaneylen18(period :float) -> float:
    """
    Returns the location of the radius valley for a given orbital period.
    Following the definition of radius valley by Van Eylen et al. (2018).

    Parameters
    ----------
    period  : float, orbital period in days.

    Returns
    -------
    radius  : float, planet radius in Earth radii at which the radius valley occurs.
    """
    return 10**(-0.09 * np.log10(period) + 0.37)
    

def neptune_desert_mazeh16(period: float) -> float:
    """
    Returns the radii corresponding to the lower and upper
    edges of the Neptune desert, following Mazeh et al. (2016).

    Parameters
    ----------
    period  : float, orbital period

    Returns
    (lower,upper)   : [float,2], Radii of the lower and upper
                    edges of the Neptune desert.
    """
    upper_radius = 10**(-0.33 * np.log10(period) + 1.17)
    lower_radius = 10**( 0.68 * np.log10(period))
    return np.array([lower_radius, upper_radius])
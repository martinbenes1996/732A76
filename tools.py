
from math import radians, degrees, sin, cos, asin, acos, sqrt
import numpy as np
import pandas as pd

import eunuts

#eunuts.init_cache()
def nuts(city, country):
    df = pd.DataFrame(data = {'city': city, 'country': country})
    return eunuts.nuts(df)


def great_circle(coords1, coords2):
    """Great circle distance / Haversine formula.
    
    Args:
        coords1,coords2 (tuple): Latitude and longitude in degrees, floats.
    Returns:
        (float) Distance between the two points in km.
    """
    lon1, lat1, lon2, lat2 = map(radians, (*coords1, *coords2))
    if lon1 == lon2 and lat1 == lat2: return 1
    return 6371 * (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))

def rbf(d, h = 100):
    """Radial base function / Guassian kernel.
    
    Args:
        d (float): Distance.
        h (float, optional): Kernel width, by default 100.
    Returns:
        (float) Kernel score.
    """
    return np.exp(-d**2 / 2 / h**2)

if __name__ == "__main__":
    nuts3 = nuts("Brno","CZ")
    print(nuts3)
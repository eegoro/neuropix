import numpy as np

def get_csd(data):
    csd_array = -np.gradient(np.gradient(data, axis=0), axis=0)

    return csd_array
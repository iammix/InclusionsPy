import numpy as np
from scipy.linalg import norm
from typing import List


def getRotationMatrix(el_coords) -> np.ndarray:
    if el_coords.shape[1] != 2:
        raise NotImplementedError('Rotation matrix only implemented for 2D')

    l0 = norm(el_coords[1] - el_coords[0])

    sinalpha = (el_coords[1, 1] - el_coords[0, 1]) / l0
    cosalpha = (el_coords[1, 0] - el_coords[0, 0]) / l0

    return np.array([[cosalpha, sinalpha], [-sinalpha, cosalpha]])



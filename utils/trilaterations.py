import math
import numpy as np
from scipy.optimize import least_squares

def trilaterate_3d(p1, p2, p3, r1, r2, r3):
    '''
    trilaterate: Perform 3D Trilateration and returns the x, y, z coords
        Parameters:
            - p1, p2, p3: 3D Vector of Reference Point
            - r1, r2, r3: Distance from unknown point to reference point
    '''
    # Convert the input points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate the differences between the points
    delta_p2_p1 = p2 - p1
    delta_p3_p1 = p3 - p1

    # Calculate the squared distances
    d1_sq = np.dot(delta_p2_p1, delta_p2_p1)
    d2_sq = np.dot(delta_p3_p1, delta_p3_p1)

    # Calculate the scaling factors
    d = np.sqrt(d1_sq)
    ex = delta_p2_p1 / d

    i = np.dot(delta_p3_p1, ex)
    ey = (delta_p3_p1 - i * ex) / np.sqrt(d2_sq - i**2)

    # Calculate the coordinates of the intersection point
    j = np.dot(delta_p2_p1, ey)
    x = (r1**2 - r2**2 + d1_sq) / (2 * d)
    y = (r1**2 - r3**2 + i**2 + j**2) / (2 * j) - (i / j) * x

    # Calculate the z-coordinate assuming all points lie on the same plane
    z = np.sqrt(r1**2 - x**2 - y**2)

    # Calculate the final coordinates
    result = p1 + x * ex + y * ey + z * np.cross(ex, ey)

    return result

def trilaterate_2d(x1, y1, d1, x2, y2, d2, x3, y3, d3):
    '''
    trilaterate_2d: Perform 2D Trilateration and returns the x, y coord of the position
        Parameters:
            - x1, x2, x3: X Coords of the reference points
            - y1, y2, y3: Y Coords of the reference points
            - d1, d2, d3: Distance from unknown point to the reference point
    '''

    # Define the position coords to list of tuples
    pos = [(x1, y1), (x2, y2), (x3, y3)]

    # Define the distances into a collection
    dis = [d1, d2, d3]

    # Count the residuals of each positions in corresponding to the distance of the unknown point 
    def residuals(x, distances, positions):
        residuals = []
        for i in range(len(dis)):
            residuals.append(np.linalg.norm(x - positions[i]) - distances[i])
        return residuals

    # Count the initial guess
    init_guess = np.mean(pos, axis=0)
    
    # Using least squares, optimize the guess using distances and positions as parameters
    res = least_squares(residuals, init_guess, args=(dis, pos))
    return res

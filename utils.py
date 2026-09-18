"""
utils.py — small helper functions shared across the pipeline.
"""

import numpy as np


def order_points(pts):
    """
    Takes 4 (x, y) points in ANY order and returns them ordered as:
        [top-left, top-right, bottom-right, bottom-left]

    Why this is needed:
    cv2.findContours gives us 4 corner points, but in no particular order.
    To compute a perspective transform, OpenCV needs to know exactly which
    point is which corner. Get this wrong and the image comes out mirrored,
    rotated, or warped incorrectly.

    The trick:
    - top-left point has the SMALLEST (x + y) sum
    - bottom-right point has the LARGEST (x + y) sum
    - top-right point has the SMALLEST (y - x) difference
    - bottom-left point has the LARGEST (y - x) difference
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def distance(pt1, pt2):
    """Euclidean distance between two (x, y) points."""
    return np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))

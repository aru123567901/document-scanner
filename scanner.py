"""
scanner.py — Step 2 & 3 of the pipeline:
    1. Find the document's 4 corners in a photo
    2. Warp the perspective so the document becomes a flat rectangle
"""

import cv2
import numpy as np
from utils import order_points, distance


def find_document_contour(image):
    """
    STEP 2: Find the document.

    Pipeline:
        grayscale -> blur -> edge detection -> find contours ->
        pick the largest 4-sided one

    Returns the 4 corner points of the document, or None if not found.
    """
    # 1. Grayscale — edge detection works on intensity, not color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur — removes small noise/texture that would create
    #    thousands of tiny false edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny edge detection — finds sharp intensity changes (edges)
    edged = cv2.Canny(blurred, 75, 200)

    # 4. Find contours — connected edge regions.
    #    RETR_LIST = get all contours, not just outer ones
    #    CHAIN_APPROX_SIMPLE = compress contour points to save memory
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # 5. Sort contours by area, largest first. The document should be
    #    the biggest thing in the photo (that's a real assumption —
    #    it's why the doc needs to fill most of the frame).
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for c in contours:
        # Approximate the contour shape — smooths it into a simpler polygon
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        # A document is a rectangle, i.e. 4 points
        if len(approx) == 4:
            doc_contour = approx
            break

    return doc_contour, edged


def four_point_transform(image, pts):
    """
    STEP 3: Flatten the document.

    Given the 4 corner points (in any order) and the original image,
    compute a perspective transform and warp the image so the document
    fills the frame as a perfect top-down rectangle.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image: the max of the top edge
    # and bottom edge distances (photo may be at an angle, so these differ)
    width_a = distance(br, bl)
    width_b = distance(tr, tl)
    max_width = max(int(width_a), int(width_b))

    # Same idea for height, using the left and right edges
    height_a = distance(tr, br)
    height_b = distance(tl, bl)
    max_height = max(int(height_a), int(height_b))

    # Destination points: a perfect rectangle of the size we just computed
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Compute the transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def scan_document(image):
    """
    High-level convenience function combining detection + flattening.
    Falls back to the original image if no 4-sided contour is found
    (e.g. background is too cluttered, or the whole photo IS the document
    edge-to-edge).

    Returns: (warped_image, edged_image_for_debugging, contour_or_None)
    """
    contour, edged = find_document_contour(image)

    if contour is None:
        # No document boundary found — assume the photo is already
        # roughly the document (common when photographed close-up)
        return image.copy(), edged, None

    warped = four_point_transform(image, contour)
    return warped, edged, contour

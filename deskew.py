"""
deskew.py — Step 4 of the pipeline: detect and fix small rotational skew
left over after the perspective correction.
"""

import cv2
import numpy as np


def compute_skew_angle(image):
    """
    Estimate how many degrees the text is tilted.

    Pipeline:
        threshold -> dilate horizontally (lines become solid blocks) ->
        find contours -> minAreaRect per block -> take the median angle

    Why dilate horizontally?
    Individual letters are small, separate shapes. If we tried to measure
    angle on individual letters, noise would dominate. Dilating smears
    each line of text into one solid horizontal blob, so minAreaRect
    gives a much more reliable angle for that line.

    Why median (not mean/average)?
    A few bad detections (e.g. a stray mark, a photo artifact) can throw
    off an average badly. The median is robust to those outliers.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold: turn the image into pure black/white so text = white blobs
    # on black background. THRESH_BINARY_INV + OTSU auto-picks the best
    # threshold value for this specific image.
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Dilate horizontally: a wide, short kernel merges letters/words
    # along a line into one connected blob per text line.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    angles = []
    for c in contours:
        # Ignore tiny specks — they're noise, not real text lines
        if cv2.contourArea(c) < 100:
            continue
        rect = cv2.minAreaRect(c)
        angle = rect[-1]

        # OpenCV's angle convention is quirky: minAreaRect returns angles
        # in the range (-90, 0]. We normalize so tilts are small numbers
        # close to 0 (slightly left or right), not close to -90.
        if angle < -45:
            angle = 90 + angle
        angles.append(angle)

    if len(angles) == 0:
        return 0.0

    return float(np.median(angles))


def deskew(image):
    """
    Rotate the image by the negative of the detected skew angle,
    straightening the text.

    Returns: (deskewed_image, angle_that_was_corrected)
    """
    angle = compute_skew_angle(image)

    # If the angle is negligible, don't bother rotating —
    # rotation always slightly degrades image quality (resampling),
    # so skip it when it wouldn't help.
    if abs(angle) < 0.5:
        return image.copy(), 0.0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Build rotation matrix and apply it.
    # borderMode=REPLICATE avoids black triangles appearing at the
    # rotated corners.
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, angle

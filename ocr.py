"""
ocr.py — Step 6 of the pipeline: extract text from the cleaned scan
using the Tesseract OCR engine (via pytesseract).
"""

import cv2
import pytesseract


def prepare_for_ocr(image):
    """
    Tesseract accuracy depends heavily on image quality. A clean, high
    contrast, pure black-text-on-white-background image works far better
    than a raw color photo with uneven lighting.

    Pipeline:
        grayscale -> adaptive threshold

    Why ADAPTIVE threshold instead of a single global threshold?
    A photo often has uneven lighting (e.g. shadow across one side of
    the page). A single global threshold value would turn the shadowed
    side into a black blob. Adaptive thresholding recalculates the
    threshold for small local regions, so it compensates for lighting
    that varies across the page.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,   # size of the local neighborhood examined
        C=15            # constant subtracted from the local mean
    )

    return thresh


def extract_text(image):
    """
    Run Tesseract OCR on the given image and return the extracted text
    as a plain string.
    """
    ocr_ready = prepare_for_ocr(image)
    text = pytesseract.image_to_string(ocr_ready)
    return text

"""
main.py — CLI entry point.

Usage:
    python main.py --input photo.jpg --output scan.jpg --ocr --text-out result.txt
"""

import argparse
import os
import sys
import cv2

from scanner import scan_document
from deskew import deskew
from ocr import extract_text


def save_debug_artifacts(debug_dir, image, edged, contour, warped, final_image):
    """
    Write out each stage of the pipeline as its own image file, numbered
    in pipeline order. This is what you'd screenshot/attach for a project
    report to show "proof of work" at every stage.

    Produces:
        01_input.jpg                  - the original photo, untouched
        02_edges.jpg                  - Canny edge-detection output
        03_contour.jpg                - the detected document boundary,
                                         drawn in green on the original photo
        04_perspective_corrected.jpg  - the flattened, top-down document
        05_deskewed.jpg               - final image after tilt correction
    (06_extracted_text.txt is written separately, after OCR runs)
    """
    os.makedirs(debug_dir, exist_ok=True)

    # 1. Input image, saved as-is so the report has a clean "before" shot
    cv2.imwrite(os.path.join(debug_dir, "01_input.jpg"), image)

    # 2. Edge-detected image. Canny output is single-channel (grayscale),
    #    which cv2.imwrite handles fine directly.
    cv2.imwrite(os.path.join(debug_dir, "02_edges.jpg"), edged)

    # 3. Contour overlay: draw the detected 4-point boundary in green,
    #    on top of a COPY of the original image (never draw on the
    #    original array in place, or you'd corrupt it for later steps).
    contour_vis = image.copy()
    if contour is not None:
        cv2.drawContours(contour_vis, [contour], -1, (0, 255, 0), 3)
    else:
        # No contour found — note that visually instead of leaving a
        # plain, unlabeled image that looks like a mistake.
        cv2.putText(
            contour_vis, "No 4-point contour found", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    cv2.imwrite(os.path.join(debug_dir, "03_contour.jpg"), contour_vis)

    # 4. Perspective-corrected (flattened) image, before deskewing
    cv2.imwrite(os.path.join(debug_dir, "04_perspective_corrected.jpg"), warped)

    # 5. Final deskewed image (what --output also saves)
    cv2.imwrite(os.path.join(debug_dir, "05_deskewed.jpg"), final_image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Turn a photo of a document into a clean, flat scan "
                     "(and optionally extract its text)."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input photo (JPG/PNG)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to save the corrected image."
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="Enable text extraction with Tesseract."
    )
    parser.add_argument(
        "--text-out", default=None,
        help="Path to save extracted text. Only used with --ocr."
    )
    parser.add_argument(
        "--deskew", dest="deskew", action="store_true", default=True,
        help="Enable deskewing (default: on)."
    )
    parser.add_argument(
        "--no-deskew", dest="deskew", action="store_false",
        help="Disable deskewing."
    )
    parser.add_argument(
        "--debug", default=None, metavar="DIR",
        help="Save every intermediate pipeline artifact (edges, contour "
             "overlay, perspective-corrected image, deskewed image) into "
             "DIR, for reports/documentation."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Step 1: Load the image ---
    image = cv2.imread(args.input)
    if image is None:
        print(f"ERROR: could not read image at '{args.input}'. "
              f"Check the path and that it's a valid JPG/PNG.")
        sys.exit(1)

    print(f"[1/4] Loaded image: {args.input}  (shape={image.shape})")

    # --- Steps 2 & 3: Find the document and flatten it ---
    warped, edged, contour = scan_document(image)
    if contour is None:
        print("[2/4] No 4-sided document contour found — "
              "proceeding with the full photo as the document.")
    else:
        print("[2/4] Document detected and perspective-corrected.")

    # --- Step 4: Fix the tilt ---
    if args.deskew:
        final_image, angle = deskew(warped)
        print(f"[3/4] Deskew applied — corrected {angle:.2f} degrees of tilt.")
    else:
        final_image = warped
        print("[3/4] Deskew skipped (--no-deskew).")

    # --- Step 5: Save the clean image ---
    cv2.imwrite(args.output, final_image)
    print(f"[4/4] Saved corrected scan to: {args.output}")

    # --- Debug artifacts (optional): stages 1-5 of the six report artifacts ---
    if args.debug:
        save_debug_artifacts(args.debug, image, edged, contour, warped, final_image)
        print(f"Saved debug artifacts (input/edges/contour/warped/deskewed) "
              f"to: {args.debug}/")

    # --- Step 6: Extract text (optional) ---
    if args.ocr:
        print("Running OCR...")
        text = extract_text(final_image)

        if args.text_out:
            with open(args.text_out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted text saved to: {args.text_out}")
        else:
            print("----- Extracted Text -----")
            print(text)
            print("---------------------------")

        # 6th debug artifact: the extracted text itself, saved alongside
        # the other 5 so the whole report set lives in one folder.
        if args.debug:
            debug_text_path = os.path.join(args.debug, "06_extracted_text.txt")
            with open(debug_text_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved extracted text copy to: {debug_text_path}")


if __name__ == "__main__":
    main()

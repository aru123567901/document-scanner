# Document Scanner

A command-line tool that takes a photo of a printed document and outputs a
clean, flat, top-down scan — plus (optionally) the extracted text.

It does for a phone photo what a flatbed scanner does: corrects the
perspective, straightens tilted text, and cleans up lighting, then runs
OCR if you want the text pulled out too.

## What it does

Given a photo taken at an angle, with some skew and uneven lighting, the
tool:

1. Detects the document's four corners in the photo
2. Warps the perspective so the document becomes a flat rectangle
3. Detects and corrects any remaining tilt in the text
4. Saves a clean scan image
5. (Optional) Runs Tesseract OCR and saves the extracted text to a `.txt` file

## Installation

### 1. Clone the repo and install Python dependencies

```bash
git clone https://github.com/aru123567901/document-scanner.git
cd document-scanner
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (required for `--ocr`)

Tesseract is a separate system program, not a Python package — `pip
install` alone will not get you OCR. Install it for your OS:

- **Windows**: download and run the [UB-Mannheim Tesseract
  installer](https://github.com/UB-Mannheim/tesseract/wiki), then add the
  install folder to your system `PATH`.
- **macOS**: `brew install tesseract`
- **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`

Verify it's installed:

```bash
tesseract --version
```

If that prints a version number, you're set. If `pytesseract` can't find
Tesseract even after installing it, point to it explicitly at the top of
`ocr.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r"/path/to/tesseract"
```

## Usage

Basic scan (no OCR):

```bash
python main.py --input photo.jpg --output scan.jpg
```

Scan + extract text to a file:

```bash
python main.py --input photo.jpg --output scan.jpg --ocr --text-out result.txt
```

Scan + print extracted text straight to the terminal:

```bash
python main.py --input photo.jpg --output scan.jpg --ocr
```

Skip deskewing (if your photo is already straight):

```bash
python main.py --input photo.jpg --output scan.jpg --no-deskew
```

Save every intermediate pipeline stage as its own image (useful for a
project report — see "Report artifacts" below):

```bash
python main.py --input photo.jpg --output scan.jpg --ocr --text-out result.txt --debug report_images
```

### CLI arguments

| Argument      | Required? | Description                                         |
|---------------|-----------|------------------------------------------------------|
| `--input`     | Yes       | Path to the input photo                              |
| `--output`    | Yes       | Path to save the corrected image                     |
| `--ocr`       | No        | Enable text extraction                                |
| `--text-out`  | No        | Path to save extracted text (only used with `--ocr`) |
| `--deskew`    | No        | Enable deskewing (default: on)                        |
| `--no-deskew` | No        | Disable deskewing                                     |
| `--debug DIR` | No        | Save every intermediate stage as an image into `DIR` |

## Report artifacts

Run with `--debug <folder>` and `--ocr --text-out <file>` together, and
the tool writes out all six proof-of-work artifacts for your report in
one place:

```
report_images/
├── 01_input.jpg                  # the raw photo, untouched
├── 02_edges.jpg                  # Canny edge-detection output
├── 03_contour.jpg                # detected document boundary, drawn in green
├── 04_perspective_corrected.jpg  # flattened, top-down document
├── 05_deskewed.jpg               # final image after tilt correction
└── 06_extracted_text.txt         # OCR output (same as --text-out)
```

Drop these six files straight into your report as the pipeline
walkthrough — each one corresponds to one stage of the "How it works"
table above.

## How it works (pipeline)

| Stage | File | What happens |
|---|---|---|
| Load | `main.py` | Read the image into memory with OpenCV |
| Detect document | `scanner.py` | Grayscale → blur → Canny edges → contours → largest 4-sided shape |
| Flatten | `scanner.py` | Order the 4 corners, compute a perspective transform, warp to a rectangle |
| Deskew | `deskew.py` | Threshold → dilate text lines → measure angle per line → rotate by the median |
| Extract text | `ocr.py` | Adaptive threshold → Tesseract OCR |

## Project structure

```
document-scanner/
├── main.py              # CLI entry point — parses arguments, calls the pipeline
├── scanner.py            # Detect document, correct perspective
├── deskew.py              # Detect and fix skew
├── ocr.py                 # Tesseract wrapper
├── utils.py               # Helper functions (order_points, etc.)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Scope and limitations

**Handles well:**
- Printed text documents
- Photos taken at moderate angles
- Uneven lighting (adaptive thresholding compensates)
- Slight to moderate skew

**Does not handle:**
- Handwritten text
- Stamps or logos overlaid on text
- Multi-column complex layouts
- Severely crumpled or torn paper

These are deliberate scope boundaries, not bugs — the pipeline is built
and tuned around flat, printed, single-column documents.

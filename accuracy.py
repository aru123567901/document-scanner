"""
accuracy.py — Compare OCR output against ground truth.

Uses normalized Levenshtein similarity, which is tolerant of
whitespace differences and small character-level errors.
"""

import os
import re
import sys


def normalize(text):
    """Lowercase, collapse all whitespace, strip punctuation noise."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)   # collapse newlines/tabs/multiple spaces
    return text.strip()


def levenshtein(a, b):
    """Classic edit-distance DP."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = curr
    return prev[-1]


def accuracy(gt_text, out_text):
    """
    Character accuracy = 1 - (edit_distance / len(ground_truth)).
    This is the standard OCR metric.
    """
    gt = normalize(gt_text)
    out = normalize(out_text)
    if len(gt) == 0:
        return 0.0
    dist = levenshtein(gt, out)
    return max(0.0, (1 - dist / len(gt)) * 100)


def main():
    gt_files = sorted(
        f for f in os.listdir('.')
        if f.startswith('ground_truth') and f.endswith('.txt')
    )
    if not gt_files:
        print("No ground_truth*.txt files found in this folder.")
        print("Rename your ground truth to ground_truth_1.txt, ground_truth_2.txt, etc.")
        return

    results = []
    for gt_file in gt_files:
        # Extract the number if present, else match by position
        m = re.search(r'ground_truth_?(\d*)', gt_file)
        num = m.group(1) if m and m.group(1) else ""

        if num:
            out_file = f"output_{num}.txt"
        else:
            # No number -> expect test_output.txt
            out_file = "test_output.txt"

        if not os.path.exists(out_file):
            print(f"[SKIP] {gt_file} - no matching {out_file}")
            continue

        gt_text = open(gt_file, encoding='utf-8').read().strip()
        out_text = open(out_file, encoding='utf-8').read().strip()

        acc = accuracy(gt_text, out_text)
        results.append((gt_file, out_file, acc))
        print(f"{gt_file} vs {out_file}: {acc:.2f}%")

    if results:
        avg = sum(r[2] for r in results) / len(results)
        print()
        print(f"Average accuracy: {avg:.2f}%")
        print(f"Documents tested: {len(results)}")


if __name__ == "__main__":
    main()
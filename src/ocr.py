from __future__ import annotations

import json
import os
import glob
from typing import List

from PIL import Image
import pytesseract
from tqdm import tqdm

from .models import OCRRecord, ocr_to_jsonable


def run_ocr_on_frames(
    frame_dir: str,
    ocr_output_path: str,
    frame_interval_seconds: int = 3,
    ocr_frame_stride: int = 2,
) -> List[OCRRecord]:
    """
    Runs OCR on sampled frames and stores the result.
    The `frame_interval_seconds` must match the interval used when extracting frames.
    """
    ocr_output_dir = os.path.dirname(ocr_output_path)
    if ocr_output_dir:
        os.makedirs(ocr_output_dir, exist_ok=True)

    if os.path.exists(ocr_output_path):
        print(f"OCR output already exists: {ocr_output_path}")
        with open(ocr_output_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = [OCRRecord.from_dict(x) for x in raw]
        print(f"Loaded OCR text for {len(records)} frames.")
        return records

    records: List[OCRRecord] = []
    print("Running OCR on sampled frames …")

    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))

    for idx, frame_path in enumerate(tqdm(frame_files)):
        if idx % ocr_frame_stride != 0:
            continue

        approx_time = idx * frame_interval_seconds

        with Image.open(frame_path) as img:
            text = pytesseract.image_to_string(img)

        records.append(
            OCRRecord(
                time=float(approx_time),
                frame=os.path.basename(frame_path),
                text=text.strip(),
            )
        )

    with open(ocr_output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_to_jsonable(records), f, ensure_ascii=False, indent=2)

    print(f"Saved OCR output for {len(records)} frames to: {ocr_output_path}")
    return records


def preview_ocr(records: List[OCRRecord], n: int = 5) -> None:
    print(f"OCR records: {len(records)}")
    for rec in records[:n]:
        print(f"[t ~ {rec.time:.1f}s] frame={rec.frame}")
        print(rec.text)
        print("-" * 40)

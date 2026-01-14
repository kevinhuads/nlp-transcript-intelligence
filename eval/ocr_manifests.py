import csv
import os
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import fitz
import pandas as pd
from tqdm.auto import tqdm


@dataclass(frozen=True)
class cols:
    time: str
    slide: str
    text: Optional[str]


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def parse_timestamp_to_seconds(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip()
    if not s:
        return None

    m = re.match(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.(\d+))?$", s)
    if m:
        h = int(m.group(1) or 0)
        mn = int(m.group(2))
        sec = int(m.group(3))
        frac = m.group(4) or "0"
        ms = int((frac + "000")[:3])
        return float(h * 3600 + mn * 60 + sec) + ms / 1000.0

    try:
        return float(s)
    except ValueError:
        return None


def detect_columns(df: pd.DataFrame) -> cols:
    names = {c: norm_key(c) for c in df.columns}
    time_candidates = [c for c, k in names.items() if "time" in k]
    slide_candidates = [c for c, k in names.items() if "slide" in k or "label" in k]
    text_candidates = [c for c, k in names.items() if "sentence" in k or "transcript" in k or k == "text"]

    if not time_candidates:
        time_candidates = [df.columns[0]]
    if not slide_candidates:
        slide_candidates = [df.columns[1]]

    time_col = time_candidates[0]
    slide_col = slide_candidates[0]
    text_col = text_candidates[0] if text_candidates else None

    return cols(time=time_col, slide=slide_col, text=text_col)


def find_best_match_file(root: str, stem_hint: str, exts: tuple[str, ...]) -> Optional[str]:
    hint = norm_key(stem_hint)
    best = None
    best_score = -1

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            score = 0
            k = norm_key(os.path.splitext(fn)[0])
            if hint and (hint in k or k in hint):
                score = min(len(hint), len(k))
            if score > best_score:
                best_score = score
                best = os.path.join(dirpath, fn)

    return best


def extract_pdf_page_text(pdf_path: str, page_index: int) -> str:
    doc = fitz.open(pdf_path)
    try:
        if page_index < 0 or page_index >= doc.page_count:
            return ""
        page = doc.load_page(page_index)
        return page.get_text("text").strip()
    finally:
        doc.close()


def save_frame_at_time(video_path: str, t_seconds: float, out_path: str) -> bool:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t_seconds) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return bool(cv2.imwrite(out_path, frame))
    finally:
        cap.release()




def _collect_lecture_items(mavils_data_dir: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    deny_tokens = (
        "audiomatching",
        "imagematching",
        "matchingwithall",
        "maxmatching",
        "meanmatching",
        "weightedsum",
        "sift",
        "kernelmatching",
        "ocr",
    )

    for dirpath, _, filenames in os.walk(mavils_data_dir):
        for fn in filenames:
            lf = fn.lower()
            if not lf.endswith((".xlsx", ".xlsm", ".xls")):
                continue

            stem = os.path.splitext(lf)[0]
            if any(t in stem for t in deny_tokens):
                continue

            if "groundtruth" not in stem and "labeled" not in stem:
                continue

            items.append((dirpath, fn))

    return items


def build_manifest(
    mavils_data_dir: str,
    videos_dir: str,
    out_csv_path: str,
    out_frames_dir: str,
    slide_index_base: int = 0,
    max_items: Optional[int] = None,
) -> None:
    rows_out = []
    count = 0

    lecture_items = _collect_lecture_items(mavils_data_dir)

    for dirpath, xlsx_name in tqdm(lecture_items, desc="Lectures", unit="file"):
        gt_path = os.path.join(dirpath, xlsx_name)

        pdf_candidates = [f for f in os.listdir(dirpath) if f.lower().endswith(".pdf")]
        pdf_path = os.path.join(dirpath, pdf_candidates[0]) if pdf_candidates else None
        if pdf_path is None:
            pdf_path = find_best_match_file(mavils_data_dir, os.path.splitext(xlsx_name)[0], (".pdf",))
        if pdf_path is None:
            continue

        lecture_id = norm_key(os.path.splitext(xlsx_name)[0])
        video_path = find_best_match_file(videos_dir, lecture_id, (".mp4", ".mkv", ".webm", ".mov", ".avi"))
        if video_path is None:
            continue

        df = pd.read_excel(gt_path, sheet_name=0, engine="openpyxl")
        if df is None or not len(df):
            continue

        c = detect_columns(df)

        row_iter = tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Rows ({lecture_id})",
            unit="row",
            leave=False,
        )

        for idx, r in row_iter:
            if max_items is not None and count >= int(max_items):
                break

            t = parse_timestamp_to_seconds(r.get(c.time))
            if t is None:
                continue

            try:
                slide_label = int(r.get(c.slide))
            except Exception:
                continue

            if slide_label < 0:
                continue

            page_index = slide_label - int(slide_index_base)
            ref_text = extract_pdf_page_text(pdf_path, page_index)
            if not ref_text:
                continue

            sample_id = f"{lecture_id}_{idx:06d}_s{slide_label}"
            img_path = os.path.join(out_frames_dir, lecture_id, f"{sample_id}.jpg")
            ok = save_frame_at_time(video_path, t, img_path)
            if not ok:
                continue

            rows_out.append(
                {
                    "id": sample_id,
                    "lecture": lecture_id,
                    "timestamp_s": round(float(t), 3),
                    "slide_label": int(slide_label),
                    "image_path": img_path,
                    "ref_text": ref_text,
                }
            )
            count += 1

            row_iter.set_postfix_str(f"kept={count}")

        if max_items is not None and count >= int(max_items):
            break

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "lecture", "timestamp_s", "slide_label", "image_path", "ref_text"],
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)



if __name__ == "__main__":
    mavils_data_dir = os.path.join(r"D:\NLP-Videos-Data\MaViLS")
    videos_dir = os.path.join(mavils_data_dir, "video")
    out_csv_path = os.path.join("eval", "manifests", "mavils_ocr_manifest.csv")
    out_frames_dir = os.path.join("eval", "ocr_frames")

    build_manifest(
        mavils_data_dir=mavils_data_dir,
        videos_dir=videos_dir,
        out_csv_path=out_csv_path,
        out_frames_dir=out_frames_dir,
        slide_index_base=0,
        max_items=None,
    )

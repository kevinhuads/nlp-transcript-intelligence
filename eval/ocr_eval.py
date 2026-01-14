from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import time
import warnings
from contextlib import nullcontext
from typing import Dict, List, Optional

import mlflow
import pandas as pd
import tqdm as tqdm_module
import yaml
from tqdm.auto import tqdm

from ocr_backends import build_ocr, parse_models_cfg
from ocr_metrics import compute_char_measures, compute_word_measures, measures_to_counts, normalize


tqdm_module.tqdm.monitor_interval = 0

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["DO_NOT_TRACK"] = "true"

for pattern in (
    r"pkg_resources is deprecated as an API\..*",
    r"Some weights of .* were not initialized.*",
):
    warnings.filterwarnings("ignore", message=pattern, category=UserWarning)

logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OCR (YAML-driven backends).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model-key", type=str, default=None)
    return p.parse_args()


def load_eval_cfg(config_path: str) -> dict:
    config_path = os.path.expanduser(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "eval_ocr" not in data:
        raise KeyError("Missing top-level key: eval_ocr")
    return data["eval_ocr"]


def safe_filename_token(text: str) -> str:
    s = str(text)
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    return s


def shorten_image_path_for_csv(image_path: str) -> str:
    p = os.path.normpath(str(image_path))
    parts = p.split(os.sep)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "eval" and i + 1 < len(parts) and parts[i + 1] == "ocr_frames":
            return os.sep.join(parts[i + 1 :])
    return os.path.basename(p)


def main() -> int:
    args = parse_args()
    cfg = load_eval_cfg(args.config)

    manifest_path = os.path.abspath(os.path.expanduser(str(cfg["manifest"])))
    df = pd.read_csv(manifest_path)

    max_items = cfg.get("max_items", None)
    if max_items is not None and int(max_items) < len(df):
        df = df.iloc[: int(max_items)].copy()

    dataset = str(cfg.get("dataset", "mavils"))
    output_folder = os.path.join(str(cfg["output_base_folder"]), dataset)
    os.makedirs(output_folder, exist_ok=True)

    models_cfg = parse_models_cfg(cfg.get("models", []))
    selected_specs = models_cfg
    if args.model_key is not None:
        selected_specs = [m for m in models_cfg if m.key == args.model_key]
        if not selected_specs:
            raise KeyError(f"Unknown model key: {args.model_key}")

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", False))
    exp = None
    if mlflow_enabled:
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri"))
        exp = mlflow.set_experiment(mlflow_cfg["experiment"])

    fieldnames = [
        "model",
        "dataset",
        "id",
        "lecture",
        "timestamp_s",
        "slide_label",
        "image_path",
        "cer",
        "wer",
        "word_substitutions",
        "word_deletions",
        "word_insertions",
        "ref_words",
        "hyp_words",
        "char_substitutions",
        "char_deletions",
        "char_insertions",
        "ref_chars",
        "hyp_chars",
        "ocr_seconds",
    ]

    norm_cfg = cfg.get("normalize", {})
    lowercase = bool(norm_cfg.get("lowercase", True))
    strip_punct = bool(norm_cfg.get("strip_punct", True))

    for spec in tqdm(selected_specs, desc="Models", unit="model"):
        model_id = spec.key
        file_id = safe_filename_token(model_id)

        output_csv = os.path.join(output_folder, f"ocr_eval_{file_id}.csv")
        output_json = os.path.join(output_folder, f"summary_ocr_eval_{file_id}.json")

        summary: Dict[str, dict] = {"models": {}}

        with open(output_csv, "w", encoding="utf-8", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()

            run_ctx = (
                mlflow.start_run(experiment_id=exp.experiment_id, run_name=f"{model_id}_{dataset}")
                if mlflow_enabled and exp is not None
                else nullcontext()
            )

            ocr = None
            with run_ctx:
                tqdm.write(f"[INFO] Loading model: key={spec.key} backend={spec.backend} name={spec.name}")
                try:
                    ocr = build_ocr(spec, cfg)

                    total_word_s = 0
                    total_word_d = 0
                    total_word_i = 0
                    total_word_n = 0

                    total_char_s = 0
                    total_char_d = 0
                    total_char_i = 0
                    total_char_n = 0

                    total_ocr_seconds = 0.0
                    item_cers: List[float] = []
                    item_wers: List[float] = []

                    for _, row in tqdm(
                        df.iterrows(),
                        total=len(df),
                        desc=f"Samples ({model_id})",
                        unit="img",
                        leave=False,
                    ):
                        sample_id = str(row["id"])
                        lecture = str(row["lecture"])
                        timestamp_s = float(row["timestamp_s"])
                        slide_label = int(row["slide_label"])
                        image_path = str(row["image_path"])
                        ref_text_raw = str(row["ref_text"])

                        ref_norm = normalize(ref_text_raw, lowercase=lowercase, strip_punct=strip_punct)

                        t0 = time.time()
                        hyp_raw = ocr.read_text(image_path)
                        ocr_seconds = time.time() - t0
                        total_ocr_seconds += ocr_seconds

                        hyp_norm = normalize(hyp_raw, lowercase=lowercase, strip_punct=strip_punct)

                        wm = compute_word_measures(ref_norm, hyp_norm)
                        cm = compute_char_measures(ref_norm, hyp_norm)

                        ws, wd, wi, wn = measures_to_counts(wm)
                        cs, cd, ci, cn = measures_to_counts(cm)

                        total_word_s += ws
                        total_word_d += wd
                        total_word_i += wi
                        total_word_n += wn

                        total_char_s += cs
                        total_char_d += cd
                        total_char_i += ci
                        total_char_n += cn

                        wer_val = float(wm["wer"])
                        cer_val = float(cm["cer"])
                        item_wers.append(wer_val)
                        item_cers.append(cer_val)

                        writer.writerow(
                            {
                                "model": model_id,
                                "dataset": dataset,
                                "id": sample_id,
                                "lecture": lecture,
                                "timestamp_s": round(timestamp_s, 3),
                                "slide_label": slide_label,
                                "image_path": shorten_image_path_for_csv(image_path),
                                "cer": cer_val,
                                "wer": wer_val,
                                "word_substitutions": ws,
                                "word_deletions": wd,
                                "word_insertions": wi,
                                "ref_words": wn,
                                "hyp_words": len(hyp_norm.split()) if hyp_norm else 0,
                                "char_substitutions": cs,
                                "char_deletions": cd,
                                "char_insertions": ci,
                                "ref_chars": cn,
                                "hyp_chars": len(hyp_norm) if hyp_norm else 0,
                                "ocr_seconds": round(ocr_seconds, 4),
                            }
                        )

                    corpus_wer = ((total_word_s + total_word_d + total_word_i) / total_word_n) if total_word_n > 0 else 0.0
                    corpus_cer = ((total_char_s + total_char_d + total_char_i) / total_char_n) if total_char_n > 0 else 0.0

                    mean_item_wer = (sum(item_wers) / len(item_wers)) if item_wers else 0.0
                    mean_item_cer = (sum(item_cers) / len(item_cers)) if item_cers else 0.0

                    summary["models"][model_id] = {
                        "corpus_wer": corpus_wer,
                        "corpus_cer": corpus_cer,
                        "mean_item_wer": mean_item_wer,
                        "mean_item_cer": mean_item_cer,
                        "total_word_substitutions": total_word_s,
                        "total_word_deletions": total_word_d,
                        "total_word_insertions": total_word_i,
                        "total_ref_words": total_word_n,
                        "total_char_substitutions": total_char_s,
                        "total_char_deletions": total_char_d,
                        "total_char_insertions": total_char_i,
                        "total_ref_chars": total_char_n,
                        "total_ocr_seconds": total_ocr_seconds,
                        "items": int(len(df)),
                        "settings": {
                            "model_key": spec.key,
                            "backend": spec.backend,
                            "name": spec.name,
                            "dataset": dataset,
                            "max_items": max_items,
                            "device": str(cfg.get("device", "cpu")),
                            "compute_type": str(cfg.get("compute_type", "float32")),
                            "normalize": {"lowercase": lowercase, "strip_punct": strip_punct},
                            "model_params": spec.params,
                        },
                    }

                    tqdm.write(
                        f"[RESULT] model={model_id} corpus_CER={corpus_cer:.4f} corpus_WER={corpus_wer:.4f} "
                        f"ocr_seconds={total_ocr_seconds:.2f}"
                    )
                finally:
                    if ocr is not None:
                        try:
                            ocr.close()
                        finally:
                            del ocr
                    gc.collect()

        with open(output_json, "w", encoding="utf-8") as fjson:
            json.dump(summary, fjson, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

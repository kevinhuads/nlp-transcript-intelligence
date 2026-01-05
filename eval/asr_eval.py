from __future__ import annotations

import os
import warnings
import logging
import argparse
import faulthandler
import gc
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import mlflow
import tqdm as tqdm_module
import yaml
from tqdm.auto import tqdm

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for pattern in (
    r"pkg_resources is deprecated as an API\..*",
    r"Some weights of .* were not initialized.*",
    r"You should probably TRAIN this model.*",
):
    warnings.filterwarnings("ignore", message=pattern, category=UserWarning)


logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("nemo").setLevel(logging.ERROR)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("nemo_logger").propagate = False
logging.getLogger("nemo_logging").setLevel(logging.ERROR)
logging.getLogger("nemo_logger.nemo_logging").setLevel(logging.ERROR)
logging.getLogger("lhotse").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)

# Disable MLflow telemetry deterministically. setdefault() does not override an existing value.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["DO_NOT_TRACK"] = "true"



from asr_backends import build_transcriber, parse_models_cfg
from asr_metrics import compute_measures, load_manifest_csv, measures_to_counts, normalize, shorten_audio_path_for_csv

# Disable tqdm's monitor thread to reduce background thread activity.
tqdm_module.tqdm.monitor_interval = 0


def move_into_results_samples(folder: str) -> str:
    parts = os.path.normpath(folder).split(os.sep)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "results":
            parts.insert(i + 1, "samples")
            return os.sep.join(parts)
    d = os.path.dirname(folder)
    b = os.path.basename(folder)
    return os.path.join(d, "samples", b)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ASR with WER (YAML-driven backends).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model-key", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--spawn-per-model", action="store_true")
    return p.parse_args()


def load_eval_cfg(config_path: str) -> dict:
    config_path = os.path.expanduser(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "eval_asr" not in data:
        raise KeyError("Missing top-level key: eval_asr")
    return data["eval_asr"]


def is_windows_cuda(device: str) -> bool:
    return os.name == "nt" and device.lower() == "cuda"


def resolve_model_key_arg(args: argparse.Namespace) -> Optional[str]:
    if args.model_key is not None:
        return args.model_key
    if args.model is not None:
        return args.model
    return None


def should_spawn_workers(cfg: dict, args: argparse.Namespace, device: str) -> bool:
    if args.model_key is not None or args.model is not None:
        return False

    if is_windows_cuda(device):
        return True

    if args.spawn_per_model:
        return True

    cfg_value = cfg.get("spawn_per_model", None)
    if cfg_value is not None:
        return bool(cfg_value)

    return False


def run_worker_subprocesses(config_path: str, model_keys: List[str]) -> int:
    script_path = os.path.abspath(sys.argv[0])
    rc_max = 0
    for model_key in model_keys:
        cmd = [sys.executable, script_path, "--config", config_path, "--model-key", model_key]
        rc = subprocess.run(cmd).returncode
        rc_max = max(rc_max, rc)
    return rc_max

def safe_filename_token(text: str) -> str:
    s = str(text)
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    return s


def make_model_id(model_key: str, max_items: Optional[int]) -> str:
    if max_items is None:
        return model_key
    return f"{model_key}{int(max_items)}"


def main() -> int:
    faulthandler.enable(all_threads=True)

    args = parse_args()
    cfg = load_eval_cfg(args.config)

    max_items = cfg.get("max_items")
    dataset = str(cfg["dataset"])
    device: str = str(cfg["device"])

    models_cfg = parse_models_cfg(cfg.get("models"))
    model_keys = [m.key for m in models_cfg]

    if should_spawn_workers(cfg, args, device):
        return run_worker_subprocesses(args.config, model_keys)

    selected_key = resolve_model_key_arg(args)
    if selected_key is None:
        selected_specs = models_cfg
    else:
        selected_specs = [m for m in models_cfg if m.key == selected_key]
        if not selected_specs:
            raise KeyError(f"Unknown model key: {selected_key}")

    manifest_path = os.path.abspath(os.path.expanduser(f"eval/manifests/librispeech_{dataset}_manifest.csv"))
    manifest_dir = os.path.dirname(manifest_path)
    rows = load_manifest_csv(manifest_path)
    if max_items is not None and int(max_items) < len(rows):
        rows = rows[: int(max_items)]

    output_folder = os.path.join(str(cfg["output_base_folder"]), dataset)
    if max_items is not None:
        output_folder = move_into_results_samples(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True))
    exp = None
    if mlflow_enabled:
        mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri"))
        exp = mlflow.set_experiment(mlflow_cfg["experiment"])

    fieldnames = [
        "model",
        "dataset",
        "id",
        "audio_path",
        "wer",
        "substitutions",
        "deletions",
        "insertions",
        "ref_words",
        "hyp_words",
        "asr_seconds",
    ]

    windows_cuda = is_windows_cuda(device)
    hard_exit_after_model = bool(selected_key is not None and windows_cuda)

    for spec in tqdm(selected_specs, desc="Models", unit="model", file=sys.stdout, position=0):
        model_id = make_model_id(spec.key, max_items)
        file_id = safe_filename_token(model_id)

        output_csv = os.path.join(output_folder, f"asr_eval_{file_id}.csv")
        output_json = os.path.join(output_folder, f"summary_asr_eval_{file_id}.json")

        summary: Dict[str, dict] = {"models": {}}

        with open(output_csv, "w", encoding="utf-8", newline="") as fcsv:
            import csv

            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()

            run_ctx = (
                mlflow.start_run(experiment_id=exp.experiment_id, run_name=f"{model_id}_{dataset}")
                if mlflow_enabled and exp is not None
                else nullcontext()
            )

            asr = None
            with run_ctx:
                tqdm.write(f"[INFO] Loading model: key={spec.key} backend={spec.backend} name={spec.name}")
                try:
                    asr = build_transcriber(spec, cfg)

                    total_s = 0
                    total_d = 0
                    total_i = 0
                    total_n = 0
                    total_asr_seconds = 0.0
                    item_wers: List[float] = []

                    for row in tqdm(
                        rows,
                        desc=f"Utterances ({model_id})",
                        unit="utt",
                        file=sys.stdout,
                        position=1,
                        leave=False,
                    ):
                        utt_id = row["id"].strip()
                        audio_path = row["audio_path"].strip()
                        ref_text = row["ref_text"].strip()

                        if not os.path.isabs(audio_path):
                            audio_path = os.path.abspath(os.path.join(manifest_dir, audio_path))

                        audio_path_csv = shorten_audio_path_for_csv(audio_path, dataset)

                        ref = normalize(ref_text)

                        t0 = time.time()
                        hyp_raw = asr.transcribe(audio_path)
                        asr_seconds = time.time() - t0
                        total_asr_seconds += asr_seconds

                        hyp = normalize(hyp_raw)

                        m = compute_measures(ref, hyp)
                        s, d, i, n = measures_to_counts(m)

                        total_s += s
                        total_d += d
                        total_i += i
                        total_n += n

                        wer_val = float(m["wer"])
                        item_wers.append(wer_val)

                        writer.writerow(
                            {
                                "model": model_id,
                                "dataset": dataset,
                                "id": utt_id,
                                "audio_path": audio_path_csv,
                                "wer": wer_val,
                                "substitutions": s,
                                "deletions": d,
                                "insertions": i,
                                "ref_words": n,
                                "hyp_words": len(hyp.split()),
                                "asr_seconds": round(asr_seconds, 4),
                            }
                        )

                    corpus_wer = ((total_s + total_d + total_i) / total_n) if total_n > 0 else 0.0
                    mean_item_wer = (sum(item_wers) / len(item_wers)) if item_wers else 0.0

                    summary["models"][model_id] = {
                        "corpus_wer": corpus_wer,
                        "mean_item_wer": mean_item_wer,
                        "total_substitutions": total_s,
                        "total_deletions": total_d,
                        "total_insertions": total_i,
                        "total_ref_words": total_n,
                        "total_asr_seconds": total_asr_seconds,
                        "items": len(rows),
                        "settings": {
                            "model_key": spec.key,
                            "backend": spec.backend,
                            "name": spec.name,
                            "dataset": dataset,
                            "max_items": max_items,
                            "device": str(cfg["device"]),
                            "compute_type": str(cfg["compute_type"]),
                            "language": cfg.get("language"),
                            "task": str(cfg["task"]),
                            "beam_size": int(cfg.get("beam_size", 5)),
                            "best_of": int(cfg.get("best_of", 5)),
                            "vad_filter": bool(cfg.get("vad_filter", False)),
                            "model_params": spec.params,
                        },
                    }

                    tqdm.write(
                        f"[RESULT] model={model_id} corpus_WER={corpus_wer:.4f} "
                        f"(S={total_s}, D={total_d}, I={total_i}, N={total_n}) "
                        f"asr_seconds={total_asr_seconds:.2f}"
                    )
                finally:
                    if asr is not None and not windows_cuda:
                        try:
                            asr.close()
                        finally:
                            del asr
                    if not windows_cuda:
                        gc.collect()

        with open(output_json, "w", encoding="utf-8") as fjson:
            json.dump(summary, fjson, indent=2, ensure_ascii=False)

        if hard_exit_after_model:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

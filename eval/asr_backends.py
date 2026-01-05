from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from faster_whisper import WhisperModel
from nemo.collections.asr.models import ASRModel
from nemo.utils import logging as nemo_logging
from omegaconf import open_dict
from transformers import AutoProcessor, Wav2Vec2ForCTC, pipeline

_real_temporarydirectory = tempfile.TemporaryDirectory


def _temporarydirectory_windows_best_effort(*args, **kwargs):
    kwargs.pop("delete", None)
    kwargs["ignore_cleanup_errors"] = True
    return _real_temporarydirectory(*args, **kwargs)


tempfile.TemporaryDirectory = _temporarydirectory_windows_best_effort


@dataclass(frozen=True)
class model_spec:
    key: str
    backend: str
    name: Optional[str] = None
    params: Dict[str, Any] = None


def _normalize_backend_token(token: str) -> str:
    t = token.strip().lower()
    if t in {"fw", "faster_whisper", "faster-whisper", "whisper"}:
        return "faster_whisper"
    if t in {"hf", "transformers", "transformers_pipeline", "pipeline"}:
        return "transformers_pipeline"
    if t in {"nemo", "nvidia_nemo"}:
        return "nemo"
    if t in {"cmd", "command"}:
        return "command"
    if t == "mms":
        return "mms"
    return t


def _infer_backend_from_name(name: str) -> str:
    n = name.strip()
    nl = n.lower()

    if nl.startswith("facebook/mms"):
        return "mms"

    if nl.startswith("nvidia/stt_") or "/stt_" in nl or n.startswith("stt_"):
        return "nemo"

    if "/" in n:
        return "transformers_pipeline"

    return "faster_whisper"


def _default_key_for(name: str, backend: str) -> str:
    key = name.strip()

    if backend == "transformers_pipeline":
        if "/" in key:
            key = key.split("/", 1)[1]
        return key

    if backend == "mms":
        if "/" in key:
            key = key.split("/", 1)[1]
        return key

    if backend == "nemo":
        if "/" in key:
            key = key.split("/", 1)[1]
        if key.startswith("stt_"):
            key = key[len("stt_") :]
        if key.startswith("en_"):
            key = key[len("en_") :]
        return key

    if backend == "faster_whisper":
        if "/" in key:
            key = key.split("/", 1)[1]
        for prefix in ("faster-whisper-", "faster_whisper-", "fasterwhisper-"):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        return key

    return key


_mms_lang_re = re.compile(r"^(?P<name>.+?)(?:[@:])(?P<lang>[A-Za-z]{2,3})$")


def _normalize_mms_lang(lang: str) -> str:
    l = str(lang).strip().lower()
    if l == "en":
        return "eng"
    return l


def _parse_model_string(item: str) -> model_spec:
    raw = item.strip()
    if not raw:
        raise ValueError("Empty model entry")

    name = raw
    backend = _normalize_backend_token(_infer_backend_from_name(name))
    params: Dict[str, Any] = {}

    if backend == "mms":
        m = _mms_lang_re.match(raw)
        if m:
            name = m.group("name").strip()
            params["lang"] = _normalize_mms_lang(m.group("lang"))
        backend = "mms"

    key = _default_key_for(name, backend)
    return model_spec(key=key, backend=backend, name=name, params=params)


def parse_models_cfg(models_cfg: Any) -> List[model_spec]:
    if not isinstance(models_cfg, list):
        raise TypeError("eval_asr.models must be a list")

    out: List[model_spec] = []
    used_keys: set[str] = set()

    for item in models_cfg:
        if isinstance(item, str):
            spec = _parse_model_string(item)
        elif isinstance(item, dict):
            key = str(item.get("key") or item.get("id") or item.get("name") or "").strip()
            backend = _normalize_backend_token(str(item.get("backend") or "faster_whisper").strip())
            name = item.get("name", None)
            params = {k: v for k, v in item.items() if k not in {"key", "id", "backend", "name"}}
            spec = model_spec(key=key, backend=backend, name=name, params=params or {})
        else:
            raise TypeError("Each model entry must be a string or a dict")

        if spec.key in used_keys:
            raise ValueError(f"Duplicate model key: {spec.key}")
        used_keys.add(spec.key)
        out.append(spec)

    return out


class transcriber:
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return


class faster_whisper_transcriber(transcriber):
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        language: Optional[str],
        task: str,
        beam_size: int,
        best_of: int,
        vad_filter: bool,
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._language = language
        self._task = task
        self._beam_size = beam_size
        self._best_of = best_of
        self._vad_filter = vad_filter
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            **(backend_kwargs or {}),
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(
            audio_path,
            language=self._language,
            task=self._task,
            beam_size=self._beam_size,
            best_of=self._best_of,
            vad_filter=self._vad_filter,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text).strip()

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None


class transformers_pipeline_transcriber(transcriber):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        compute_type: str,
        language: Optional[str],
        task: str,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        device_index = 0 if str(device).lower() == "cuda" else -1

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(str(compute_type).lower(), None)

        self._generate_kwargs = dict(generate_kwargs or {})
        if language is not None and "language" not in self._generate_kwargs:
            self._generate_kwargs["language"] = language
        if task and "task" not in self._generate_kwargs:
            self._generate_kwargs["task"] = task

        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name_or_path,
            device=device_index,
            torch_dtype=torch_dtype,
            **(pipeline_kwargs or {}),
        )

    def transcribe(self, audio_path: str) -> str:
        out = self._pipe(audio_path, generate_kwargs=self._generate_kwargs)
        if isinstance(out, dict) and "text" in out:
            return str(out["text"]).strip()
        if isinstance(out, str):
            return out.strip()
        return str(out).strip()

    def close(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None


@contextlib.contextmanager
def suppress_stdio_fds():
    devnull = open(os.devnull, "w", encoding="utf-8")
    try:
        sys.stdout.flush()
        sys.stderr.flush()

        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()

        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)

        os.dup2(devnull.fileno(), stdout_fd)
        os.dup2(devnull.fileno(), stderr_fd)

        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)

        os.close(saved_stdout)
        os.close(saved_stderr)
        devnull.close()

def _disable_cuda_graphs(decoding) -> None:
    for section_name in ("greedy", "greedy_batch"):
        if not hasattr(decoding, section_name):
            continue
        section = getattr(decoding, section_name)
        with open_dict(section):
            if "use_cuda_graph_decoder" in section:
                section.use_cuda_graph_decoder = False
            for k in list(section.keys()):
                if "cuda_graph" in str(k).lower():
                    section[k] = False


class nemo_transcriber(transcriber):
    def __init__(
        self,
        name: str,
        nemo_restore_path: Optional[str] = None,
        nemo_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        nemo_logging.setLevel("ERROR")

        if nemo_restore_path:
            self._model = ASRModel.restore_from(restore_path=nemo_restore_path, **(nemo_kwargs or {}))
        else:
            self._model = ASRModel.from_pretrained(model_name=name, **(nemo_kwargs or {}))

        self._model.eval()
        self._force_cpu = False

        if torch.cuda.is_available():
            self._model.to("cuda")

        decoding = getattr(getattr(self._model, "cfg", None), "decoding", None)
        if decoding is not None:
            _disable_cuda_graphs(decoding)
            change = getattr(self._model, "change_decoding_strategy", None)
            if callable(change):
                try:
                    change(decoding_cfg=decoding)
                except TypeError:
                    change(decoding)

    def transcribe(self, audio_path: str) -> str:
        try:
            out = self._model.transcribe([audio_path], verbose=False)
        except Exception as e:
            msg = str(e)
            if (not self._force_cpu) and ("CUDA failure! 35" in msg or "cuda" in msg.lower()):
                self._force_cpu = True
                self._model.to("cpu")
                out = self._model.transcribe([audio_path], verbose=False)
            else:
                raise

        if isinstance(out, tuple) and out:
            out = out[0]
        if isinstance(out, list) and out:
            out = out[0]
        if hasattr(out, "text"):
            out = out.text
        elif isinstance(out, dict) and "text" in out:
            out = out["text"]

        text = str(out).strip()
        text = text.splitlines()[0].strip()
        text = text.replace("|", " ").replace("_", " ")
        return text


class command_transcriber(transcriber):
    def __init__(
        self,
        cmd: List[str],
        timeout_s: Optional[float] = None,
        stdout_extract: Optional[Dict[str, Any]] = None,
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._cmd = [str(x) for x in cmd]
        self._timeout_s = float(timeout_s) if timeout_s is not None else None
        self._workdir = workdir
        self._env = env
        self._stdout_extract = stdout_extract or None

        self._extract_re: Optional[re.Pattern[str]] = None
        self._extract_group = 1
        if self._stdout_extract is not None:
            pattern = str(self._stdout_extract.get("pattern", "")).strip()
            group = int(self._stdout_extract.get("group", 1))
            if pattern:
                self._extract_re = re.compile(pattern)
                self._extract_group = group

    def transcribe(self, audio_path: str) -> str:
        cmd = [part.replace("{audio}", audio_path) for part in self._cmd]
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self._timeout_s,
            cwd=self._workdir,
            env=self._env,
        )
        if cp.returncode != 0:
            raise RuntimeError(f"command backend failed (rc={cp.returncode}): {cp.stderr.strip()}")

        out = cp.stdout.strip()
        if self._extract_re is not None:
            m = self._extract_re.search(out)
            if m:
                return str(m.group(self._extract_group)).strip()
        return out


class mms_transcriber(transcriber):
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        lang: str,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(str(compute_type).lower(), None)

        processor = AutoProcessor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name, torch_dtype=torch_dtype)

        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "set_target_lang"):
            processor.tokenizer.set_target_lang(str(lang))
        if hasattr(model, "load_adapter"):
            model.load_adapter(str(lang))

        device_index = 0 if str(device).lower() == "cuda" else -1

        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=getattr(processor, "tokenizer", None),
            feature_extractor=getattr(processor, "feature_extractor", None),
            device=device_index,
            torch_dtype=torch_dtype,
            **(pipeline_kwargs or {}),
        )

    def transcribe(self, audio_path: str) -> str:
        out = self._pipe(audio_path)
        if isinstance(out, dict) and "text" in out:
            return str(out["text"]).strip()
        if isinstance(out, str):
            return out.strip()
        return str(out).strip()

    def close(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None


def build_transcriber(spec: model_spec, cfg: dict) -> transcriber:
    device: str = str(cfg["device"])
    compute_type: str = str(cfg["compute_type"])
    language: Optional[str] = cfg.get("language")
    task: str = str(cfg["task"])
    beam_size: int = int(cfg.get("beam_size", 5))
    best_of: int = int(cfg.get("best_of", 5))
    vad_filter: bool = bool(cfg.get("vad_filter", False))

    backend = _normalize_backend_token(spec.backend)

    if backend == "mms":
        if spec.name is None:
            raise ValueError(f"mms backend requires spec.name for model key {spec.key}")

        lang = (spec.params or {}).get("lang") or language or "eng"
        lang = _normalize_mms_lang(lang)

        return mms_transcriber(
            model_name=str(spec.name),
            device=device,
            compute_type=compute_type,
            lang=str(lang),
            pipeline_kwargs=(spec.params or {}).get("pipeline_kwargs", None),
        )

    if backend == "faster_whisper":
        if spec.name is None:
            raise ValueError(f"faster_whisper backend requires spec.name for model key {spec.key}")
        return faster_whisper_transcriber(
            model_name=str(spec.name),
            device=device,
            compute_type=compute_type,
            language=language,
            task=task,
            beam_size=beam_size,
            best_of=best_of,
            vad_filter=vad_filter,
            backend_kwargs=(spec.params or {}).get("backend_kwargs", None),
        )

    if backend == "transformers_pipeline":
        if spec.name is None:
            raise ValueError(f"transformers_pipeline backend requires spec.name for model key {spec.key}")
        return transformers_pipeline_transcriber(
            model_name_or_path=str(spec.name),
            device=device,
            compute_type=compute_type,
            language=language,
            task=task,
            pipeline_kwargs=(spec.params or {}).get("pipeline_kwargs", None),
            generate_kwargs=(spec.params or {}).get("generate_kwargs", None),
        )

    if backend == "nemo":
        if spec.name is None:
            raise ValueError(f"nemo backend requires spec.name for model key {spec.key}")
        return nemo_transcriber(
            name=str(spec.name),
            nemo_restore_path=(spec.params or {}).get("restore_path", None),
            nemo_kwargs=(spec.params or {}).get("nemo_kwargs", None),
        )

    if backend == "command":
        return command_transcriber(
            cmd=(spec.params or {}).get("cmd", None),
            timeout_s=(spec.params or {}).get("timeout_s", None),
            stdout_extract=(spec.params or {}).get("stdout_extract", None),
            workdir=(spec.params or {}).get("workdir", None),
            env=(spec.params or {}).get("env", None),
        )

    raise ValueError(f"Unknown backend '{spec.backend}' for model key {spec.key}")
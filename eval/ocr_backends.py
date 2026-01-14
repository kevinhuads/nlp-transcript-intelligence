from __future__ import annotations

import contextlib
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

_easyocr_re = re.compile(r"^easyocr(?:[@:](?P<langs>[A-Za-z,+_-]+))?$", re.IGNORECASE)
_doctr_re = re.compile(r"^doctr(?:[@:](?P<det>[A-Za-z0-9_]+)\+(?P<rec>[A-Za-z0-9_]+))?$", re.IGNORECASE)
_paddleocr_re = re.compile(r"^paddleocr(?:[@:](?P<variant>[A-Za-z0-9_-]+))?$", re.IGNORECASE)
_tess_re = re.compile(r"^(?:tess|tesseract)(?:[@:](?P<lang>[A-Za-z_+-]+))?$", re.IGNORECASE)
_psm_re = re.compile(r"(?:^|[,;])psm=(?P<psm>\d+)(?:$|[,;])", re.IGNORECASE)
_oem_re = re.compile(r"(?:^|[,;])oem=(?P<oem>\d+)(?:$|[,;])", re.IGNORECASE)

_cloud_map = {
    "google_cloud_vision": ["python", "eval/tools/google_vision_ocr.py", "--image", "{image}"],
    "google_document_ai": ["python", "eval/tools/google_documentai_ocr.py", "--image", "{image}"],
    "azure_read": ["python", "eval/tools/azure_read_ocr.py", "--image", "{image}"],
    "aws_textract": ["python", "eval/tools/aws_textract_ocr.py", "--image", "{image}"],
}



@dataclass(frozen=True)
class model_spec:
    key: str
    backend: str
    name: Optional[str] = None
    params: Dict[str, Any] = None


def _normalize_backend_token(token: str) -> str:
    t = str(token).strip().lower()
    if t in {"tess", "tesseract"}:
        return "tesseract"
    if t in {"trocr", "transformers_trocr", "hf_trocr"}:
        return "transformers_trocr"
    if t in {"cmd", "command"}:
        return "command"
    return t


def _infer_backend_from_name(name: str) -> str:
    n = str(name).strip().lower()
    if _tess_re.match(n):
        return "tesseract"
    if _easyocr_re.match(n):
        return "easyocr"
    if _doctr_re.match(n):
        return "doctr"
    if _paddleocr_re.match(n):
        return "paddleocr"
    if n in _cloud_map:
        return "command"
    if "trocr" in n or n.startswith("microsoft/trocr"):
        return "transformers_trocr"
    if "/" in n:
        return "transformers_trocr"
    return "tesseract"



def _default_key_for(name: Optional[str], backend: str) -> str:
    if backend == "transformers_trocr" and name:
        return str(name).split("/", 1)[-1]
    if name:
        return str(name).split("/", 1)[-1]
    return backend


def _parse_model_string(item: str) -> model_spec:
    raw = str(item).strip()
    if not raw:
        raise ValueError("Empty model entry")

    name = raw
    backend = _normalize_backend_token(_infer_backend_from_name(name))
    params: Dict[str, Any] = {}

    if backend == "tesseract":
        m = _tess_re.match(raw)
        if m:
            lang = m.group("lang")
            if lang:
                params["lang"] = str(lang)

        rest = ""
        if "@" in raw:
            _, rest = raw.split("@", 1)
        elif ":" in raw and raw.lower().startswith(("tess", "tesseract")):
            _, rest = raw.split(":", 1)

        if rest:
            mp = _psm_re.search(rest)
            mo = _oem_re.search(rest)
            if mp:
                params["psm"] = int(mp.group("psm"))
            if mo:
                params["oem"] = int(mo.group("oem"))

        name = None
        key = "tesseract"
        if "lang" in params:
            key = f"{key}-{params['lang']}"
        if "psm" in params:
            key = f"{key}-psm{int(params['psm'])}"
        if "oem" in params:
            key = f"{key}-oem{int(params['oem'])}"

        return model_spec(key=key, backend="tesseract", name=None, params=params)

    if backend == "easyocr":
        m = _easyocr_re.match(raw)
        langs = ["en"]
        if m and m.group("langs"):
            langs = [x.strip() for x in re.split(r"[,+]", m.group("langs")) if x.strip()]
        params["languages"] = langs
        key = "easyocr_" + "_".join(langs)
        return model_spec(key=key, backend="easyocr", name=None, params=params)

    if backend == "doctr":
        m = _doctr_re.match(raw)
        det = "db_resnet50"
        rec = "crnn_vgg16_bn"
        if m and m.group("det") and m.group("rec"):
            det = str(m.group("det"))
            rec = str(m.group("rec"))
        params["detector"] = det
        params["recognizer"] = rec
        key = f"doctr_{det}_{rec}"
        return model_spec(key=key, backend="doctr", name=None, params=params)

    if backend == "paddleocr":
        m = _paddleocr_re.match(raw)
        variant = (m.group("variant") if m else None) or "en"
        v = str(variant).lower().replace("_", "-")
        params["variant"] = v
        key = "paddleocr_" + v
        return model_spec(key=key, backend="paddleocr", name=None, params=params)

    if backend == "command":
        k = raw.strip().lower()
        if k in _cloud_map:
            params["cmd"] = list(_cloud_map[k])
            return model_spec(key=k, backend="command", name=None, params=params)

    if backend == "transformers_trocr":
        key = _default_key_for(name, backend)
        return model_spec(key=key, backend="transformers_trocr", name=name, params=params)

    key = _default_key_for(name, backend)
    return model_spec(key=key, backend=backend, name=name, params=params)



def parse_models_cfg(models_cfg: Any) -> List[model_spec]:
    if not isinstance(models_cfg, list):
        raise TypeError("eval_ocr.models must be a list")

    out: List[model_spec] = []
    used_keys: set[str] = set()

    for item in models_cfg:
        if isinstance(item, str):
            spec = _parse_model_string(item)
        elif isinstance(item, dict):
            key = str(item.get("key") or item.get("id") or item.get("name") or "").strip()
            backend = _normalize_backend_token(str(item.get("backend") or "tesseract").strip())
            name = item.get("name", None)
            params = {k: v for k, v in item.items() if k not in {"key", "id", "backend", "name"}}
            if not key:
                key = _default_key_for(name, backend)
            spec = model_spec(key=key, backend=backend, name=name, params=params or {})
        else:
            raise TypeError("Each model entry must be a string or a dict")

        if spec.key in used_keys:
            raise ValueError(f"Duplicate model key: {spec.key}")
        used_keys.add(spec.key)
        out.append(spec)

    return out


class ocr_engine:
    def read_text(self, image_path: str) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return


def _load_image_bgr(image_path: str) -> Optional[np.ndarray]:
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img


def _preprocess_bgr(img: np.ndarray, preprocess: Dict[str, Any]) -> np.ndarray:
    import cv2

    gray = bool(preprocess.get("grayscale", True))
    resize = float(preprocess.get("resize", 1.0))
    threshold = str(preprocess.get("threshold", "none")).strip().lower()
    median_blur = int(preprocess.get("median_blur", 0))

    out = img

    if resize and abs(resize - 1.0) > 1e-6:
        h, w = out.shape[:2]
        nh = max(1, int(round(h * resize)))
        nw = max(1, int(round(w * resize)))
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_CUBIC)

    if gray:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    if median_blur and median_blur > 0:
        k = median_blur if median_blur % 2 == 1 else median_blur + 1
        out = cv2.medianBlur(out, k)

    if threshold in {"otsu", "binary_otsu"}:
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold in {"adaptive", "adaptive_mean"}:
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
    elif threshold in {"adaptive_gaussian"}:
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    return out


class tesseract_ocr(ocr_engine):
    def __init__(self, lang: str, psm: int, oem: int, preprocess: Optional[Dict[str, Any]] = None) -> None:
        self._lang = str(lang)
        self._psm = int(psm)
        self._oem = int(oem)
        self._preprocess = dict(preprocess or {})

    def read_text(self, image_path: str) -> str:
        import cv2
        import pytesseract

        img = _load_image_bgr(image_path)
        if img is None:
            return ""

        img = _preprocess_bgr(img, self._preprocess)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        config = f"--oem {self._oem} --psm {self._psm}"
        txt = pytesseract.image_to_string(rgb, lang=self._lang, config=config)
        return str(txt).strip()

class easyocr_ocr(ocr_engine):
    def __init__(self, languages: List[str], gpu: bool, reader_kwargs: Optional[Dict[str, Any]] = None) -> None:
        import easyocr

        self._reader = easyocr.Reader(list(languages), gpu=bool(gpu), **(reader_kwargs or {}))

    def read_text(self, image_path: str) -> str:
        out = self._reader.readtext(image_path, detail=0, paragraph=True)
        if isinstance(out, list):
            return " ".join(str(x).strip() for x in out if str(x).strip()).strip()
        return str(out).strip()

    def close(self) -> None:
        if self._reader is not None:
            del self._reader
            self._reader = None


class doctr_ocr(ocr_engine):
    def __init__(
        self,
        detector: str,
        recognizer: str,
        device: str,
        pretrained: bool,
        predictor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        from doctr.models import ocr_predictor

        dev = str(device).lower()
        self._predictor = ocr_predictor(
            det_arch=str(detector),
            reco_arch=str(recognizer),
            pretrained=bool(pretrained),
            **(predictor_kwargs or {}),
        )
        self._device = "cuda" if dev == "cuda" else "cpu"

    def read_text(self, image_path: str) -> str:
        from doctr.io import DocumentFile

        doc = DocumentFile.from_images([image_path])
        res = self._predictor(doc)

        exported = res.export()
        chunks: List[str] = []
        for page in exported.get("pages", []):
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    words = [w.get("value", "") for w in line.get("words", [])]
                    s = " ".join(x.strip() for x in words if str(x).strip()).strip()
                    if s:
                        chunks.append(s)
        return "\n".join(chunks).strip()

    def close(self) -> None:
        if self._predictor is not None:
            del self._predictor
            self._predictor = None


class paddleocr_ocr(ocr_engine):
    def __init__(self, lang: str, use_angle_cls: bool, ocr_kwargs: Optional[Dict[str, Any]] = None) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            use_angle_cls=bool(use_angle_cls),
            lang=str(lang),
            show_log=False,
            **(ocr_kwargs or {}),
        )

    def read_text(self, image_path: str) -> str:
        out = self._ocr.ocr(image_path, cls=True)
        chunks: List[str] = []
        if isinstance(out, list):
            for page in out:
                if not isinstance(page, list):
                    continue
                for item in page:
                    if not isinstance(item, list) or len(item) < 2:
                        continue
                    rec = item[1]
                    if isinstance(rec, (list, tuple)) and rec:
                        chunks.append(str(rec[0]).strip())
        return "\n".join(x for x in chunks if x).strip()

    def close(self) -> None:
        if self._ocr is not None:
            del self._ocr
            self._ocr = None


class trocr_ocr(ocr_engine):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        compute_type: str,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        preprocess: Optional[Dict[str, Any]] = None,
    ) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(str(compute_type).lower(), None)

        self._device = "cuda" if str(device).lower() == "cuda" and torch.cuda.is_available() else "cpu"
        self._processor = TrOCRProcessor.from_pretrained(model_name_or_path)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)

        self._model.eval()
        self._model.to(self._device)

        self._generate_kwargs = dict(generate_kwargs or {})
        self._preprocess = dict(preprocess or {})

    def read_text(self, image_path: str) -> str:
        import cv2
        import torch

        img = _load_image_bgr(image_path)
        if img is None:
            return ""

        img = _preprocess_bgr(img, self._preprocess)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inputs = self._processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)

        with torch.no_grad():
            out_ids = self._model.generate(pixel_values, **self._generate_kwargs)

        text = self._processor.batch_decode(out_ids, skip_special_tokens=True)
        if isinstance(text, list) and text:
            return str(text[0]).strip()
        return str(text).strip()

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None


class command_ocr(ocr_engine):
    def __init__(
        self,
        cmd: List[str],
        timeout_s: Optional[float] = None,
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._cmd = [str(x) for x in cmd]
        self._timeout_s = float(timeout_s) if timeout_s is not None else None
        self._workdir = workdir
        self._env = env

    def read_text(self, image_path: str) -> str:
        cmd = [part.replace("{image}", image_path) for part in self._cmd]
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
        return cp.stdout.strip()


def build_ocr(spec: model_spec, cfg: dict) -> ocr_engine:
    backend = _normalize_backend_token(spec.backend)
    params = dict(spec.params or {})

    preprocess = params.get("preprocess", None)

    if backend == "tesseract":
        return tesseract_ocr(
            lang=str(params.get("lang", "eng")),
            psm=int(params.get("psm", 6)),
            oem=int(params.get("oem", 3)),
            preprocess=preprocess,
        )

    if backend == "easyocr":
        return easyocr_ocr(
            languages=list(params.get("languages", ["en"])),
            gpu=str(cfg.get("device", "cpu")).lower() == "cuda",
            reader_kwargs=params.get("reader_kwargs", None),
        )

    if backend == "doctr":
        return doctr_ocr(
            detector=str(params.get("detector", "db_resnet50")),
            recognizer=str(params.get("recognizer", "crnn_vgg16_bn")),
            device=str(cfg.get("device", "cpu")),
            pretrained=bool(params.get("pretrained", True)),
            predictor_kwargs=params.get("predictor_kwargs", None),
        )

    if backend == "paddleocr":
        variant = str(params.get("variant", "en")).lower()
        lang = "en"
        use_angle_cls = True
        if variant in {"ppocrv4-en", "ppocrv5-en"}:
            lang = "en"
        elif variant in {"en"}:
            lang = "en"

        return paddleocr_ocr(
            lang=lang,
            use_angle_cls=use_angle_cls,
            ocr_kwargs=params.get("ocr_kwargs", None),
        )

    if backend == "transformers_trocr":
        if spec.name is None:
            raise ValueError(f"transformers_trocr backend requires spec.name for model key {spec.key}")
        return trocr_ocr(
            model_name_or_path=str(spec.name),
            device=str(cfg.get("device", "cpu")),
            compute_type=str(cfg.get("compute_type", "float32")),
            generate_kwargs=params.get("generate_kwargs", None),
            preprocess=preprocess,
        )

    if backend == "command":
        cmd = params.get("cmd", None)
        if not isinstance(cmd, list) or not cmd:
            raise ValueError(f"command backend requires params.cmd list for model key {spec.key}")
        return command_ocr(
            cmd=cmd,
            timeout_s=params.get("timeout_s", None),
            workdir=params.get("workdir", None),
            env=params.get("env", None),
        )

    raise ValueError(f"Unknown backend '{spec.backend}' for model key {spec.key}")

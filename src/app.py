import json
import os
import shutil
import uuid
import glob
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import urllib.request
import urllib.error


import streamlit as st

from .ingest import inspect_video, extract_audio, extract_frames
from .asr import run_asr
from .ocr import run_ocr_on_frames
from .align import align_transcript_and_ocr
from .summarise import summarise_segments
from .qa import answer_question, build_qa_index


video_exts = {".mp4", ".mov", ".mkv", ".avi"}
audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
text_exts = {".txt"}
default_video_path = "uploaded_videos/input.mp4"


def detect_input_type_from_filename(filename: str) -> Optional[str]:
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext in video_exts:
        return "Video"
    if ext in audio_exts:
        return "Audio"
    if ext in text_exts:
        return "Text"
    return None


def init_session_state() -> None:
    defaults = {
        "input_type": None,
        "uploaded_filename": None,
        "uploaded_file_path": None,
        "run_id": None,
        "video_path": None,
        "audio_path": None,
        "text_input": "",
        "output_dir": "outputs",
        "frame_interval_seconds": 3,
        "ocr_frame_stride": 2,
        "summariser_model_name": "facebook/bart-large-cnn",
        "summariser_device": 0,
        "frame_dir": None,
        "transcript_path": None,
        "ocr_output_path": None,
        "segments_path": None,
        "inspect_result": None,
        "transcript_segments": None,
        "ocr_records": None,
        "segments_merged": None,
        "summary_text": None,
        "use_default_input": False,
        "qa_index_path": None,
        "qa_last_answer": None,
        "qa_llm_model": "llama3.2:3b",
        "qa_embed_model": "nomic-embed-text",
        "qa_host": "http://localhost:11434",
        "qa_top_k": 5,
        "qa_chunk_chars": 900,
        "qa_chunk_overlap": 150,
        "qa_question": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if st.session_state["run_id"] is None:
        st.session_state["run_id"] = uuid.uuid4().hex[:8]


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    if not os.path.isfile(path):
        return
    try:
        os.remove(path)
    except OSError as exc:
        st.warning(f"Could not delete file {path}: {exc}")


def _safe_rmtree(path: Optional[str]) -> None:
    if not path:
        return
    if not os.path.isdir(path):
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        st.warning(f"Could not delete directory {path}: {exc}")


def _run_dir() -> str:
    base = st.session_state.get("output_dir") or "outputs"
    run_id = st.session_state.get("run_id") or uuid.uuid4().hex[:8]
    return os.path.join(base, run_id)


def ensure_output_paths() -> None:
    run_dir = _run_dir()
    os.makedirs(run_dir, exist_ok=True)

    frame_dir = os.path.join(run_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    st.session_state["frame_dir"] = frame_dir
    st.session_state["transcript_path"] = os.path.join(run_dir, "transcript_segments.json")
    st.session_state["ocr_output_path"] = os.path.join(run_dir, "ocr_frames.json")
    st.session_state["segments_path"] = os.path.join(run_dir, "segments.json")

    if st.session_state.get("input_type") == "Video":
        st.session_state["audio_path"] = os.path.join(run_dir, "audio.wav")


def reset_downstream_state() -> None:
    uploaded_file_path = st.session_state.get("uploaded_file_path")
    if uploaded_file_path:
        default_abs = os.path.abspath(default_video_path)
        current_abs = os.path.abspath(uploaded_file_path)
        if current_abs != default_abs:
            _safe_remove(uploaded_file_path)

    _safe_rmtree(st.session_state.get("frame_dir"))
    _safe_remove(st.session_state.get("audio_path"))
    _safe_remove(st.session_state.get("transcript_path"))
    _safe_remove(st.session_state.get("ocr_output_path"))
    _safe_remove(st.session_state.get("segments_path"))

    _safe_rmtree(_run_dir())

    st.session_state["video_path"] = None
    st.session_state["audio_path"] = None
    st.session_state["text_input"] = ""
    st.session_state["frame_dir"] = None
    st.session_state["transcript_path"] = None
    st.session_state["ocr_output_path"] = None
    st.session_state["segments_path"] = None
    st.session_state["inspect_result"] = None
    st.session_state["transcript_segments"] = None
    st.session_state["ocr_records"] = None
    st.session_state["segments_merged"] = None
    st.session_state["summary_text"] = None
    st.session_state["qa_index_path"] = None
    st.session_state["qa_last_answer"] = None
    st.session_state["uploaded_file_path"] = None
    st.session_state["uploaded_filename"] = None
    st.session_state["use_default_input"] = False
    st.session_state["run_id"] = uuid.uuid4().hex[:8]


def save_uploaded_file(uploaded_file) -> Tuple[str, Optional[str]]:
    filename = uploaded_file.name
    detected = detect_input_type_from_filename(filename)

    subdir_map = {
        "Video": "uploaded_videos",
        "Audio": "uploaded_audio",
        "Text": "uploaded_text",
    }
    subdir = subdir_map.get(detected, "uploaded_other")
    os.makedirs(subdir, exist_ok=True)

    dest = os.path.join(subdir, filename)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return dest, detected


def load_json_if_exists(file_path: Optional[str]) -> Any:
    if not file_path:
        return None
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.warning(f"Could not read JSON file at {file_path}: {exc}")
        return None


def normalize_records(records: Any) -> List[Any]:
    if records is None:
        return []
    if isinstance(records, dict):
        if "segments" in records and isinstance(records["segments"], list):
            return records["segments"]
        return list(records.values())
    if isinstance(records, list):
        return records
    try:
        return list(records)
    except TypeError:
        return [records]


def load_normalized(memory_key: str, path_key: str) -> List[Any]:
    cached = st.session_state.get(memory_key)
    if cached is not None:
        return normalize_records(cached)
    loaded = load_json_if_exists(st.session_state.get(path_key))
    return normalize_records(loaded)


def build_segments_from_text(text: str) -> List[dict]:
    t = (text or "").strip()
    if not t:
        return []
    return [{"start": 0.0, "end": 0.0, "text": t, "asr_text": t}]


def list_sample_frames(frame_dir: Optional[str], max_items: int = 6) -> List[str]:
    if not frame_dir or not os.path.isdir(frame_dir):
        return []
    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    return frames[:max_items]


def inspect_audio_file(file_path: str) -> dict:
    info = {"path": file_path, "exists": os.path.exists(file_path)}
    if os.path.exists(file_path):
        info["size_bytes"] = os.path.getsize(file_path)
        _, ext = os.path.splitext(file_path)
        info["extension"] = ext
    return info


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def render_preview_table(
    records: Sequence[Any],
    cols: Sequence[Tuple[str, Callable[[Any], Any]]],
    title: str,
    slider_key: str,
    max_preview: int = 50,
) -> None:
    import pandas as pd

    if not records:
        st.caption("No records available yet.")
        return

    total = len(records)
    col_slide, col_main = st.columns([1, 4])
    with col_slide:
        st.write(f"{title}: {total}")
        max_slider = min(max_preview, total)
        count = st.slider(
            "Number of records to preview",
            min_value=1,
            max_value=max_slider,
            value=min(15, max_slider),
            key=slider_key,
        )

    with col_main:
        rows: List[Dict[str, Any]] = []
        for r in records[:count]:
            row: Dict[str, Any] = {}
            for name, fn in cols:
                row[name] = fn(r)
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), width="stretch")


def _progress_cb_count(bar, label: str) -> Callable[[int, int, str], None]:
    def cb(i: int, total: int, message: str = "") -> None:
        pct = int(i * 100 / max(total, 1))
        pct = max(0, min(100, pct))
        bar.progress(pct, text=message or f"{label}: {i}/{total}")

    return cb


def _progress_cb_time(bar, label: str) -> Callable[[float, float, str], None]:
    def cb(current: float, total: float, message: str = "") -> None:
        pct = int(float(current) * 100 / max(float(total), 1.0))
        pct = max(0, min(100, pct))
        current_s = int(round(float(current)))
        total_s = int(round(float(total)))
        bar.progress(pct, text=message or f"{label}: {current_s}s / {total_s}s")

    return cb


def _clear_cached_file(memory_key: str, path_key: str) -> None:
    st.session_state[memory_key] = None
    _safe_remove(st.session_state.get(path_key))


def _log(status, msg: str) -> None:
    if status is not None:
        status.write(msg)


def _require(ok: bool, msg: str, strict: bool) -> bool:
    if ok:
        return True
    if strict:
        raise RuntimeError(msg)
    st.warning(msg)
    return False


def _segments_for(input_type: str, prefer_aligned: bool) -> List[Any]:
    if input_type == "Text":
        return build_segments_from_text(st.session_state.get("text_input", ""))

    if input_type == "Audio":
        return load_normalized("transcript_segments", "transcript_path")

    if input_type == "Video":
        if prefer_aligned:
            merged = st.session_state.get("segments_merged")
            if merged is not None:
                return normalize_records(merged)
            segs = load_normalized("segments_merged", "segments_path")
            if segs:
                return segs
        trs = load_normalized("transcript_segments", "transcript_path")
        return trs

    return []


def _step_plan(input_type: str) -> List[str]:
    if input_type == "Video":
        return ["Inspect", "Ingest", "ASR", "OCR", "Align", "Summarise"]
    if input_type == "Audio":
        return ["Inspect", "ASR", "Summarise"]
    if input_type == "Text":
        return ["Summarise"]
    return []


def _box_labels(input_type: str) -> List[str]:
    if input_type == "Video":
        return ["Inspect", "Ingest", "ASR", "OCR", "Align", "Summarise", "Questions and Answers"]
    if input_type == "Audio":
        return ["Inspect", "ASR", "Summarise", "Questions and Answers"]
    if input_type == "Text":
        return ["Summarise", "Questions and Answers"]
    return []


def _box_key(input_type: str, label: str) -> str:
    return f"box_expanded::{input_type}::{label}"


def set_all_boxes_expanded(input_type: str, expanded: bool) -> None:
    for label in _box_labels(input_type):
        st.session_state[_box_key(input_type, label)] = expanded


def do_inspect(input_type: str, status=None, strict: bool = True) -> bool:
    if input_type == "Video":
        video_path = st.session_state.get("video_path")
        if not _require(bool(video_path), "No video available.", strict):
            return False
        _log(status, "Running video inspection...")
        st.session_state["inspect_result"] = inspect_video(video_path)
        _log(status, "Inspection completed.")
        return True

    if input_type == "Audio":
        audio_path = st.session_state.get("audio_path")
        if not _require(bool(audio_path), "No audio available.", strict):
            return False
        _log(status, "Collecting basic audio information...")
        st.session_state["inspect_result"] = inspect_audio_file(audio_path)
        _log(status, "Inspection completed.")
        return True

    return _require(False, "Inspection is only available for video and audio inputs.", strict)


def do_ingest(input_type: str, status=None, strict: bool = True) -> bool:
    if not _require(input_type == "Video", "Ingest is only relevant for video input.", strict):
        return False

    video_path = st.session_state.get("video_path")
    if not _require(bool(video_path), "No video available.", strict):
        return False

    ensure_output_paths()

    frame_bar = st.progress(0, text="Frames: waiting")
    phase_bar = st.progress(0, text="Ingest: starting")
    frames_cb = _progress_cb_count(frame_bar, "Extracting frames")

    _log(status, "Extracting audio...")
    phase_bar.progress(20, text="Extracting audio")
    extract_audio(video_path, st.session_state["audio_path"])

    _log(status, "Extracting frames...")
    phase_bar.progress(50, text="Extracting frames")
    extract_frames(
        video_path,
        st.session_state["frame_dir"],
        interval_seconds=st.session_state["frame_interval_seconds"],
        progress_cb=frames_cb,
    )

    phase_bar.progress(100, text="Ingest: done")
    frame_bar.progress(100, text="Frames: done")
    _log(status, "Ingest completed.")
    return True


def do_asr(input_type: str, status=None, strict: bool = True) -> bool:
    audio_path = st.session_state.get("audio_path")
    if not _require(bool(audio_path), "Audio file not found.", strict):
        return False

    ensure_output_paths()
    _clear_cached_file("transcript_segments", "transcript_path")

    _log(status, "Transcribing audio...")
    bar = st.progress(0, text="ASR: waiting")
    cb = _progress_cb_time(bar, "ASR")

    transcript_segments = run_asr(
        audio_path,
        st.session_state["transcript_path"],
        progress_cb=cb,
    )
    st.session_state["transcript_segments"] = transcript_segments
    _log(status, "ASR completed.")
    return True


def do_ocr(input_type: str, status=None, strict: bool = True) -> bool:
    if not _require(input_type == "Video", "OCR is only relevant for video input.", strict):
        return False

    frame_dir = st.session_state.get("frame_dir")
    if not _require(bool(frame_dir), "Frames directory not found.", strict):
        return False

    ensure_output_paths()
    _clear_cached_file("ocr_records", "ocr_output_path")

    _log(status, "Processing frames for OCR...")
    bar = st.progress(0, text="OCR: waiting")
    cb = _progress_cb_count(bar, "OCR")

    ocr_records = run_ocr_on_frames(
        frame_dir=frame_dir,
        ocr_output_path=st.session_state["ocr_output_path"],
        frame_interval_seconds=st.session_state["frame_interval_seconds"],
        ocr_frame_stride=st.session_state["ocr_frame_stride"],
        progress_cb=cb,
    )
    st.session_state["ocr_records"] = ocr_records
    _log(status, "OCR completed.")
    return True


def do_align(input_type: str, status=None, strict: bool = True) -> bool:
    if not _require(input_type == "Video", "Alignment is only relevant for video input.", strict):
        return False

    ensure_output_paths()

    transcript_segments = load_normalized("transcript_segments", "transcript_path")
    ocr_records = load_normalized("ocr_records", "ocr_output_path")

    if not _require(bool(transcript_segments), "Missing transcript segments.", strict):
        return False
    if not _require(bool(ocr_records), "Missing OCR records.", strict):
        return False

    _log(status, "Aligning ASR and OCR segments...")
    st.session_state["segments_merged"] = align_transcript_and_ocr(
        transcript_segments=transcript_segments,
        ocr_records=ocr_records,
        segments_path=st.session_state["segments_path"],
    )
    _log(status, "Alignment completed.")
    return True


def do_summarise(input_type: str, status=None, strict: bool = True) -> bool:
    ensure_output_paths()

    prefer_aligned = input_type == "Video"
    segments = _segments_for(input_type, prefer_aligned=prefer_aligned)

    if input_type == "Video" and not segments:
        _log(status, "No aligned segments found. Falling back to ASR transcript.")
        segments = _segments_for("Video", prefer_aligned=False)

    if not _require(bool(segments), "No segments available for summarisation.", strict):
        return False

    _log(status, f"Summarising {len(segments)} segments with model {st.session_state['summariser_model_name']}.")
    bar = st.progress(0, text="Summarise: waiting")
    cb = _progress_cb_count(bar, "Summarise")

    summary = summarise_segments(
        segments,
        model_name=st.session_state["summariser_model_name"],
        device=st.session_state["summariser_device"],
        max_chunk_chars=3000,
        max_length=500,
        min_length=40,
        progress_cb=cb,
    )
    st.session_state["summary_text"] = summary
    _log(status, "Summarisation completed.")
    return True


_do_fns: Dict[str, Callable[[str, Any, bool], bool]] = {
    "Inspect": do_inspect,
    "Ingest": do_ingest,
    "ASR": do_asr,
    "OCR": do_ocr,
    "Align": do_align,
    "Summarise": do_summarise,
}

_step_titles = {
    "Inspect": "Inspecting",
    "Ingest": "Ingesting",
    "ASR": "Running ASR",
    "OCR": "Running OCR",
    "Align": "Aligning",
    "Summarise": "Summarising",
}


def run_step_ui(step: str, input_type: str) -> None:
    title = _step_titles.get(step, step)
    do_fn = _do_fns[step]

    with st.status(title, expanded=True) as status:
        try:
            ran = do_fn(input_type, status=status, strict=False)
            if ran:
                status.update(label=f"{step} completed.", state="complete", expanded=False)
                st.success(f"{step} completed.")
            else:
                status.update(label=f"{step} skipped.", state="complete", expanded=False)
        except Exception as exc:
            status.update(label=f"{step} failed.", state="error")
            st.exception(exc)


def run_all_pipeline(input_type: str) -> None:
    ensure_output_paths()
    steps = _step_plan(input_type)
    if not steps:
        st.warning("Unsupported input type.")
        return

    total = len(steps)
    overall = st.progress(0, text="Pipeline: starting")

    def set_overall(i: int, label: str) -> None:
        pct = int(i * 100 / max(total, 1))
        pct = max(0, min(100, pct))
        overall.progress(pct, text=f"Pipeline: {label}")

    with st.status("Running all steps", expanded=True) as status:
        try:
            for idx, name in enumerate(steps, start=1):
                set_overall(idx - 1, f"{name} (starting)")
                status.markdown(f"### **Step {idx}/{total}: {name}**")
                _do_fns[name](input_type, status=status, strict=True)
                set_overall(idx, f"{name} (done)")

            overall.progress(100, text="Pipeline: done")
            status.write("All steps completed.")
            set_all_boxes_expanded(input_type, True)
            status.update(label="Run all completed.", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Run all failed", state="error")
            st.error("Error while running the pipeline.")
            st.exception(exc)
            return

    st.success("Run all completed.")


def render_run_all_controls(input_type: str) -> None:
    st.subheader("Run all")
    label_map = {
        "Video": "Run all (Inspect → Ingest → ASR → OCR → Align → Summarise)",
        "Audio": "Run all (Inspect → ASR → Summarise)",
        "Text": "Run all (Summarise)",
    }
    label = label_map.get(input_type, "Run all steps")
    if st.button(label, type="primary"):
        run_all_pipeline(input_type)


def render_advanced_section(input_type: str, max_cols: int = 6) -> None:
    with st.expander("Advanced", expanded=False):
        steps = _step_plan(input_type)
        cols_count = max(1, min(max_cols, len(steps)))
        cols = st.columns(cols_count)
        for i, name in enumerate(steps):
            with cols[i % len(cols)]:
                if st.button(name, key=f"adv_{name.lower()}"):
                    run_step_ui(name, input_type)



@st.dialog("Clean generated files")
def cleanup_dialog() -> None:
    st.write("This will delete the uploaded file and generated artefacts for the current session.")
    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("Cancel"):
            st.rerun()
    with col_confirm:
        if st.button("Confirm cleanup"):
            reset_downstream_state()
            st.success("Temporary files and in-memory state removed.")
            st.rerun()


def render_sidebar() -> None:
    st.sidebar.header("Input")

    default_exists = os.path.isfile(default_video_path)
    if default_exists:
        use_default = st.sidebar.checkbox(
            "Use default video (uploaded_videos/input.mp4)",
            value=st.session_state.get("use_default_input", False),
        )
        if use_default and not st.session_state.get("use_default_input", False):
            reset_downstream_state()
            st.session_state["use_default_input"] = True
            st.session_state["input_type"] = "Video"
            st.session_state["video_path"] = default_video_path

        if not use_default and st.session_state.get("use_default_input", False):
            reset_downstream_state()

    uploaded_file = st.sidebar.file_uploader(
        "Upload a video, audio, or text file",
        type=list(video_exts | audio_exts | text_exts),
        disabled=st.session_state.get("use_default_input", False),
    )

    if uploaded_file is not None:
        filename = uploaded_file.name
        if st.session_state.get("uploaded_filename") != filename:
            reset_downstream_state()
            st.session_state["uploaded_filename"] = filename

        file_path, detected = save_uploaded_file(uploaded_file)
        st.session_state["uploaded_file_path"] = file_path
        st.session_state["input_type"] = detected
        st.session_state["use_default_input"] = False

        if detected == "Video":
            st.session_state["video_path"] = file_path
            st.sidebar.success(f"Detected video file: {file_path}")
        elif detected == "Audio":
            st.session_state["audio_path"] = file_path
            st.sidebar.success(f"Detected audio file: {file_path}")
        elif detected == "Text":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    st.session_state["text_input"] = f.read()
            except Exception:
                st.session_state["text_input"] = ""
            st.sidebar.success("Detected text file and loaded its content.")
        else:
            st.sidebar.warning("File extension not recognised as video, audio, or text.")
            st.session_state["input_type"] = None

    st.sidebar.markdown("---")
    st.sidebar.header("Configuration")

    st.session_state["output_dir"] = st.sidebar.text_input(
        "Base output directory",
        value=st.session_state["output_dir"],
        help="Per-session artefacts are written under this directory.",
    )

    with st.sidebar.expander("Summariser settings", expanded=False):
        st.session_state["summariser_model_name"] = st.text_input(
            "Summariser model",
            value=st.session_state["summariser_model_name"],
            help="Hugging Face model name used for summarisation.",
        )
        st.session_state["summariser_device"] = st.number_input(
            "Summariser device (GPU index, -1 for CPU)",
            min_value=-1,
            max_value=8,
            value=st.session_state["summariser_device"],
            step=1,
            help="Set -1 to run on CPU.",
        )

    input_type = st.session_state.get("input_type")
    if input_type == "Video":
        with st.sidebar.expander("Video processing", expanded=False):
            st.session_state["frame_interval_seconds"] = st.number_input(
                "Frame interval (seconds)",
                min_value=1,
                max_value=30,
                value=st.session_state["frame_interval_seconds"],
                step=1,
                help="Time between extracted frames.",
            )
            st.session_state["ocr_frame_stride"] = st.number_input(
                "OCR frame stride",
                min_value=1,
                max_value=10,
                value=st.session_state["ocr_frame_stride"],
                step=1,
                help="Process every Nth frame during OCR.",
            )

    st.sidebar.markdown("---")
    if st.sidebar.button("Clean generated files for this session"):
        cleanup_dialog()


def render_inspect_box(input_type: str) -> None:
    col_main, col_preview = st.columns([1.3, 1])

    with col_main:
        if input_type in {"Video", "Audio"}:
            if st.session_state.get("inspect_result") is not None:
                st.subheader(f"{input_type} metadata")
                st.json(st.session_state["inspect_result"])
            else:
                st.caption("No inspection results available yet.")
        else:
            st.info("Inspection is only available for video and audio inputs.")

    with col_preview:
        st.subheader("Input preview")
        if input_type == "Video" and st.session_state.get("video_path"):
            st.video(st.session_state["video_path"])
        elif input_type == "Audio" and st.session_state.get("audio_path"):
            st.audio(st.session_state["audio_path"])
        elif input_type == "Text":
            t = st.session_state.get("text_input") or ""
            if t:
                st.text_area("Text preview", value=t[:4000], height=300)
            else:
                st.caption("No text loaded yet.")


def render_ingest_box() -> None:
    st.subheader("Extract audio and frames")

    if st.session_state.get("input_type") != "Video":
        st.warning("Ingest is only relevant for video input.")
        return

    if st.session_state.get("video_path") is None:
        st.warning("No video available. Upload a video in the sidebar.")
        return

    st.subheader("Outputs")
    st.markdown(
        f"- Audio path: `{st.session_state.get('audio_path')}`\n"
        f"- Frames directory: `{st.session_state.get('frame_dir')}`"
    )

    frames = list_sample_frames(st.session_state.get("frame_dir"))
    if frames:
        st.subheader("Sample frames")
        cols = st.columns(min(len(frames), 3))
        for idx, fp in enumerate(frames):
            with cols[idx % len(cols)]:
                st.image(fp, caption=os.path.basename(fp), width="stretch")
    else:
        st.caption("No frames found yet in the frames directory.")


def render_asr_box(input_type: str) -> None:
    st.subheader("Transcript")

    audio_path = st.session_state.get("audio_path")
    if audio_path is None:
        if input_type == "Video":
            st.warning("Audio file not found. Run Ingest in Advanced.")
        else:
            st.warning("Audio file not found.")
    else:
        st.caption(f"Audio path: `{audio_path}`")

    segments = load_normalized("transcript_segments", "transcript_path")

    if segments:
        render_preview_table(
            segments,
            cols=[
                ("start", lambda s: _get(s, "start")),
                ("end", lambda s: _get(s, "end")),
                ("text", lambda s: _get(s, "text", _get(s, "asr_text", ""))),
            ],
            title="Total records",
            slider_key="asr_preview",
        )
        transcript_path = st.session_state.get("transcript_path")
        if transcript_path and os.path.isfile(transcript_path):
            with open(transcript_path, "rb") as f:
                st.download_button(
                    "Download transcript JSON",
                    data=f,
                    file_name="transcript_segments.json",
                    mime="application/json",
                )
    else:
        st.caption("No transcript segments available yet.")


def render_ocr_box() -> None:
    if st.session_state.get("input_type") != "Video":
        st.warning("OCR is only relevant for video input.")
        return

    if st.session_state.get("frame_dir") is None:
        st.warning("Frames directory not found. Run Ingest in Advanced.")
        return

    st.subheader("OCR")

    records = load_normalized("ocr_records", "ocr_output_path")

    if records:
        render_preview_table(
            records,
            cols=[
                ("time", lambda r: _get(r, "time")),
                ("frame", lambda r: _get(r, "frame")),
                ("text", lambda r: _get(r, "text", "")),
            ],
            title="Total OCR records",
            slider_key="ocr_preview",
        )
        ocr_output_path = st.session_state.get("ocr_output_path")
        if ocr_output_path and os.path.isfile(ocr_output_path):
            with open(ocr_output_path, "rb") as f:
                st.download_button(
                    "Download OCR JSON",
                    data=f,
                    file_name="ocr_frames.json",
                    mime="application/json",
                )
    else:
        st.caption("No OCR records available yet.")


def render_align_box() -> None:
    if st.session_state.get("input_type") != "Video":
        st.warning("Alignment is only relevant for video input.")
        return

    st.subheader("Alignment")

    segments = load_normalized("segments_merged", "segments_path")

    if segments:
        render_preview_table(
            segments,
            cols=[
                ("start", lambda s: _get(s, "start")),
                ("end", lambda s: _get(s, "end")),
                ("asr_text", lambda s: _get(s, "asr_text", _get(s, "text", ""))),
                ("ocr_text", lambda s: _get(s, "ocr_text", "")),
            ],
            title="Total aligned segments",
            slider_key="aligned_preview",
        )
        segments_path = st.session_state.get("segments_path")
        if segments_path and os.path.isfile(segments_path):
            with open(segments_path, "rb") as f:
                st.download_button(
                    "Download aligned segments JSON",
                    data=f,
                    file_name="segments.json",
                    mime="application/json",
                )
    else:
        st.caption("No aligned segments available yet.")



def render_summary_box(input_type: str) -> None:
    if input_type == "Text":
        st.subheader("Input text")
        st.session_state["text_input"] = st.text_area(
            "Paste or edit the text to summarise",
            value=st.session_state.get("text_input", ""),
            height=300,
        )

    st.subheader("Global summary")
    summary_text = st.session_state.get("summary_text")
    if summary_text:
        st.text_area("Summary", summary_text, height=300)
        st.download_button(
            "Download summary as text",
            data=summary_text.encode("utf-8"),
            file_name="summary.txt",
            mime="text/plain",
        )
    else:
        st.caption("No summary available yet.")


def _normalize_host(host: str) -> str:
    h = (host or "").strip()
    return h.rstrip("/")


@st.cache_data(ttl=10, show_spinner=False)
def fetch_ollama_model_names(host: str) -> List[str]:
    h = _normalize_host(host)
    if not h:
        return []

    url = f"{h}/api/tags"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
        data = json.loads(payload)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return []

    models = data.get("models", [])
    names: List[str] = []
    for m in models:
        if isinstance(m, dict):
            name = m.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())

    return sorted(set(names))


def _partition_ollama_models(names: List[str]) -> Tuple[List[str], List[str]]:
    embed = [n for n in names if "embed" in n.lower()]
    llm = [n for n in names if n not in embed]

    if not llm:
        llm = list(names)
    if not embed:
        embed = list(names)

    return llm, embed


def _ensure_in_options(options: List[str], current: str) -> List[str]:
    cur = (current or "").strip()
    if not cur:
        return options
    if cur in options:
        return options
    return [cur] + options


def render_qa_box(input_type: str) -> None:
    ensure_output_paths()

    index_path = os.path.join(_run_dir(), "qa_index.json")
    st.session_state["qa_index_path"] = index_path

    st.subheader("Retrieval-augmented Q/A")

    col_cfg, col_run = st.columns([1.2, 1])
    with col_cfg:
        host = st.text_input("Ollama host", value=st.session_state.get("qa_host", "http://localhost:11434"))

        col_models_left, col_models_right = st.columns([1, 1])
        with col_models_left:
            if st.button("Refresh model list"):
                fetch_ollama_model_names.clear()
                st.rerun()

        names_all = fetch_ollama_model_names(host)
        llm_options, embed_options = _partition_ollama_models(names_all)

        current_llm = st.session_state.get("qa_llm_model", "llama3.1")
        current_embed = st.session_state.get("qa_embed_model", "nomic-embed-text")

        if names_all:
            llm_options = _ensure_in_options(llm_options, current_llm)
            embed_options = _ensure_in_options(embed_options, current_embed)

            llm_model = st.selectbox(
                "Ollama LLM model",
                options=llm_options,
                index=llm_options.index(current_llm) if current_llm in llm_options else 0,
            )
            embed_model = st.selectbox(
                "Ollama embedding model",
                options=embed_options,
                index=embed_options.index(current_embed) if current_embed in embed_options else 0,
            )
        else:
            st.caption("No Ollama models found via /api/tags. Falling back to manual entry.")
            llm_model = st.text_input("Ollama LLM model", value=current_llm)
            embed_model = st.text_input("Ollama embedding model", value=current_embed)


        top_k = st.number_input(
            "Top-k context chunks",
            min_value=1,
            max_value=20,
            value=int(st.session_state.get("qa_top_k", 5)),
            step=1,
        )
        chunk_chars = st.number_input(
            "Chunk size (chars)",
            min_value=200,
            max_value=5000,
            value=int(st.session_state.get("qa_chunk_chars", 900)),
            step=50,
        )
        chunk_overlap = st.number_input(
            "Chunk overlap (chars)",
            min_value=0,
            max_value=2000,
            value=int(st.session_state.get("qa_chunk_overlap", 150)),
            step=25,
        )

        st.session_state["qa_llm_model"] = llm_model
        st.session_state["qa_embed_model"] = embed_model
        st.session_state["qa_host"] = host
        st.session_state["qa_top_k"] = int(top_k)
        st.session_state["qa_chunk_chars"] = int(chunk_chars)
        st.session_state["qa_chunk_overlap"] = int(chunk_overlap)

    with col_run:
        st.caption(f"Index file: `{index_path}`")
        build_bar = st.progress(0, text="RAG index: idle")
        build_cb = _progress_cb_count(build_bar, "RAG index")

        if st.button("Build or rebuild index"):
            segments = _segments_for(input_type, prefer_aligned=(input_type == "Video"))
            if not segments:
                st.warning("No content available yet. Run earlier phases first.")
            else:
                with st.status("Building RAG index", expanded=True) as status:
                    try:
                        status.write(f"Preparing {len(segments)} segments...")
                        build_qa_index(
                            segments=segments,
                            index_path=index_path,
                            embed_model=embed_model,
                            chunk_chars=int(chunk_chars),
                            chunk_overlap=int(chunk_overlap),
                            text_mode="aligned" if input_type == "Video" else "plain",
                            force=True,
                            host=host,
                            progress_cb=build_cb,
                        )
                        build_bar.progress(100, text="RAG index: done")
                        st.success("Index built.")
                    except Exception as exc:
                        status.update(label="Index build failed", state="error")
                        st.exception(exc)

    st.markdown("---")
    question = st.text_input("Question", value=st.session_state.get("qa_question", ""))
    st.session_state["qa_question"] = question

    ask_bar = st.progress(0, text="Q/A: idle")
    ask_cb = _progress_cb_time(ask_bar, "Q/A")

    if st.button("Ask"):
        q = (question or "").strip()
        if not q:
            st.warning("Enter a question first.")
        elif not os.path.exists(index_path):
            st.warning("Index not found. Build the index first.")
        else:
            with st.status("Answering", expanded=True) as status:
                try:
                    status.write("Running retrieval and generation...")
                    result = answer_question(
                        question=q,
                        index_path=index_path,
                        llm_model=llm_model,
                        embed_model=embed_model,
                        top_k=int(top_k),
                        host=host,
                        progress_cb=ask_cb,
                    )
                    st.session_state["qa_last_answer"] = result
                    ask_bar.progress(100, text="Q/A: done")
                    st.success("Answer ready.")
                except Exception as exc:
                    status.update(label="Q/A failed", state="error")
                    st.exception(exc)

    result = st.session_state.get("qa_last_answer")
    if result:
        st.subheader("Answer")
        st.write(result.get("answer", ""))

        sources = result.get("sources", []) or []
        if sources:
            st.subheader("Sources")
            for src in sources:
                meta = src.get("meta", {}) or {}
                start = meta.get("start")
                end = meta.get("end")
                time_part = ""
                if start is not None and end is not None:
                    time_part = f"{start:.2f}-{end:.2f}s"
                title = f"[{src.get('source_id','')}] {time_part}".strip()
                with st.expander(title, expanded=False):
                    st.write(src.get("text", ""))
                    st.json(meta)


def render_boxes(input_type: str) -> None:
    boxes: Dict[str, List[Tuple[str, Callable[[], None]]]] = {
        "Video": [
            ("Inspect", lambda: render_inspect_box(input_type)),
            ("Ingest", render_ingest_box),
            ("ASR", lambda: render_asr_box(input_type)),
            ("OCR", render_ocr_box),
            ("Align", render_align_box),
            ("Summarise", lambda: render_summary_box(input_type)),
            ("Questions and Answers", lambda: render_qa_box(input_type)),
        ],
        "Audio": [
            ("Inspect", lambda: render_inspect_box(input_type)),
            ("ASR", lambda: render_asr_box(input_type)),
            ("Summarise", lambda: render_summary_box(input_type)),
            ("Questions and Answers", lambda: render_qa_box(input_type)),
        ],
        "Text": [
            ("Summarise", lambda: render_summary_box(input_type)),
            ("Questions and Answers", lambda: render_qa_box(input_type)),
        ],
    }

    default_expanded = {""}

    for label, fn in boxes.get(input_type, []):
        key = _box_key(input_type, label)
        expanded = st.session_state.get(key, label in default_expanded)
        with st.expander(label, expanded=expanded):
            fn()


def run() -> None:
    st.set_page_config(
        page_title="NLP Transcript Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    render_sidebar()

    st.title("NLP Transcript Intelligence")
    st.caption("Interactive Streamlit interface for video, audio, and text understanding.")

    input_type = st.session_state.get("input_type")
    if input_type is None:
        st.info("Upload a video, audio, or text file in the sidebar to begin.")
        return

    tab_run, tab_adv = st.tabs(["Run all at once", "Run Step by Step"])

    with tab_run:
        render_run_all_controls(input_type)

    with tab_adv:
        render_advanced_section(input_type)

    render_boxes(input_type)

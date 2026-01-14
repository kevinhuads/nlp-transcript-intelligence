import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

import streamlit as st

from .ingest import inspect_video, extract_audio, extract_frames
from .asr import run_asr
from .ocr import run_ocr_on_frames
from .align import align_transcript_and_ocr
from .summarise import summarise_segments


# ---------- Constants ----------


video_exts = {".mp4", ".mov", ".mkv", ".avi"}
audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
text_exts = {".txt"}
default_video_path = "uploaded_videos/input.mp4"


# ---------- Session state helpers ----------


def detect_input_type_from_filename(filename: str) -> Optional[str]:
    ext = Path(filename).suffix.lower()
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
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if st.session_state["run_id"] is None:
        st.session_state["run_id"] = uuid.uuid4().hex[:8]


def reset_downstream_state() -> None:
    uploaded_file_path = st.session_state.get("uploaded_file_path")
    if uploaded_file_path and os.path.exists(uploaded_file_path):
        default_abs = str(Path(default_video_path).resolve())
        current_abs = str(Path(uploaded_file_path).resolve())
        if current_abs != default_abs:
            try:
                os.remove(uploaded_file_path)
            except OSError as exc:
                st.warning(f"Could not delete uploaded file at {uploaded_file_path}: {exc}")

    frame_dir = st.session_state.get("frame_dir")
    if frame_dir and os.path.isdir(frame_dir):
        try:
            shutil.rmtree(frame_dir)
        except OSError as exc:
            st.warning(f"Could not delete frame directory {frame_dir}: {exc}")

    for key in ("audio_path", "transcript_path", "ocr_output_path", "segments_path"):
        file_path = st.session_state.get(key)
        if file_path and os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError as exc:
                st.warning(f"Could not delete file {file_path}: {exc}")

    output_dir = st.session_state.get("output_dir")
    run_id = st.session_state.get("run_id")
    if output_dir and run_id:
        run_dir = Path(output_dir) / run_id
        if run_dir.exists():
            try:
                shutil.rmtree(run_dir)
            except OSError as exc:
                st.warning(f"Could not delete run directory {run_dir}: {exc}")

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
    st.session_state["uploaded_file_path"] = None
    st.session_state["uploaded_filename"] = None
    st.session_state["use_default_input"] = False
    st.session_state["run_id"] = uuid.uuid4().hex[:8]


def save_uploaded_file(uploaded_file) -> Tuple[str, Optional[str]]:
    filename = uploaded_file.name
    detected_type = detect_input_type_from_filename(filename)

    if detected_type == "Video":
        subdir = "uploaded_videos"
    elif detected_type == "Audio":
        subdir = "uploaded_audio"
    elif detected_type == "Text":
        subdir = "uploaded_text"
    else:
        subdir = "uploaded_other"

    tmp_dir = Path(subdir)
    tmp_dir.mkdir(exist_ok=True)
    dest = tmp_dir / filename
    with dest.open("wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(dest), detected_type


def ensure_output_paths() -> None:
    base = Path(st.session_state["output_dir"])
    run_id = st.session_state.get("run_id")
    if run_id:
        base = base / run_id

    base.mkdir(parents=True, exist_ok=True)

    frame_dir = base / "frames"
    transcript_path = base / "transcript_segments.json"
    ocr_output_path = base / "ocr_frames.json"
    segments_path = base / "segments.json"

    frame_dir.mkdir(parents=True, exist_ok=True)

    st.session_state["frame_dir"] = str(frame_dir)
    st.session_state["transcript_path"] = str(transcript_path)
    st.session_state["ocr_output_path"] = str(ocr_output_path)
    st.session_state["segments_path"] = str(segments_path)

    if st.session_state["input_type"] == "Video":
        audio_path = base / "audio.wav"
        st.session_state["audio_path"] = str(audio_path)


# ---------- Data helpers ----------


def load_json_if_exists(file_path: Optional[str]) -> Any:
    if file_path is None:
        return None
    p = Path(file_path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.warning(f"Could not read JSON file at {file_path}: {exc}")
        return None


def list_sample_frames(frame_dir: Optional[str], max_items: int = 6) -> List[Path]:
    if frame_dir is None:
        return []
    d = Path(frame_dir)
    if not d.exists():
        return []
    frames = sorted(d.glob("frame_*.jpg"))
    return frames[:max_items]


def inspect_audio_file(file_path: str) -> dict:
    info = {
        "path": file_path,
        "exists": os.path.exists(file_path),
    }
    if os.path.exists(file_path):
        info["size_bytes"] = os.path.getsize(file_path)
        info["extension"] = Path(file_path).suffix
    return info


def build_segments_from_text(text: str) -> List[dict]:
    text = text.strip()
    if not text:
        return []
    return [
        {
            "start": 0.0,
            "end": 0.0,
            "text": text,
            "asr_text": text,
        }
    ]


def get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_segments(segments: Any) -> List[Any]:
    if segments is None:
        return []
    if isinstance(segments, dict):
        if "segments" in segments and isinstance(segments["segments"], list):
            return segments["segments"]
        return list(segments.values())
    if isinstance(segments, list):
        return segments
    try:
        return list(segments)
    except TypeError:
        return [segments]


def load_and_normalize(memory_key: str, path_key: str) -> List[Any]:
    cached = st.session_state.get(memory_key)
    if cached is not None:
        return normalize_segments(cached)
    file_path = st.session_state.get(path_key)
    loaded = load_json_if_exists(file_path) if file_path else None
    return normalize_segments(loaded)


def _file_has_content(path_key: str) -> bool:
    file_path = st.session_state.get(path_key)
    if not file_path:
        return False
    if not os.path.isfile(file_path):
        return False
    try:
        return os.path.getsize(file_path) > 0
    except OSError:
        return False


def _frames_extracted(frame_dir_key: str = "frame_dir") -> bool:
    frame_dir = st.session_state.get(frame_dir_key)
    if not frame_dir or not os.path.isdir(frame_dir):
        return False
    d = Path(frame_dir)
    for _ in d.glob("frame_*.jpg"):
        return True
    return False


def display_segments_preview(
    segments: List[Any],
    max_preview: int = 20,
    slider_key: str = "segments_preview",
) -> None:
    import pandas as pd

    if not segments:
        st.caption("No records available yet.")
        return

    total = len(segments)
    st.write(f"Total records: {total}")
    max_slider = min(max_preview, total)
    count = st.slider(
        "Number of records to preview",
        min_value=1,
        max_value=max_slider,
        value=min(5, max_slider),
        key=slider_key,
    )

    rows = []
    for seg in segments[:count]:
        rows.append(
            {
                "start": get_attr_or_key(seg, "start", None),
                "end": get_attr_or_key(seg, "end", None),
                "text": get_attr_or_key(seg, "text", get_attr_or_key(seg, "asr_text", "")),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")


def display_ocr_preview(
    records: List[Any],
    max_preview: int = 20,
    slider_key: str = "ocr_preview",
) -> None:
    import pandas as pd

    if not records:
        st.caption("No OCR records available yet.")
        return

    total = len(records)
    st.write(f"Total OCR records: {total}")
    max_slider = min(max_preview, total)
    count = st.slider(
        "Number of OCR records to preview",
        min_value=1,
        max_value=max_slider,
        value=min(5, max_slider),
        key=slider_key,
    )

    rows = []
    for rec in records[:count]:
        rows.append(
            {
                "time": get_attr_or_key(rec, "time", None),
                "frame": get_attr_or_key(rec, "frame", None),
                "text": get_attr_or_key(rec, "text", ""),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")


def display_aligned_preview(
    segments: List[Any],
    max_preview: int = 20,
    slider_key: str = "aligned_preview",
) -> None:
    import pandas as pd

    if not segments:
        st.caption("No aligned segments available yet.")
        return

    total = len(segments)
    st.write(f"Total aligned segments: {total}")
    max_slider = min(max_preview, total)
    count = st.slider(
        "Number of aligned segments to preview",
        min_value=1,
        max_value=max_slider,
        value=min(5, max_slider),
        key=slider_key,
    )

    rows = []
    for seg in segments[:count]:
        rows.append(
            {
                "start": get_attr_or_key(seg, "start", None),
                "end": get_attr_or_key(seg, "end", None),
                "asr_text": get_attr_or_key(seg, "asr_text", get_attr_or_key(seg, "text", "")),
                "ocr_text": get_attr_or_key(seg, "ocr_text", ""),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")


# ---------- UI helpers ----------


def render_pipeline_status(input_type: Optional[str]) -> None:
    if input_type is None:
        return

    st.subheader("Pipeline status")

    if input_type == "Video":
        steps = [
            ("Inspect", bool(st.session_state.get("inspect_result"))),
            ("Ingest", _file_has_content("audio_path") and _frames_extracted("frame_dir")),
            ("ASR", bool(st.session_state.get("transcript_segments")) or _file_has_content("transcript_path")),
            ("OCR", bool(st.session_state.get("ocr_records")) or _file_has_content("ocr_output_path")),
            ("Align", bool(st.session_state.get("segments_merged")) or _file_has_content("segments_path")),
            ("Summarise", bool(st.session_state.get("summary_text"))),
        ]
    elif input_type == "Audio":
        steps = [
            ("Inspect", bool(st.session_state.get("inspect_result"))),
            ("ASR", bool(st.session_state.get("transcript_segments")) or _file_has_content("transcript_path")),
            ("Summarise", bool(st.session_state.get("summary_text"))),
        ]
    elif input_type == "Text":
        steps = [
            ("Summarise", bool(st.session_state.get("summary_text"))),
        ]
    else:
        return

    cols = st.columns(len(steps))
    for col, (name, done) in zip(cols, steps):
        with col:
            st.metric(label=name, value="✅" if done else "⚪")


@st.dialog("Clean generated files")
def cleanup_dialog() -> None:
    st.write(
        "This will delete the uploaded file and generated artefacts "
        "for the current session."
    )
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

    default_exists = Path(default_video_path).exists()
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

        file_path, detected_type = save_uploaded_file(uploaded_file)
        st.session_state["uploaded_file_path"] = file_path
        st.session_state["input_type"] = detected_type
        st.session_state["use_default_input"] = False

        if detected_type == "Video":
            st.session_state["video_path"] = file_path
            st.sidebar.success(f"Detected video file: {file_path}")
        elif detected_type == "Audio":
            st.session_state["audio_path"] = file_path
            st.sidebar.success(f"Detected audio file: {file_path}")
        elif detected_type == "Text":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_value = f.read()
            except Exception:
                text_value = ""
            st.session_state["text_input"] = text_value
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


# ---------- Phase renderers ----------


def render_inspect_tab(input_type: str) -> None:
    col_main, col_preview = st.columns([1.3, 1])

    with col_main:
        if input_type == "Video":
            if st.session_state["video_path"] is None:
                st.warning("No video available. Upload a video in the sidebar.")
            else:
                if st.button("Run video inspection"):
                    with st.status("Inspecting video", expanded=True) as status:
                        try:
                            status.write("Running video inspection...")
                            result = inspect_video(st.session_state["video_path"])
                            st.session_state["inspect_result"] = result
                            status.write("Inspection completed.")
                        except Exception as exc:
                            status.update(label="Video inspection failed", state="error")
                            st.error("Error during video inspection.")
                            st.exception(exc)

                if st.session_state["inspect_result"] is not None:
                    st.subheader("Video metadata")
                    st.json(st.session_state["inspect_result"])
                else:
                    st.caption("Run the inspection to display metadata here.")

        elif input_type == "Audio":
            if st.session_state["audio_path"] is None:
                st.warning("No audio available. Upload an audio file in the sidebar.")
            else:
                if st.button("Show audio information"):
                    with st.status("Inspecting audio", expanded=True) as status:
                        try:
                            status.write("Collecting basic audio information...")
                            result = inspect_audio_file(st.session_state["audio_path"])
                            st.session_state["inspect_result"] = result
                            status.write("Inspection completed.")
                        except Exception as exc:
                            status.update(label="Audio inspection failed", state="error")
                            st.error("Error during audio inspection.")
                            st.exception(exc)

                if st.session_state["inspect_result"] is not None:
                    st.subheader("Audio metadata")
                    st.json(st.session_state["inspect_result"])
                else:
                    st.caption("Click on the button above to show basic audio information.")

        else:
            st.info("Inspection is only available for video and audio inputs.")

    with col_preview:
        st.subheader("Input preview")
        if input_type == "Video" and st.session_state.get("video_path"):
            st.video(st.session_state["video_path"])
        elif input_type == "Audio" and st.session_state.get("audio_path"):
            st.audio(st.session_state["audio_path"])
        elif input_type == "Text":
            if st.session_state.get("text_input"):
                st.text_area(
                    "Text preview",
                    value=st.session_state["text_input"][:4000],
                    height=300,
                )
            else:
                st.caption("No text loaded yet.")


def render_ingest_tab() -> None:
    st.subheader("Extract audio and frames")

    if st.session_state.get("input_type") != "Video":
        st.warning("Ingest is only relevant for video input.")
        return

    if st.session_state.get("video_path") is None:
        st.warning("No video available. Upload a video in the sidebar.")
        return

    col_main, col_outputs = st.columns([1, 1.3])

    with col_main:
        if st.button("Run ingest"):
            ensure_output_paths()

            frame_bar = st.progress(0, text="Frames: waiting")
            phase_bar = st.progress(0, text="Ingest: starting")

            def frames_progress(i: int, total: int, message: str = "") -> None:
                pct = int(i * 100 / max(total, 1))
                frame_bar.progress(pct, text=message or f"Extracting frames: {i}/{total}")

            with st.status("Ingesting video", expanded=True) as status:
                try:
                    phase_bar.progress(20, text="Extracting audio")
                    extract_audio(st.session_state["video_path"], st.session_state["audio_path"])

                    phase_bar.progress(50, text="Extracting frames")
                    extract_frames(
                        st.session_state["video_path"],
                        st.session_state["frame_dir"],
                        interval_seconds=st.session_state["frame_interval_seconds"],
                        progress_cb=frames_progress,
                    )

                    phase_bar.progress(100, text="Ingest: done")
                    frame_bar.progress(100, text="Frames: done")
                    st.success("Ingestion completed.")
                except Exception as exc:
                    status.update(label="Ingestion failed", state="error")
                    st.exception(exc)


    with col_outputs:
        st.subheader("Outputs")
        st.markdown(
            f"- Audio path: `{st.session_state.get('audio_path')}`\n"
            f"- Frames directory: `{st.session_state.get('frame_dir')}`"
        )

        frames = list_sample_frames(st.session_state.get("frame_dir"))
        if frames:
            st.subheader("Sample frames")
            cols = st.columns(min(len(frames), 3))
            for idx, frame_path in enumerate(frames):
                with cols[idx % len(cols)]:
                    st.image(
                        str(frame_path),
                        caption=frame_path.name,
                        width="stretch",
                    )
        else:
            st.caption("No frames found yet in the frames directory.")


def render_asr_tab(input_type: str) -> None:
    col_main, col_preview = st.columns([1, 1.4])

    with col_main:
        audio_path = st.session_state.get("audio_path")
        if audio_path is None:
            if input_type == "Video":
                st.warning("Audio file not found. Run the Ingest phase first.")
            else:
                st.warning("Audio file not found. Upload an audio file first.")
        else:
            if st.button("Run ASR"):
                ensure_output_paths()
                transcript_path = st.session_state.get("transcript_path")
                if transcript_path and os.path.exists(transcript_path):
                    try:
                        os.remove(transcript_path)
                    except OSError as exc:
                        st.warning(f"Could not delete transcript file {transcript_path}: {exc}")
                st.session_state["transcript_segments"] = None

                with st.status("Running ASR", expanded=True) as status:
                    try:
                        status.write("Transcribing audio...")
                        asr_bar = st.progress(0, text="ASR: waiting")

                        def asr_progress(current: float, total: float, message: str = "") -> None:
                            pct = int(current * 100 / max(total, 1.0))
                            if pct < 0:
                                pct = 0
                            if pct > 100:
                                pct = 100
                            current_s = int(round(current))
                            total_s = int(round(total))

                            asr_bar.progress(pct, text=f"ASR: {current_s}s / {total_s}s")

                        transcript_segments = run_asr(
                            st.session_state["audio_path"],
                            st.session_state["transcript_path"],
                            progress_cb=asr_progress,
                        )
                        st.session_state["transcript_segments"] = transcript_segments
                        status.write("ASR completed.")
                        st.success("ASR completed.")
                    except Exception as exc:
                        status.update(label="ASR failed", state="error")
                        st.error("Error during ASR.")
                        st.exception(exc)

    with col_preview:
        st.subheader("Transcript segments (preview)")

        segments = load_and_normalize("transcript_segments", "transcript_path")
        if segments:
            display_segments_preview(segments, slider_key="asr_preview")
            transcript_path = st.session_state.get("transcript_path")
            if transcript_path and os.path.exists(transcript_path):
                with open(transcript_path, "rb") as f:
                    st.download_button(
                        "Download transcript JSON",
                        data=f,
                        file_name="transcript_segments.json",
                        mime="application/json",
                    )
        else:
            st.caption("No transcript segments available yet.")


def render_ocr_tab() -> None:
    input_type = st.session_state.get("input_type")
    if input_type != "Video":
        st.warning("OCR is only relevant for video input.")
        return

    frame_dir = st.session_state.get("frame_dir")
    if frame_dir is None:
        st.warning("Frames directory not found. Run the Ingest phase first.")
        return

    col_main, col_preview = st.columns([1, 1.4])

    with col_main:
        if st.button("Run OCR"):
            ensure_output_paths()
            ocr_output_path = st.session_state.get("ocr_output_path")
            if ocr_output_path and os.path.exists(ocr_output_path):
                try:
                    os.remove(ocr_output_path)
                except OSError as exc:
                    st.warning(f"Could not delete OCR output file {ocr_output_path}: {exc}")
            st.session_state["ocr_records"] = None

            with st.status("Running OCR on frames", expanded=True) as status:
                try:
                    status.write("Processing frames for OCR...")
                    ocr_bar = st.progress(0, text="OCR: waiting")

                    def ocr_progress(i: int, total: int, message: str = "") -> None:
                        pct = int(i * 100 / max(total, 1))
                        if pct < 0:
                            pct = 0
                        if pct > 100:
                            pct = 100
                        ocr_bar.progress(pct, text=message or f"OCR: {i}/{total}")

                    ocr_records = run_ocr_on_frames(
                        frame_dir=st.session_state["frame_dir"],
                        ocr_output_path=st.session_state["ocr_output_path"],
                        frame_interval_seconds=st.session_state["frame_interval_seconds"],
                        ocr_frame_stride=st.session_state["ocr_frame_stride"],
                        progress_cb=ocr_progress,
                    )

                    st.session_state["ocr_records"] = ocr_records
                    status.write("OCR completed.")
                    st.success("OCR completed.")
                except Exception as exc:
                    status.update(label="OCR failed", state="error")
                    st.error("Error during OCR.")
                    st.exception(exc)

    with col_preview:
        st.subheader("OCR records (preview)")

        records = load_and_normalize("ocr_records", "ocr_output_path")
        if records:
            display_ocr_preview(records)
            ocr_output_path = st.session_state.get("ocr_output_path")
            if ocr_output_path and os.path.exists(ocr_output_path):
                with open(ocr_output_path, "rb") as f:
                    st.download_button(
                        "Download OCR JSON",
                        data=f,
                        file_name="ocr_frames.json",
                        mime="application/json",
                    )
        else:
            st.caption("No OCR records available yet.")


def render_align_tab() -> None:
    input_type = st.session_state.get("input_type")
    if input_type != "Video":
        st.warning("Alignment is only relevant for video input.")
        return

    if st.session_state.get("transcript_path") is None or st.session_state.get("ocr_output_path") is None:
        st.warning("Transcript or OCR outputs not found. Run ASR and OCR first.")
        return

    col_main, col_preview = st.columns([1, 1.4])

    with col_main:
        if st.button("Run alignment"):
            ensure_output_paths()

            with st.status("Aligning ASR and OCR segments", expanded=True) as status:
                try:
                    transcript_segments = load_and_normalize("transcript_segments", "transcript_path")
                    ocr_records = load_and_normalize("ocr_records", "ocr_output_path")

                    if not transcript_segments or not ocr_records:
                        status.update(label="Alignment failed", state="error")
                        st.error("Missing transcript segments or OCR records.")
                    else:
                        segments_merged = align_transcript_and_ocr(
                            transcript_segments=transcript_segments,
                            ocr_records=ocr_records,
                            segments_path=st.session_state["segments_path"],
                        )
                        st.session_state["segments_merged"] = segments_merged
                        status.write("Alignment completed.")
                        st.success("Alignment completed.")
                except Exception as exc:
                    status.update(label="Alignment failed", state="error")
                    st.error("Error during alignment.")
                    st.exception(exc)

    with col_preview:
        st.subheader("Aligned segments (preview)")

        segments = load_and_normalize("segments_merged", "segments_path")
        if segments:
            display_aligned_preview(segments)
            segments_path = st.session_state.get("segments_path")
            if segments_path and os.path.exists(segments_path):
                with open(segments_path, "rb") as f:
                    st.download_button(
                        "Download aligned segments JSON",
                        data=f,
                        file_name="segments.json",
                        mime="application/json",
                    )
        else:
            st.caption("No aligned segments available yet.")


def render_summarise_tab(input_type: str) -> None:
    if input_type == "Video":
        col_main, col_summary = st.columns([1, 1.4])

        with col_main:
            if st.button("Run summarisation from video"):
                with st.status("Generating summary from video segments", expanded=True) as status:
                    try:
                        segments_raw = st.session_state.get("segments_merged") or load_json_if_exists(
                            st.session_state.get("segments_path")
                        )
                        if not segments_raw:
                            status.write("No aligned segments found. Falling back to ASR transcript.")
                            segments_raw = st.session_state.get("transcript_segments") or load_json_if_exists(
                                st.session_state.get("transcript_path")
                            )

                        segments = normalize_segments(segments_raw)

                        if not segments:
                            status.update(label="Summarisation failed", state="error")
                            st.error("No segments available for summarisation.")
                        else:
                            status.write(
                                f"Summarising {len(segments)} segments with model "
                                f"{st.session_state['summariser_model_name']}."
                            )
                            summary = summarise_segments(
                                segments,
                                model_name=st.session_state["summariser_model_name"],
                                device=st.session_state["summariser_device"],
                                max_chunk_chars=3000,
                                max_length=500,
                                min_length=40,
                            )
                            st.session_state["summary_text"] = summary
                            status.write("Summarisation completed.")
                            st.success("Summarisation completed.")
                    except Exception as exc:
                        status.update(label="Summarisation failed", state="error")
                        st.error("Error during summarisation.")
                        st.exception(exc)

        with col_summary:
            render_summary_output()

    elif input_type == "Audio":
        col_main, col_summary = st.columns([1, 1.4])

        with col_main:
            if st.button("Run summarisation from audio transcript"):
                with st.status("Generating summary from audio transcript", expanded=True) as status:
                    try:
                        segments_raw = st.session_state.get("transcript_segments") or load_json_if_exists(
                            st.session_state.get("transcript_path")
                        )
                        segments = normalize_segments(segments_raw)

                        if not segments:
                            status.update(label="Summarisation failed", state="error")
                            st.error("No transcript segments available for summarisation.")
                        else:
                            status.write(
                                f"Summarising {len(segments)} segments with model "
                                f"{st.session_state['summariser_model_name']}."
                            )
                            summary = summarise_segments(
                                segments,
                                model_name=st.session_state["summariser_model_name"],
                                device=st.session_state["summariser_device"],
                                max_chunk_chars=3000,
                                max_length=500,
                                min_length=40,
                            )
                            st.session_state["summary_text"] = summary
                            status.write("Summarisation completed.")
                            st.success("Summarisation completed.")
                    except Exception as exc:
                        status.update(label="Summarisation failed", state="error")
                        st.error("Error during summarisation.")
                        st.exception(exc)

        with col_summary:
            render_summary_output()

    elif input_type == "Text":
        col_main, col_summary = st.columns([1, 1.4])

        with col_main:
            st.subheader("Input text")

            text_value = st.text_area(
                "Paste or edit the text to summarise",
                value=st.session_state.get("text_input", ""),
                height=300,
            )
            st.session_state["text_input"] = text_value

            if st.button("Run summarisation on text"):
                with st.status("Generating summary from text input", expanded=True) as status:
                    try:
                        segments = build_segments_from_text(st.session_state["text_input"])
                        if not segments:
                            status.update(label="Summarisation failed", state="error")
                            st.error("No text provided for summarisation.")
                        else:
                            status.write(
                                "Summarising text with model "
                                f"{st.session_state['summariser_model_name']}."
                            )
                            summary = summarise_segments(
                                segments,
                                model_name=st.session_state["summariser_model_name"],
                                device=st.session_state["summariser_device"],
                                max_chunk_chars=3000,
                                max_length=500,
                                min_length=40,
                            )
                            st.session_state["summary_text"] = summary
                            status.write("Summarisation completed.")
                            st.success("Summarisation completed.")
                    except Exception as exc:
                        status.update(label="Summarisation failed", state="error")
                        st.error("Error during summarisation.")
                        st.exception(exc)

        with col_summary:
            render_summary_output()

    else:
        st.info("Upload a video, audio, or text file in the sidebar to enable summarisation.")


def render_summary_output() -> None:
    st.subheader("Global summary")

    summary_text = st.session_state.get("summary_text")
    if summary_text:
        st.text_area(
            "Summary",
            summary_text,
            height=300,
        )
        summary_bytes = summary_text.encode("utf-8")
        st.download_button(
            "Download summary as text",
            data=summary_bytes,
            file_name="summary.txt",
            mime="text/plain",
        )
    else:
        st.caption("No summary available yet.")

def render_boxes(input_type: str) -> None:
    renderers = {
        "Video": [
            ("Inspect", lambda: render_inspect_tab(input_type)),
            ("Ingest", render_ingest_tab),
            ("ASR", lambda: render_asr_tab(input_type)),
            ("OCR", render_ocr_tab),
            ("Align", render_align_tab),
            ("Summarise", lambda: render_summarise_tab(input_type)),
        ],
        "Audio": [
            ("Inspect", lambda: render_inspect_tab(input_type)),
            ("ASR", lambda: render_asr_tab(input_type)),
            ("Summarise", lambda: render_summarise_tab(input_type)),
        ],
        "Text": [
            ("Summarise", lambda: render_summarise_tab(input_type)),
        ],
    }

    default_expanded = {"Inspect", "Summarise"}

    for label, fn in renderers.get(input_type, []):
        with st.expander(label, expanded=(label in default_expanded)):
            fn()


# ---------- Main entrypoint for Streamlit ----------


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

    render_pipeline_status(input_type)

    render_boxes(input_type)

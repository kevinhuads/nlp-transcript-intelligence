# NLP Video Pipeline

This repository contains the early version of a multimodal NLP pipeline for videos.  
Starting from a raw video file, the pipeline extracts audio and frames, runs ASR and OCR, aligns speech with slide text, and produces both a global textual summary and a structured summary artefact.

The project is in its initial stages and is designed to be extended and refactored over time. The current focus is on having a clear, testable core pipeline that can be built upon.

---

## Features (current state)

At the moment, the project provides:

1. **Video inspection**
   - Basic metadata: duration, frame rate, file size.

2. **Ingestion**
   - Extracts audio from a video file to `audio.wav`.
   - Extracts frames at a fixed interval (default 3 seconds) and saves them as JPEG files.

3. **Automatic Speech Recognition (ASR)**
   - Uses `faster-whisper` to transcribe the audio track.
   - Stores transcript segments in JSON (`transcript_segments.json`).

4. **Optical Character Recognition (OCR) on frames**
   - Uses `Pillow` and `pytesseract` on sampled frames.
   - Stores OCR records as JSON (`ocr_frames.json`).

5. **Alignment of transcript and OCR**
   - Aligns transcript segments with nearest OCR timestamps.
   - Produces a multimodal `segments.json` that combines speech and slide text.

6. **Summarisation**
   - Uses a Hugging Face `transformers` summarization pipeline.
   - Splits the transcript into chunks and produces a global textual summary.
   - Provides a programmatic helper that computes a global summary and writes a structured summary artefact (`summary.json`) containing:
     - Global summary text.
     - Simple statistics (number of segments, chunks, character counts).
     - Per-chunk summaries and metadata.

All of the above are orchestrated via a single pipeline entry point, and the individual stages can also be called from Python.

---

## Project status

This is a **work in progress** and currently focuses on:

- Establishing a clean modular structure under `src/`.
- Ensuring each step is individually testable.
- Having a basic end to end flow run via a command line interface.
- Beginning to persist derived artefacts in structured form to enable downstream tasks (for example search, RAG, and analytics).

Expect breaking changes as the project evolves. The API and CLI are not considered stable yet.

---

## Repository structure

The relevant part of the repository currently looks like:

```text
.
├── configs/
│   ├── align.yaml
│   ├── asr.yaml
│   ├── full_pipeline.yaml
│   ├── ingest.yaml
│   ├── ocr.yaml
│   ├── project_example.yaml
│   └── summarise.yaml
├── src/
│   ├── __init__.py
│   ├── align.py
│   ├── app.py
│   ├── asr.py
│   ├── ingest.py
│   ├── main.py
│   ├── models.py
│   ├── ocr.py
│   └── summarise.py
└── tests/
    ├── conftest.py
    ├── test_aligns.py
    ├── test_asr.py
    ├── test_ingest.py
    ├── test_main.py
    ├── test_models.py
    ├── test_ocr.py
    └── test_summarise.py
```

The `configs/` directory in the repository root contains example YAML configuration files for individual stages of the pipeline (`align.yaml`, `asr.yaml`, `ingest.yaml`, `ocr.yaml`, `summarise.yaml`), a combined configuration for the full pipeline (`full_pipeline.yaml`), and a project level example (`project_example.yaml`). These files can be used as starting points for structured configuration of the CLI or for integrating the pipeline into larger projects.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kevinhuads/nlp-transcript-intelligence.git
cd NLP-Videos
```

### 2. Create and activate a virtual environment

Example with `venv`:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate
```

### 3. Install Python dependencies

Install using a preferred tool (for example `pip`). The exact versions should be defined in `pyproject.toml` or `requirements.txt`.

Typical runtime dependencies include (non exhaustive):

- `faster-whisper`
- `moviepy`
- `Pillow`
- `pytesseract`
- `tqdm`
- `transformers`
- `torch` (or compatible backend for `transformers`)

Example:

```bash
pip install -r requirements.txt
```

or, when using a `pyproject.toml`, follow the chosen build tool (for example `poetry install`).

### 4. System dependencies

Some components require system level tools.

- **FFmpeg**  
  Required by `moviepy` for reading videos and writing audio.

- **Tesseract OCR**  
  Required by `pytesseract` for running OCR on frames.

Make sure both are installed and accessible in the `PATH`.

---

## Usage

### 1. Basic pipeline run (CLI)

The current CLI entry point is `src.main`. From the project root, run:

```bash
python -m src.main \
  --video-path path/to/input_video.mp4 \
  --output-dir path/to/output_dir \
  --frame-interval-seconds 3 \
  --ocr-frame-stride 2 \
  --summariser-device 0
```

Arguments:

- `--video-path` (required)  
  Path to the input video file.

- `--output-dir` (required)  
  Directory where all artefacts will be written.

- `--frame-interval-seconds` (optional, default `3`)  
  Interval in seconds between extracted frames.

- `--ocr-frame-stride` (optional, default `2`)  
  Use every N th extracted frame for OCR (reduces OCR cost).

- `--summariser-device` (optional, default `0`)  
  Device index for the summarisation model. Use `-1` for CPU.

The CLI internally calls:

- `inspect_video`
- `extract_audio`
- `extract_frames`
- `run_asr` and `preview_transcript`
- `run_ocr_on_frames` and `preview_ocr`
- `align_transcript_and_ocr` and `preview_segments`
- `summarise_segments`

and prints progress to the console. The global summary is printed to stdout.

### 2. Expected output structure

Given `--output-dir OUT_DIR`, the pipeline currently creates:

```text
OUT_DIR/
├── audio.wav
├── frames/
│   ├── frame_00000.jpg
│   ├── frame_00001.jpg
│   └── ...
├── transcript_segments.json
├── ocr_frames.json
└── segments.json
```

- `audio.wav`  
  Audio track extracted from the video.

- `frames/`  
  JPEG frames sampled every `frame_interval_seconds`.

- `transcript_segments.json`  
  List of ASR segments, each with `start`, `end`, and `text`.

- `ocr_frames.json`  
  List of OCR records for sampled frames, each with `time`, `frame`, and `text`.

- `segments.json`  
  Aligned multimodal segments, combining transcript and nearest OCR text.

When using the structured summarisation helper from Python (see below), an additional artefact is created:

```text
OUT_DIR/
└── summary.json
```

`summary.json` contains the global summary text together with basic statistics and per-chunk summaries.

---

## Streamlit application

In addition to the CLI, the repository provides an interactive web interface built with Streamlit. This application exposes the same stages of the pipeline through a set of tabs:

- **Inspect**: video or audio metadata preview and media playback.
- **Ingest**: extraction of audio and frames from video input.
- **ASR**: running speech recognition and previewing transcript segments.
- **OCR**: running OCR on sampled frames and previewing OCR records.
- **Align**: aligning ASR segments with OCR timestamps.
- **Summarise**: generating a global textual summary from aligned or ASR-only segments, or directly from text input.

From the project root, the application can be started with:

```bash
streamlit run src/app.py
```

Adjust the path if the Streamlit entry point lives under a different filename or package. The application manages per session artefacts under a configurable output directory and keeps intermediate results in Streamlit session state so that each tab can build upon previous steps.

---

## Python API

It is also possible to call the pipeline directly from Python code.

### Full pipeline

```python
from src.main import run_full_pipeline

run_full_pipeline(
    video_path="path/to/video.mp4",
    output_dir="path/to/output",
    frame_interval_seconds=3,
    ocr_frame_stride=2,
    device_summariser=0,  # use -1 for CPU
)
```

### Using individual building blocks

Each stage is available as a standalone function:

```python
import os

from src.ingest import inspect_video, extract_audio, extract_frames
from src.asr import run_asr, preview_transcript
from src.ocr import run_ocr_on_frames, preview_ocr
from src.align import align_transcript_and_ocr, preview_segments
from src.summarise import summarise_segments

video_path = "path/to/video.mp4"
output_dir = "path/to/output"
os.makedirs(output_dir, exist_ok=True)

frame_dir = os.path.join(output_dir, "frames")
audio_path = os.path.join(output_dir, "audio.wav")
transcript_path = os.path.join(output_dir, "transcript_segments.json")
ocr_output_path = os.path.join(output_dir, "ocr_frames.json")
segments_path = os.path.join(output_dir, "segments.json")

inspect_video(video_path)
extract_audio(video_path, audio_path)
extract_frames(video_path, frame_dir, interval_seconds=3)

transcript_segments = run_asr(audio_path, transcript_path)
preview_transcript(transcript_segments, n=5)

ocr_records = run_ocr_on_frames(
    frame_dir=frame_dir,
    ocr_output_path=ocr_output_path,
    frame_interval_seconds=3,
    ocr_frame_stride=2,
)
preview_ocr(ocr_records, n=5)

segments_merged = align_transcript_and_ocr(
    transcript_segments=transcript_segments,
    ocr_records=ocr_records,
    segments_path=segments_path,
)
preview_segments(segments_merged, n=5)

# Global summary printed to stdout and returned as a string
summary_text = summarise_segments(
    segments_merged,
    model_name="facebook/bart-large-cnn",
    device=0,
    max_chunk_chars=3000,
    max_length=500,
    min_length=40,
)
print(summary_text)

# Structured summary persisted to summary.json
video_summary = summarise_segments(
    segments=segments_merged,
    video_path=video_path,
    output_dir=output_dir,
    model_name="facebook/bart-large-cnn",
    device=0,
    max_chunk_chars=3000,
    max_length=500,
    min_length=40,
)

print(video_summary.video_id)
print(video_summary.stats)
```

The `summarise_segments_and_save` helper computes a global summary, derives simple statistics, and writes a `summary.json` file in `output_dir`. It returns a `VideoSummary` object that contains:

- `video_id`, `video_path`, `output_dir`
- `generated_at`, `model_name`
- `parameters` used for summarisation
- `stats` about the input and output text
- `summary_text` and a list of `SummaryChunk` objects

This structured artefact is intended for downstream tasks such as search, RAG, evaluation, or analytics.

---

## Testing

The project includes a pytest test suite for the current modules.

From the project root, run:

```bash
pytest -v
```

Tests cover:

- Data models and JSON serialisation (`models.py`)
- Alignment logic (`align.py`)
- ASR caching and interaction with `faster-whisper` (`asr.py`)
- Video ingestion and frame extraction (`ingest.py`)
- OCR processing and caching (`ocr.py`)
- Summarisation logic and chunking (`summarise.py`)
- CLI and orchestration (`main.py`)

Heavy external dependencies (video processing, OCR, large models) are mocked in tests.

---

## Evaluation and benchmarks

This repository includes an evaluation suite under `eval/`. Its purpose is to compare multiple ASR backends using a consistent WER (Word Error Rate) methodology, and to make the accuracy versus latency trade-off explicit.

### Current ASR benchmark summary

The benchmark is run on two complementary subsets of the same public speech dataset:
- A subset with relatively clean recordings.
- A subset with more challenging acoustic conditions.

Key observations from the current runs:
- On the clean subset, top-performing English-focused baselines achieve corpus WER in the high 0.01 range (around 0.017 in the best runs).
- On the harder subset, the best corpus WER in the current runs is in the low 0.03 range (around 0.034 in the best runs).
- Whisper-family models show a clear throughput trade-off. Faster variants reduce latency substantially but do not lead on this dataset-specific WER ranking. They remain attractive as a general-purpose default when the application must handle broader, open-world audio conditions and multilingual inputs.

For details on the dataset splits, manifest format, normalization choices, model list, decoding settings, and the analysis notebook, see `eval/README.md` and `eval/asr_eval_notebook.ipynb`.


### Outputs

Results are written under `eval/results/<dataset>/`:

- `asr_eval_<model>.csv`: per-utterance WER, S/D/I counts, and decoding time.
- `summary_asr_eval_<model>.json`: corpus-level aggregates (including corpus WER) and the exact settings used.

The analysis notebook `eval/asr_eval_notebook.ipynb` compares models on accuracy, latency, and error profiles.


## Roadmap and ideas for future work

This repository is intended to grow into a richer toolkit for video understanding. Some possible future directions include:

- More robust configuration (YAML or JSON config files, environment based settings).
- Support for multiple ASR and OCR backends and language selection.
- Better management of summarisation models and task specific prompts.
- Richer alignment strategies between speech and visual content.
- Metadata export in formats suitable for search and indexing.
- Extension of the Streamlit web UI and exploration of alternative UI frontends.
- Performance optimisations, batching, and distributed processing.
- More comprehensive logging, metrics, and monitoring.

Since this is an early stage project, the actual roadmap may change significantly as the codebase evolves.

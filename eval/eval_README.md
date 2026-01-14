# Evaluation

This folder contains evaluation utilities for the project. The first implemented benchmark is **ASR evaluation with WER (Word Error Rate)** on LibriSpeech splits.
The **OCR evaluation** is still under development.

Workflow:

1. Build a manifest CSV with `id`, `audio_path`, `ref_text`.
2. Run `asr_eval.py` with a YAML config (models, decoding, device).
3. Analyze `asr_eval_*.csv` outputs and aggregated summaries.

---

## Contents

- `asr_eval.py`: YAML-driven runner. Transcribes each utterance, computes WER, writes CSV + JSON summary.
- `asr_backends.py`: Pluggable ASR backends:
  - `faster_whisper`, `transformers_pipeline`, `nemo`, `mms`, `command`
- `asr_metrics.py`: normalization + WER bookkeeping (S/D/I/N).
- `configs/asr.yaml`: evaluation configuration.
- `manifests/`: input manifests (`librispeech_<split>_manifest.csv`).
- `results/`: outputs by dataset.
- `write_manifests.py`: generates LibriSpeech manifests from `.trans.txt`.
- `asr_eval_notebook.ipynb`: analysis and plots.
- `eval_notebook_utils.py`: utilities for the notebooks.

---

## Data and manifests

### Manifest format

CSV columns:

- `id`: utterance id
- `audio_path`: path to `.flac` (absolute or relative)
- `ref_text`: ground-truth transcript

### Generate LibriSpeech manifests

Edit the LibriSpeech root in `write_manifests.py`, then run:

```bash
python eval/write_manifests.py
```

This creates:

- `eval/manifests/librispeech_dev-clean_manifest.csv`
- `eval/manifests/librispeech_dev-other_manifest.csv`
- `eval/manifests/librispeech_test-clean_manifest.csv`
- `eval/manifests/librispeech_test-other_manifest.csv`

---

## Run ASR evaluation

Run all models from the YAML:

```bash
python eval/asr_eval.py --config eval/configs/asr.yaml
```

Run one model only:

```bash
python eval/asr_eval.py --config eval/configs/asr.yaml --model-key large-v3-turbo
```

On Windows with CUDA, the script can spawn one subprocess per model to reduce instability:

```bash
python eval/asr_eval.py --config eval/configs/asr.yaml --spawn-per-model
```

---

## Configuration notes

`eval_asr.models` accepts strings or dicts.

String form (backend inferred):

```yaml
models:
  - large-v3-turbo
  - facebook/hubert-xlarge-ls960-ft
  - stt_en_conformer_ctc_large
```

Backend inference highlights:

- `facebook/...` → `transformers_pipeline`
- `stt_en_*` / NeMo ids → `nemo`
- Whisper-like names → `faster_whisper`
- `facebook/mms-...@eng` (or `:eng`) → `mms` with language adapter

---

## Outputs

For each dataset and model:

- `asr_eval_<model>.csv`: per-utterance `wer`, `substitutions`, `deletions`, `insertions`, `ref_words`, `hyp_words`, `asr_seconds`
- `summary_asr_eval_<model>.json`: aggregates + settings

WER is computed after normalization (`asr_metrics.normalize`): lowercase, punctuation removal, whitespace collapse.

Interpretation:

- **Corpus WER** (preferred for ranking): `sum(S+D+I) / sum(N)`
- **Mean item WER**: average of utterance WERs (more sensitive to short utterances)

---

## Notebook

`asr_eval_notebook.ipynb` loads result CSVs under `eval/results/<dataset>/` and compares models on WER, latency, error composition, and tail behavior.

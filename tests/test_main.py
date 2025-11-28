import os

from src.main import run_full_pipeline, main as cli_main


def test_run_full_pipeline_calls_all_components(tmp_path, monkeypatch):
    base_dir = str(tmp_path)
    video_path = os.path.join(base_dir, "video.mp4")
    output_dir = os.path.join(base_dir, "out")

    # Ensure video path exists for inspect_video if needed
    os.makedirs(base_dir, exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(b"\x00")

    called = {
        "inspect_video": 0,
        "extract_audio": 0,
        "extract_frames": 0,
        "run_asr": 0,
        "preview_transcript": 0,
        "run_ocr_on_frames": 0,
        "preview_ocr": 0,
        "align_transcript_and_ocr": 0,
        "preview_segments": 0,
        "summarise_segments": 0,
    }

    def fake_inspect_video(vp):
        called["inspect_video"] += 1
        assert vp == video_path

    def fake_extract_audio(vp, ap):
        called["extract_audio"] += 1
        assert vp == video_path
        assert ap == os.path.join(output_dir, "audio.wav")

    def fake_extract_frames(vp, fd, interval_seconds):
        called["extract_frames"] += 1
        assert vp == video_path
        assert fd == os.path.join(output_dir, "frames")

    def fake_run_asr(ap, tp, model_size="small", device="cuda", compute_type="int8"):
        called["run_asr"] += 1
        return ["asr_segment_1", "asr_segment_2"]

    def fake_preview_transcript(segments, n=5):
        called["preview_transcript"] += 1
        assert segments == ["asr_segment_1", "asr_segment_2"]

    def fake_run_ocr_on_frames(frame_dir, ocr_output_path, frame_interval_seconds, ocr_frame_stride):
        called["run_ocr_on_frames"] += 1
        return ["ocr_record_1", "ocr_record_2"]

    def fake_preview_ocr(records, n=5):
        called["preview_ocr"] += 1
        assert records == ["ocr_record_1", "ocr_record_2"]

    def fake_align_transcript_and_ocr(transcript_segments, ocr_records, segments_path):
        called["align_transcript_and_ocr"] += 1
        assert transcript_segments == ["asr_segment_1", "asr_segment_2"]
        assert ocr_records == ["ocr_record_1", "ocr_record_2"]
        return ["merged_segment_1", "merged_segment_2"]

    def fake_preview_segments(segments, n=5):
        called["preview_segments"] += 1
        assert segments == ["merged_segment_1", "merged_segment_2"]

    def fake_summarise_segments(
        segments,
        model_name="facebook/bart-large-cnn",
        device=0,
        max_chunk_chars=3000,
        max_length=500,
        min_length=40,
    ):
        called["summarise_segments"] += 1
        assert segments == ["merged_segment_1", "merged_segment_2"]
        return "SUMMARY"

    monkeypatch.setattr("src.main.inspect_video", fake_inspect_video)
    monkeypatch.setattr("src.main.extract_audio", fake_extract_audio)
    monkeypatch.setattr("src.main.extract_frames", fake_extract_frames)
    monkeypatch.setattr("src.main.run_asr", fake_run_asr)
    monkeypatch.setattr("src.main.preview_transcript", fake_preview_transcript)
    monkeypatch.setattr("src.main.run_ocr_on_frames", fake_run_ocr_on_frames)
    monkeypatch.setattr("src.main.preview_ocr", fake_preview_ocr)
    monkeypatch.setattr("src.main.align_transcript_and_ocr", fake_align_transcript_and_ocr)
    monkeypatch.setattr("src.main.preview_segments", fake_preview_segments)
    monkeypatch.setattr("src.main.summarise_segments", fake_summarise_segments)

    run_full_pipeline(
        video_path=video_path,
        output_dir=output_dir,
        frame_interval_seconds=3,
        ocr_frame_stride=2,
        device_summariser=0,
    )

    assert os.path.isdir(output_dir)
    assert os.path.isdir(os.path.join(output_dir, "frames"))

    for key, count in called.items():
        assert count == 1, "Expected {} to be called once".format(key)


def test_cli_main_parses_arguments_and_calls_run_full_pipeline(monkeypatch):
    # Capture the parameters passed to run_full_pipeline
    captured = {}

    def fake_run_full_pipeline(video_path, output_dir, frame_interval_seconds, ocr_frame_stride, device_summariser):
        captured["video_path"] = video_path
        captured["output_dir"] = output_dir
        captured["frame_interval_seconds"] = frame_interval_seconds
        captured["ocr_frame_stride"] = ocr_frame_stride
        captured["device_summariser"] = device_summariser

    class DummyParser:
        def __init__(self, *args, **kwargs):
            pass

        def add_argument(self, *args, **kwargs):
            # No-op, we do not need to store metadata
            pass

        def parse_args(self):
            class Args:
                pass

            args = Args()
            args.video_path = "video_cli.mp4"
            args.output_dir = "out_cli"
            args.frame_interval_seconds = 5
            args.ocr_frame_stride = 3
            args.summariser_device = -1
            return args

    monkeypatch.setattr("src.main.argparse.ArgumentParser", DummyParser)
    monkeypatch.setattr("src.main.run_full_pipeline", fake_run_full_pipeline)

    cli_main()

    assert captured["video_path"] == "video_cli.mp4"
    assert captured["output_dir"] == "out_cli"
    assert captured["frame_interval_seconds"] == 5
    assert captured["ocr_frame_stride"] == 3
    assert captured["device_summariser"] == -1

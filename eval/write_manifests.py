import os
import csv


def iter_transcript_files(root_dir):
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".trans.txt"):
                yield os.path.join(dirpath, name)


def parse_transcript_file(trans_path):
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, ref_text = parts[0].strip(), parts[1].strip()
            if utt_id and ref_text:
                yield utt_id, ref_text


def build_manifest(temp_dir, out_csv_path):
    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "audio_path", "ref_text"])
        writer.writeheader()

        for trans_path in iter_transcript_files(temp_dir):
            trans_dir = os.path.dirname(trans_path)

            for utt_id, ref_text in parse_transcript_file(trans_path):
                audio_path = os.path.join(trans_dir, utt_id + ".flac")
                if not os.path.exists(audio_path):
                    continue

                writer.writerow(
                    {
                        "id": utt_id,
                        "audio_path": audio_path,
                        "ref_text": ref_text,
                    }
                )


if __name__ == "__main__":
    
    for file in ["dev-clean", "dev-other","test-clean","test-other"]:
        temp_dir = fr"D:\NLP-Videos-Data\LibriSpeech\{file}"
        out_csv_path = fr"eval\manifests\librispeech_{file}_manifest.csv"
        build_manifest(temp_dir, out_csv_path)

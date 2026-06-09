"""
Batch forced alignment — processes every audio file in a folder.

Transcript lookup (optional): if a .txt file with the same stem exists in
the transcripts/ dir, it is used for forced alignment. Otherwise Whisper
transcribes automatically.

Usage:
    uv run python scripts/batch_align.py audio/ --model base
    uv run python scripts/batch_align.py audio/ --model small --language en
"""

import argparse
import sys
from pathlib import Path

import stable_whisper

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def process(model, audio_path: Path, transcript_dir: Path, output_dir: Path, language: str | None) -> None:
    txt_path = transcript_dir / (audio_path.stem + ".txt")
    text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else None
    mode = "forced-align" if text else "transcribe+align"
    print(f"\n[{mode}] {audio_path.name}")

    if text:
        result = model.align(str(audio_path), text, language=language)
    else:
        result = model.transcribe(str(audio_path), language=language, word_timestamps=True)

    out = output_dir / audio_path.stem
    out.mkdir(parents=True, exist_ok=True)
    result.save_as_json(str(out / "aligned.json"))
    result.to_srt_vtt(str(out / "aligned.srt"), word_level=True)
    result.to_tsv(str(out / "aligned.tsv"), word_level=True)
    print(f"  → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch alignment for a folder of audio files")
    parser.add_argument("audio_dir", type=Path, help="Folder containing audio files")
    parser.add_argument("--transcripts-dir", type=Path, default=Path("transcripts"),
                        help="Folder of .txt transcripts matching audio stems (default: transcripts/)")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                        help="Root output directory (default: output/)")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    audio_files = [f for f in sorted(args.audio_dir.iterdir()) if f.suffix.lower() in AUDIO_EXTENSIONS]
    if not audio_files:
        print(f"No audio files found in {args.audio_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[batch] Found {len(audio_files)} audio file(s). Loading model: {args.model}")
    model = stable_whisper.load_model(args.model)

    for audio_path in audio_files:
        try:
            process(model, audio_path, args.transcripts_dir, args.output_dir, args.language)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print("\n[batch] Done.")


if __name__ == "__main__":
    main()

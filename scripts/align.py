"""
Forced alignment for ElevenLabs TTS audio files using stable-whisper.

Two modes:
  1. Transcribe + align  (no --text): Whisper transcribes, then aligns word-level.
  2. Forced align        (--text):    Align known transcript to audio directly.

Outputs per run (in output/<stem>/):
  aligned.json   – full word/segment timing data
  aligned.srt    – subtitle file
  aligned.tsv    – tab-separated word timestamps
"""

import argparse
import json
import sys
from pathlib import Path

import stable_whisper


def run(
    audio_path: Path,
    text: str | None,
    model_name: str,
    language: str | None,
    output_dir: Path,
) -> None:
    print(f"[align] Loading model: {model_name}")
    model = stable_whisper.load_model(model_name)

    if text:
        print(f"[align] Forced-aligning known text to: {audio_path.name}")
        result = model.align(str(audio_path), text, language=language)
    else:
        print(f"[align] Transcribing + aligning: {audio_path.name}")
        result = model.transcribe(str(audio_path), language=language, word_timestamps=True)

    out = output_dir / audio_path.stem
    out.mkdir(parents=True, exist_ok=True)

    result.save_as_json(str(out / "aligned.json"))
    result.to_srt_vtt(str(out / "aligned.srt"), word_level=True)
    result.to_tsv(str(out / "aligned.tsv"), word_level=True)

    print(f"[align] Saved outputs to: {out}")
    _print_summary(result)


def _print_summary(result) -> None:
    words = [w for seg in result.segments for w in (seg.words or [])]
    print(f"\n  Segments : {len(result.segments)}")
    print(f"  Words    : {len(words)}")
    if words:
        print(f"\n  First 5 words:")
        for w in words[:5]:
            print(f"    [{w.start:.3f}s -> {w.end:.3f}s]  {w.word!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forced alignment for ElevenLabs TTS audio"
    )
    parser.add_argument("audio", type=Path, help="Path to audio file (mp3/wav/etc.)")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Known transcript text for forced alignment. Omit to let Whisper transcribe.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Path to a .txt file containing the transcript (alternative to --text).",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: base). Larger = more accurate but slower.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. 'en'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root output directory (default: output/).",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    text = args.text
    if args.text_file:
        if not args.text_file.exists():
            print(f"Error: text file not found: {args.text_file}", file=sys.stderr)
            sys.exit(1)
        text = args.text_file.read_text(encoding="utf-8")

    run(
        audio_path=args.audio,
        text=text,
        model_name=args.model,
        language=args.language,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

"""Command-line interface for meetscribe.

Usage:
    python -m meetscribe record --title "Weekly Sync"
    python -m meetscribe process recordings/2026-09-14_1534_my-meeting

Start `record` when your meeting begins; press Enter when the meeting ends. As
soon as you stop, it transcribes the audio and generates notes automatically,
then saves everything into a per-meeting folder under ./recordings.

If transcription ever fails (e.g. a network drop while downloading a model),
the recording is still saved -- re-run it later with `process <folder>`.
"""

from __future__ import annotations

import argparse
import sys
import time

from .notes import DEFAULT_MODEL as DEFAULT_LLM
from .notes import generate_notes
from .recorder import MeetingRecorder
from .storage import MeetingStore
from .transcriber import transcribe


def _transcribe_and_note(
    store: MeetingStore, args: argparse.Namespace, extra_meta: dict
) -> None:
    """Transcribe the store's audio, generate notes, and finalize metadata."""
    print(f"  → Transcribing with Whisper ({args.model})...")
    transcript = transcribe(store.audio_path, model_size=args.model)
    store.save_transcript(transcript.to_timestamped_text(), transcript.language)
    print(f"    transcript.txt saved ({len(transcript.segments)} segments)")

    if args.no_notes:
        print("  → Skipping notes (--no-notes).")
        notes_model = None
    else:
        print(f"  → Generating notes with {args.llm}...")
        notes = generate_notes(transcript.text, model=args.llm)
        store.save_notes(notes, store.title)
        notes_model = args.llm
        print("    notes.md saved")

    meta = {"whisper_model": args.model, "llm_model": notes_model}
    meta.update(extra_meta)
    store.finalize(meta)
    print(f"\n  ✓ Done. Open: {store.notes_path}\n")


def _record(args: argparse.Namespace) -> int:
    store = MeetingStore(base_dir=args.output, title=args.title)
    recorder = MeetingRecorder(
        store.audio_path,
        capture_mic=not args.no_mic,
        mic_index=args.mic_index,
    )

    print(f"\n  Meeting: {args.title}")
    print(f"  Saving to: {store.dir}")
    recorder.start()
    started = time.time()
    sources = "system audio + your mic" if recorder.mic_active else "system audio only"
    if not args.no_mic and not recorder.mic_active:
        sources += "  (no microphone found)"
    print(f"\n  ● Recording {sources}.  Press Enter to stop.\n")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    recorder.stop()
    duration = time.time() - started
    print(f"  ■ Stopped after {duration:.0f}s. Processing...\n")

    try:
        _transcribe_and_note(
            store,
            args,
            {
                "duration_seconds": round(duration),
                "microphone_captured": recorder.mic_active,
            },
        )
    except Exception as exc:
        print(f"\n  ✗ Transcription/notes failed: {exc}")
        print(f"  Your recording is SAFE at: {store.audio_path}")
        print("  Re-run it later (no need to re-record) with:\n")
        print(f'    python -m meetscribe process "{store.dir}"\n')
        return 1
    return 0


def _process(args: argparse.Namespace) -> int:
    try:
        store = MeetingStore.from_dir(args.folder)
    except FileNotFoundError as exc:
        print(f"  {exc}")
        return 1
    if not store.audio_path.exists():
        print(f"  No audio.wav found in {store.dir}")
        return 1

    print(f"\n  Meeting: {store.title}")
    print(f"  Folder: {store.dir}\n")
    try:
        _transcribe_and_note(store, args, {})
    except Exception as exc:
        print(f"\n  ✗ Processing failed: {exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles/redirected output can default to cp1252, which can't
    # encode the status glyphs (or accented transcript text). Force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="meetscribe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="record a meeting, then transcribe + take notes")
    rec.add_argument("--title", default="meeting", help="meeting title (used in folder + notes)")
    rec.add_argument("--output", default="recordings", help="base directory for meetings")
    rec.add_argument("--model", default="small", help="Whisper size: tiny/base/small/medium/large-v3")
    rec.add_argument("--llm", default=DEFAULT_LLM, help="Ollama model for notes")
    rec.add_argument("--no-mic", action="store_true", help="capture system audio only (don't record your microphone)")
    rec.add_argument("--mic-index", type=int, default=None, help="specific input-device index to use as the mic")
    rec.add_argument("--no-notes", action="store_true", help="transcribe only, skip note generation")
    rec.set_defaults(func=_record)

    proc = sub.add_parser(
        "process",
        help="transcribe + take notes on an already-recorded meeting folder",
    )
    proc.add_argument("folder", help="path to a meeting folder containing audio.wav")
    proc.add_argument("--model", default="small", help="Whisper size: tiny/base/small/medium/large-v3")
    proc.add_argument("--llm", default=DEFAULT_LLM, help="Ollama model for notes")
    proc.add_argument("--no-notes", action="store_true", help="transcribe only, skip note generation")
    proc.set_defaults(func=_process)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

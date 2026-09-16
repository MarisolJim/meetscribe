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
import re
import sys
import time

from .notes import DEFAULT_MODEL as DEFAULT_LLM
from .notes import generate_notes
from .recorder import MeetingRecorder
from .storage import MeetingStore
from .transcriber import (
    dedupe_echo,
    labeled_plain_text,
    labeled_timestamped_text,
    merge_labeled,
    transcribe,
)


def _transcribe_and_note(
    store: MeetingStore, args: argparse.Namespace, extra_meta: dict
) -> None:
    """Transcribe the store's audio, generate notes, and finalize metadata.

    Transcription failure is fatal (nothing to save), but the recording is kept
    by the caller. A notes failure is NOT fatal: the transcript is preserved and
    metadata is still written, so the run always leaves usable output behind.
    """
    # When separate speaker tracks exist (a mic was captured), transcribe each
    # and label the merged transcript "You" vs "Others". Otherwise fall back to
    # the single mixed track (older recordings, or --no-mic runs).
    labeled = store.you_path.exists() and store.others_path.exists()
    if labeled:
        print(f"  → Transcribing 2 tracks (You + Others) with Whisper ({args.model})...")
        you_t = transcribe(store.you_path, model_size=args.model)
        others_t = transcribe(store.others_path, model_size=args.model)
        model_used = you_t.model_used or others_t.model_used or args.model
        segments = dedupe_echo(merge_labeled([("You", you_t), ("Others", others_t)]))
        store.save_transcript(
            labeled_timestamped_text(segments), you_t.language or others_t.language
        )
        notes_input = labeled_plain_text(segments)
        n_segments = len(segments)
    else:
        print(f"  → Transcribing with Whisper ({args.model})...")
        transcript = transcribe(store.audio_path, model_size=args.model)
        model_used = transcript.model_used
        store.save_transcript(transcript.to_timestamped_text(), transcript.language)
        notes_input = transcript.text
        n_segments = len(transcript.segments)
    print(f"    transcript.txt saved ({n_segments} segments)")

    notes_model: str | None = None
    notes_ok = False
    if args.no_notes:
        print("  → Skipping notes (--no-notes).")
    else:
        print(f"  → Generating notes with {args.llm}...")
        try:
            notes = generate_notes(notes_input, model=args.llm)
            store.save_notes(notes, store.title)
            notes_model = args.llm
            notes_ok = True
            print("    notes.md saved")
        except Exception as exc:
            # Don't lose the transcript over a notes failure (e.g. Ollama down).
            print(f"    ! Note generation failed: {exc}")
            print("      Is Ollama running? Your transcript is saved; regenerate notes with:")
            print(f'        python -m meetscribe process "{store.dir}"')
            store.save_notes(
                f"_Note generation failed ({exc}). The transcript is in "
                f"transcript.txt; re-run `process` on this folder to try again._",
                store.title,
            )

    meta = {
        "whisper_model": model_used or args.model,
        "whisper_model_requested": args.model,
        "llm_model": notes_model,
        "notes_generated": notes_ok,
        "speaker_labeled": labeled,
    }
    meta.update(extra_meta)
    store.finalize(meta)

    if model_used and model_used != args.model:
        print(f"  (note: used cached '{model_used}' model instead of '{args.model}')")
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
    print(f"\n  ● Recording {sources}.  Press Enter to stop.")
    if recorder.mic_active:
        print("    (tip: use headphones so 'You' and 'Others' stay cleanly separated)")
    print()
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


def _regenerate_notes(store: MeetingStore, args: argparse.Namespace) -> None:
    """Regenerate notes from the existing transcript, without re-transcribing."""
    raw = store.transcript_path.read_text(encoding="utf-8")
    # transcript.txt is timestamped ("[00:00:01 -> 00:00:05] text"); strip the
    # timestamps to feed the model cleaner prose.
    plain = "\n".join(
        re.sub(r"^\[[^\]]*\]\s*", "", line) for line in raw.splitlines()
    ).strip()
    print(f"  → Regenerating notes with {args.llm}...")
    notes = generate_notes(plain, model=args.llm)
    store.save_notes(notes, store.title)
    store.finalize({"llm_model": args.llm, "notes_generated": True})
    print(f"    notes.md saved\n\n  ✓ Done. Open: {store.notes_path}\n")


def _process(args: argparse.Namespace) -> int:
    try:
        store = MeetingStore.from_dir(args.folder)
    except FileNotFoundError as exc:
        print(f"  {exc}")
        return 1

    print(f"\n  Meeting: {store.title}")
    print(f"  Folder: {store.dir}\n")

    notes_only = getattr(args, "notes_only", False)
    if notes_only and not store.transcript_path.exists():
        print("  --notes-only needs an existing transcript.txt (none found).")
        return 1
    if not notes_only and not store.audio_path.exists():
        print(f"  No audio.wav found in {store.dir}")
        return 1

    try:
        if notes_only:
            _regenerate_notes(store, args)
        else:
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
    proc.add_argument("--notes-only", action="store_true", help="regenerate notes from the existing transcript (no re-transcribing)")
    proc.set_defaults(func=_process)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

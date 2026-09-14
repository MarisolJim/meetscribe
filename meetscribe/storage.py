"""Local, per-meeting storage.

Every meeting gets its own timestamped folder under ``recordings/`` holding:
    audio.wav        the raw recording
    transcript.txt   timestamped transcript
    notes.md         the generated notes
    meeting.json     metadata (title, timestamps, model names)

Nothing is ever uploaded anywhere -- it all lives on your disk.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug or "meeting"


class MeetingStore:
    """Manages the folder and files for a single meeting."""

    def __init__(self, base_dir: str | Path = "recordings", title: str = "meeting"):
        self.title = title
        started = datetime.now()
        folder_name = f"{started:%Y-%m-%d_%H%M}_{_slugify(title)}"
        self.dir = Path(base_dir) / folder_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {
            "title": title,
            "started_at": started.isoformat(timespec="seconds"),
        }

    @property
    def audio_path(self) -> Path:
        return self.dir / "audio.wav"

    @property
    def transcript_path(self) -> Path:
        return self.dir / "transcript.txt"

    @property
    def notes_path(self) -> Path:
        return self.dir / "notes.md"

    def save_transcript(self, timestamped_text: str, language: str) -> None:
        self.transcript_path.write_text(timestamped_text, encoding="utf-8")
        self.meta["language"] = language

    def save_notes(self, notes_md: str, title: str) -> None:
        header = f"# {title}\n\n_{self.meta['started_at']}_\n\n"
        self.notes_path.write_text(header + notes_md + "\n", encoding="utf-8")

    def finalize(self, extra: dict | None = None) -> None:
        if extra:
            self.meta.update(extra)
        self.meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
        (self.dir / "meeting.json").write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )

"""Transcribe a WAV file into text using faster-whisper (a local Whisper model).

The model weights are downloaded once from Hugging Face on first use and cached
under ~/.cache/huggingface, so subsequent runs are fully offline.

Model sizes trade accuracy for speed:
    tiny  base  small  medium  large-v3
    fast  <-------------------->  accurate
On a CPU, "small" is a good default; "medium" is more accurate but slower.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    language: str
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments).strip()

    def to_timestamped_text(self) -> str:
        lines = []
        for s in self.segments:
            stamp = f"[{_fmt(s.start)} -> {_fmt(s.end)}]"
            lines.append(f"{stamp} {s.text.strip()}")
        return "\n".join(lines)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe(
    wav_path: str | Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
) -> Transcript:
    """Transcribe a WAV file and return a Transcript.

    compute_type="int8" keeps memory/CPU usage low while staying accurate.
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=True,  # skip long silences -> faster, cleaner transcript
    )
    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text)
        for s in segments_iter
    ]
    return Transcript(language=info.language, segments=segments)

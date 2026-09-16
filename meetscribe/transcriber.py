"""Transcribe a WAV file into text using faster-whisper (a local Whisper model).

The model weights are downloaded once from Hugging Face on first use and cached
under ~/.cache/huggingface, so subsequent runs are fully offline.

Model sizes trade accuracy for speed:
    tiny  base  small  medium  large-v3
    fast  <-------------------->  accurate
On a CPU, "small" is a good default; "medium" is more accurate but slower.

Robustness: if the requested model isn't cached and can't be downloaded (e.g. a
network drop), transcription falls back to the best already-cached model instead
of failing, so the pipeline always completes when *any* model is available.
"""

from __future__ import annotations

import os
import re

# The "xet" transfer backend can fail on flaky networks (connection resets while
# fetching xet-read-token). Fall back to plain HTTPS, which retries more reliably.
# Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel
from huggingface_hub import constants as hf_constants

# Whisper model sizes in ascending order of quality (and cost).
QUALITY_ORDER = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
_REQUIRED_FILES = {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    language: str
    segments: list[TranscriptSegment]
    model_used: str = ""  # the model that actually ran (may differ if we fell back)

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


@dataclass
class LabeledSegment:
    start: float
    end: float
    speaker: str
    text: str


def merge_labeled(parts: list[tuple[str, Transcript]]) -> list[LabeledSegment]:
    """Interleave segments from several labeled transcripts, ordered by time."""
    merged: list[LabeledSegment] = []
    for speaker, transcript in parts:
        for s in transcript.segments:
            if s.text.strip():
                merged.append(LabeledSegment(s.start, s.end, speaker, s.text.strip()))
    merged.sort(key=lambda seg: seg.start)
    return merged


def _tokens(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def dedupe_echo(segments: list[LabeledSegment], tol: float = 2.0) -> list[LabeledSegment]:
    """Drop "You" segments that echo an "Others" segment at the same time.

    Without headphones, the mic picks up the other participants coming out of the
    speakers, so their words appear on both tracks. The loopback ("Others") is the
    authoritative copy, so we remove the mic's echoed duplicate. Real "You" speech
    doesn't match any "Others" line, so it is kept.
    """
    others = [s for s in segments if s.speaker == "Others"]
    kept: list[LabeledSegment] = []
    for s in segments:
        if s.speaker == "You":
            toks = _tokens(s.text)
            if toks:
                is_echo = False
                for o in others:
                    # overlapping (with tolerance) and highly similar text
                    if s.start <= o.end + tol and o.start <= s.end + tol:
                        ot = _tokens(o.text)
                        if ot and len(toks & ot) / max(len(toks), len(ot)) >= 0.8:
                            is_echo = True
                            break
                if is_echo:
                    continue
        kept.append(s)
    return kept


def labeled_timestamped_text(segments: list[LabeledSegment]) -> str:
    """e.g. "[00:01:23] Others: ...". Saved as the transcript."""
    return "\n".join(f"[{_fmt(s.start)}] {s.speaker}: {s.text}" for s in segments)


def labeled_plain_text(segments: list[LabeledSegment]) -> str:
    """e.g. "Others: ...". Fed to the notes model so it can attribute speakers."""
    return "\n".join(f"{s.speaker}: {s.text}" for s in segments)


def is_model_cached(size: str) -> bool:
    """True if every required file for this model size is already on disk."""
    repo_dir = Path(hf_constants.HF_HUB_CACHE) / f"models--Systran--faster-whisper-{size}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir() and _REQUIRED_FILES <= {p.name for p in snap.iterdir()}:
            return True
    return False


def cached_models() -> list[str]:
    """Known model sizes that are fully cached, ascending by quality."""
    return [s for s in QUALITY_ORDER if is_model_cached(s)]


def _pick_fallback(requested: str, cached: list[str]) -> str | None:
    """Choose the cached model closest to the requested quality (prefer faster on ties)."""
    if not cached:
        return None
    req_rank = QUALITY_ORDER.index(requested) if requested in QUALITY_ORDER else len(QUALITY_ORDER)
    return min(cached, key=lambda s: (abs(QUALITY_ORDER.index(s) - req_rank), QUALITY_ORDER.index(s)))


def _load_model(model_size: str, device: str, compute_type: str) -> tuple[WhisperModel, str]:
    """Load the requested model, or fall back to a cached one. Returns (model, size_used)."""
    kwargs = {"device": device, "compute_type": compute_type}

    # 1. Already cached -> load offline (fast, no network at all).
    if is_model_cached(model_size):
        return WhisperModel(model_size, local_files_only=True, **kwargs), model_size

    # 2. Not cached -> try to download it.
    try:
        return WhisperModel(model_size, **kwargs), model_size
    except Exception as exc:
        # 3. Download failed -> fall back to the best cached model, if any.
        fallback = _pick_fallback(model_size, cached_models())
        if fallback is None:
            raise RuntimeError(
                f"Could not load Whisper model '{model_size}' and no model is cached "
                f"locally to fall back to. Connect to the internet once to download a "
                f"model (e.g. `--model base`). Original error: {exc}"
            ) from exc
        print(
            f"  ! Could not load '{model_size}' (no network?). "
            f"Falling back to cached '{fallback}'."
        )
        return WhisperModel(fallback, local_files_only=True, **kwargs), fallback


def transcribe(
    wav_path: str | Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
) -> Transcript:
    """Transcribe a WAV file and return a Transcript.

    compute_type="int8" keeps memory/CPU usage low while staying accurate.
    Falls back to a cached model if the requested one is unavailable.
    """
    model, model_used = _load_model(model_size, device, compute_type)
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=True,  # skip long silences -> faster, cleaner transcript
    )
    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text)
        for s in segments_iter
    ]
    return Transcript(language=info.language, segments=segments, model_used=model_used)

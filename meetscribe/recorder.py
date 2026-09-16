"""Record meeting audio to a local WAV file.

Interactive meetings need BOTH sides:
  * the system-audio *loopback* (everyone else, via WASAPI), and
  * your *microphone* (you).

PyAudio delivers audio buffers to a callback as they arrive, so stopping simply
closes the streams -- no blocking read that could wedge when audio goes idle.

The two sources can't be mixed by naive index alignment: the loopback delivers
NO data while the system is silent, whereas the mic streams continuously. So
each incoming buffer is timestamped against one shared clock and placed at its
true position on a master timeline (silent gaps become silence). This keeps your
voice and the others' voices aligned in real time.

Audio is written as 16-bit PCM at 16 kHz mono, which is exactly what Whisper
wants -- so no resampling step is needed later.
"""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

# Whisper is trained on 16 kHz mono audio.
TARGET_RATE = 16000
TARGET_CHANNELS = 1
CHUNK = 1024


def _write_wav(path, samples: np.ndarray) -> None:
    """Write a mono 16-bit 16 kHz WAV from an int16 sample array."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(TARGET_RATE)
        wav.writeframes(samples.tobytes())


def _to_mono_16k(raw: bytes, src_rate: int, src_channels: int) -> np.ndarray:
    """Downmix to mono and resample to 16 kHz, returning an int16 array."""
    samples = np.frombuffer(raw, dtype=np.int16)
    if src_channels > 1:
        samples = samples.reshape(-1, src_channels).mean(axis=1).astype(np.int16)

    if src_rate != TARGET_RATE and samples.size:
        # Simple linear resample -- fine for speech transcription.
        n_out = int(round(samples.size * TARGET_RATE / src_rate))
        x_old = np.linspace(0, 1, samples.size, endpoint=False)
        x_new = np.linspace(0, 1, n_out, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.int16)

    return samples


class _DeviceCapture:
    """Captures one input/loopback device into timestamped 16 kHz mono buffers.

    Each buffer is tagged with the wall-clock offset (seconds since the shared
    recording start) at which its callback fired, so it can later be placed on a
    common timeline regardless of gaps in delivery.
    """

    def __init__(self, pa: pyaudio.PyAudio, device: dict, start_time: float):
        self._device = device
        self._start_time = start_time
        self._src_rate = int(device["defaultSampleRate"])
        self._src_channels = int(device["maxInputChannels"])
        self._lock = threading.Lock()
        self._buffers: list[tuple[float, np.ndarray]] = []  # (end_offset_s, samples)

        self._stream = pa.open(
            format=pyaudio.paInt16,
            channels=self._src_channels,
            rate=self._src_rate,
            frames_per_buffer=CHUNK,
            input=True,
            input_device_index=device["index"],
            stream_callback=self._callback,
        )

    def _callback(self, in_data, frame_count, time_info, status):
        end_offset = time.monotonic() - self._start_time
        samples = _to_mono_16k(in_data, self._src_rate, self._src_channels)
        with self._lock:
            self._buffers.append((end_offset, samples))
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        self._stream.start_stream()

    def stop(self) -> None:
        self._stream.stop_stream()
        self._stream.close()

    def to_timeline(self, total_samples: int) -> np.ndarray:
        """Place captured buffers onto a zero-filled timeline of the given length."""
        timeline = np.zeros(total_samples, dtype=np.int32)
        with self._lock:
            buffers = list(self._buffers)
        for end_offset, samples in buffers:
            n = samples.size
            if n == 0:
                continue
            end_idx = int(round(end_offset * TARGET_RATE))
            start_idx = end_idx - n
            # Clip the buffer to the timeline bounds.
            src_lo = max(0, -start_idx)
            start_idx = max(0, start_idx)
            end_idx = min(total_samples, end_idx)
            width = end_idx - start_idx
            if width > 0:
                timeline[start_idx:end_idx] = samples[src_lo:src_lo + width]
        return timeline


class MeetingRecorder:
    """Records mic + system-audio loopback, mixed into one WAV on stop."""

    def __init__(
        self,
        output_path: str | Path,
        capture_mic: bool = True,
        mic_index: int | None = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.capture_mic = capture_mic
        self.mic_index = mic_index
        self._pa: pyaudio.PyAudio | None = None
        self._loopback: _DeviceCapture | None = None
        self._mic: _DeviceCapture | None = None
        self._start_time = 0.0
        self.mic_active = False  # set True if a mic stream was opened
        # Per-speaker tracks written alongside the mix, for "You vs Others"
        # labeling. Set to a real path in stop() when that track has audio.
        self.you_path: Path | None = None
        self.others_path: Path | None = None

    def _find_loopback_device(self, p: pyaudio.PyAudio) -> dict:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for lb in p.get_loopback_device_info_generator():
            if default_speakers["name"] in lb["name"]:
                return lb
        for lb in p.get_loopback_device_info_generator():
            return lb
        raise RuntimeError(
            "No WASAPI loopback device found. Is an audio output device active?"
        )

    def _find_mic_device(self, p: pyaudio.PyAudio) -> dict | None:
        try:
            if self.mic_index is not None:
                return p.get_device_info_by_index(self.mic_index)
            return p.get_default_input_device_info()
        except (OSError, ValueError):
            return None

    def start(self) -> None:
        if self._pa is not None:
            raise RuntimeError("Recorder already started.")

        self._pa = pyaudio.PyAudio()
        loop_dev = self._find_loopback_device(self._pa)

        mic_dev = self._find_mic_device(self._pa) if self.capture_mic else None
        # A loopback device also shows up as an input; don't use it as the mic.
        if mic_dev is not None and mic_dev.get("isLoopbackDevice"):
            mic_dev = None

        # Single shared clock for both streams.
        self._start_time = time.monotonic()
        self._loopback = _DeviceCapture(self._pa, loop_dev, self._start_time)

        if mic_dev is not None:
            try:
                self._mic = _DeviceCapture(self._pa, mic_dev, self._start_time)
                self.mic_active = True
            except OSError:
                self._mic = None  # mic busy/unavailable -> loopback only

        self._loopback.start()
        if self._mic is not None:
            self._mic.start()

    def stop(self) -> Path:
        """Stop both streams, write the mix, and write per-speaker tracks.

        Writes ``audio.wav`` (the mix, for playback) plus ``others.wav`` (system
        loopback) and, when a mic was captured, ``you.wav`` -- the separate
        tracks used later to label the transcript as "You" vs "Others".
        """
        elapsed = time.monotonic() - self._start_time
        if self._loopback is not None:
            self._loopback.stop()
        if self._mic is not None:
            self._mic.stop()
        if self._pa is not None:
            self._pa.terminate()

        total_samples = max(1, int(round(elapsed * TARGET_RATE)))
        others = (
            self._loopback.to_timeline(total_samples)
            if self._loopback is not None
            else np.zeros(total_samples, dtype=np.int32)
        )
        you = (
            self._mic.to_timeline(total_samples)
            if self._mic is not None
            else None
        )

        # Mix (clip: both talking at once can exceed int16 range).
        mix = others.copy()
        if you is not None:
            mix += you
        _write_wav(self.output_path, np.clip(mix, -32768, 32767).astype(np.int16))

        # Per-speaker tracks for labeling.
        others_path = self.output_path.parent / "others.wav"
        _write_wav(others_path, np.clip(others, -32768, 32767).astype(np.int16))
        self.others_path = others_path
        if you is not None:
            you_path = self.output_path.parent / "you.wav"
            _write_wav(you_path, np.clip(you, -32768, 32767).astype(np.int16))
            self.you_path = you_path

        self._pa = None
        self._loopback = None
        self._mic = None
        return self.output_path


class LoopbackRecorder(MeetingRecorder):
    """System-audio only (no microphone). Kept for probes/tests."""

    def __init__(self, output_path: str | Path):
        super().__init__(output_path, capture_mic=False)

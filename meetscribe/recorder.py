"""Record meeting audio to a local WAV file.

We capture the system-audio *loopback* (everything coming out of your speakers,
i.e. the other participants) using WASAPI. PyAudio delivers audio buffers to a
callback as they arrive, so stopping simply closes the stream -- there is no
blocking read that could wedge if the audio goes idle mid-meeting.

Audio is written as 16-bit PCM at 16 kHz mono, which is exactly what Whisper
wants -- so no resampling step is needed later.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

# Whisper is trained on 16 kHz mono audio.
TARGET_RATE = 16000
TARGET_CHANNELS = 1
CHUNK = 1024


class LoopbackRecorder:
    """Records system-audio loopback to a WAV file until stopped."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._wav: wave.Wave_write | None = None
        self._lock = threading.Lock()
        self._src_rate = TARGET_RATE
        self._src_channels = TARGET_CHANNELS

    def _find_loopback_device(self, p: pyaudio.PyAudio) -> dict:
        """Return the loopback device matching the default speakers."""
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])

        # Prefer the loopback whose name matches the current default output.
        for lb in p.get_loopback_device_info_generator():
            if default_speakers["name"] in lb["name"]:
                return lb
        # Fall back to the first loopback device we can find.
        for lb in p.get_loopback_device_info_generator():
            return lb
        raise RuntimeError(
            "No WASAPI loopback device found. Is an audio output device active?"
        )

    def _callback(self, in_data, frame_count, time_info, status):
        """Called by PortAudio on its own thread with each incoming buffer."""
        with self._lock:
            if self._wav is not None:
                self._wav.writeframes(self._to_target(in_data))
        return (None, pyaudio.paContinue)

    def _to_target(self, raw: bytes) -> bytes:
        """Downmix to mono and resample to 16 kHz."""
        samples = np.frombuffer(raw, dtype=np.int16)
        if self._src_channels > 1:
            samples = (
                samples.reshape(-1, self._src_channels).mean(axis=1).astype(np.int16)
            )

        if self._src_rate != TARGET_RATE and samples.size:
            # Simple linear resample -- fine for speech transcription.
            n_out = int(round(samples.size * TARGET_RATE / self._src_rate))
            x_old = np.linspace(0, 1, samples.size, endpoint=False)
            x_new = np.linspace(0, 1, n_out, endpoint=False)
            samples = np.interp(x_new, x_old, samples).astype(np.int16)

        return samples.tobytes()

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("Recorder already started.")

        self._pa = pyaudio.PyAudio()
        device = self._find_loopback_device(self._pa)
        self._src_rate = int(device["defaultSampleRate"])
        self._src_channels = int(device["maxInputChannels"])

        self._wav = wave.open(str(self.output_path), "wb")
        self._wav.setnchannels(TARGET_CHANNELS)
        self._wav.setsampwidth(2)  # 16-bit
        self._wav.setframerate(TARGET_RATE)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._src_channels,
            rate=self._src_rate,
            frames_per_buffer=CHUNK,
            input=True,
            input_device_index=device["index"],
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def stop(self) -> Path:
        """Stop recording and return the path to the finished WAV file."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        with self._lock:
            if self._wav is not None:
                self._wav.close()
                self._wav = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        return self.output_path

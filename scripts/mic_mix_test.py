"""Verify mic + loopback mixing: play TTS (loopback) while capturing, then mix."""

import os
import subprocess
import time
import wave

import numpy as np

from meetscribe.recorder import MeetingRecorder

SPOKEN = "This is the other participant speaking in the meeting."
PS_TTS = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    f"$s.Rate = 0; $s.Speak('{SPOKEN}')"
)

rec = MeetingRecorder("recordings/_mixtest.wav", capture_mic=True)
rec.start()
print("mic_active =", rec.mic_active, flush=True)
subprocess.run(["powershell", "-NoProfile", "-Command", PS_TTS], check=True)
time.sleep(0.5)
path = rec.stop()
print("Stopped cleanly.", flush=True)

with wave.open(str(path)) as w:
    n, rt = w.getnframes(), w.getframerate()
    data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
rms = float(np.sqrt(np.mean(data**2))) if data.size else 0.0
print(f"Mixed: {n/rt:.1f}s  {os.path.getsize(path)} bytes  RMS={rms:.0f}", flush=True)
print("RESULT:", "system audio present in mix" if rms > 50 else "mix is silent", flush=True)

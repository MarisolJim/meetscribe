"""End-to-end smoke test: speak -> record via loopback -> transcribe."""

import subprocess
import sys
import time

from meetscribe.recorder import LoopbackRecorder
from meetscribe.transcriber import transcribe

SPOKEN = (
    "Hello, this is a test of the meeting note taking application. "
    "Today we will discuss the project timeline and assign action items to the team."
)

PS_TTS = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Rate = 0; "
    f"$s.Speak('{SPOKEN}')"
)


def main() -> None:
    rec = LoopbackRecorder("recordings/_e2e.wav")
    rec.start()
    print("Recording... speaking the test sentence through the speakers.", flush=True)

    # Speak synchronously; loopback captures it while this runs.
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", PS_TTS],
        check=True,
    )
    time.sleep(0.5)  # let the tail of the audio flush
    wav = rec.stop()
    print("Recording stopped. Transcribing (first run downloads the model)...", flush=True)

    t0 = time.time()
    transcript = transcribe(wav, model_size="base")
    elapsed = time.time() - t0

    print("-" * 60, flush=True)
    print(f"Detected language: {transcript.language}", flush=True)
    print(f"Transcription took {elapsed:.1f}s", flush=True)
    print("TRANSCRIPT:", flush=True)
    print(transcript.text, flush=True)
    print("-" * 60, flush=True)
    print("ORIGINAL: ", SPOKEN, flush=True)


if __name__ == "__main__":
    sys.exit(main())

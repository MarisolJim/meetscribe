"""Automated run of the full pipeline (non-interactive), to validate storage."""

import subprocess
import time
from pathlib import Path

from meetscribe.notes import generate_notes
from meetscribe.recorder import LoopbackRecorder
from meetscribe.storage import MeetingStore
from meetscribe.transcriber import transcribe

SPOKEN = (
    "Welcome to the design review. Maria will finalize the logo by Thursday. "
    "We decided to ship the beta on October first. "
    "James, please email the investors with the updated roadmap."
)
PS_TTS = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    f"$s.Rate = 0; $s.Speak('{SPOKEN}')"
)

store = MeetingStore(title="Design Review")
rec = LoopbackRecorder(store.audio_path)
rec.start()
subprocess.run(["powershell", "-NoProfile", "-Command", PS_TTS], check=True)
time.sleep(0.5)
rec.stop()

transcript = transcribe(store.audio_path, model_size="base")
store.save_transcript(transcript.to_timestamped_text(), transcript.language)
notes = generate_notes(transcript.text)
store.save_notes(notes, store.title)
store.finalize({"whisper_model": "base", "llm_model": "llama3.2:3b"})

print("Meeting folder contents:")
for f in sorted(store.dir.iterdir()):
    print(f"  {f.name:16} {f.stat().st_size:>7} bytes")
print("\n--- notes.md ---")
print(Path(store.notes_path).read_text(encoding="utf-8"))

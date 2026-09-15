# meetscribe

Record your online meetings, transcribe them locally, and automatically generate
notes when the meeting ends — all running **100% on your machine** with
open-source models. Nothing is uploaded anywhere.

- 🎙️ **Capture** both sides of an interactive meeting — everyone else via Windows WASAPI loopback **and your own microphone** — mixed on a shared timeline
- 📝 **Transcribe** locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (an OpenAI Whisper model from Hugging Face)
- 🧠 **Summarize** into structured notes with a local LLM via [Ollama](https://ollama.com)
- 💾 **Store** each meeting in its own folder on your disk

## How it works

```
 you (mic) ─┐
            ├─▶ mix on shared timeline ─▶ transcribe (Whisper) ─▶ notes (local LLM)
others (loopback) ─┘        audio.wav          transcript.txt          notes.md
```

Because the loopback delivers no data while the system is silent (but your mic
streams continuously), each incoming audio buffer is timestamped against one
shared clock and placed at its true position on a master timeline. That keeps
your voice and the others' voices aligned in real time.

## Requirements

- Windows 10/11 (uses WASAPI loopback for system-audio capture)
- Python 3.12+ (developed on 3.14)
- [Ollama](https://ollama.com) installed and running, with a model pulled:
  ```bash
  ollama pull llama3.2:3b
  ```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Start it when your meeting begins; press **Enter** when it ends. It then
transcribes and writes notes automatically.

```bash
python -m meetscribe record --title "Weekly Sync"
```

### Shortcut (recommended)

`meet.sh` activates the virtual environment for you, so you can run the tool
from anywhere without `cd`-ing in or activating the venv by hand. Add this line
to your `~/.bashrc` (Git Bash) once:

```bash
alias meet="$HOME/Documents/meetscribe/meet.sh"
```

Then simply:

```bash
meet record --title "Weekly Sync"
meet process "recordings/2026-09-15_1001_my-meeting" --notes-only
```

To regenerate notes for **every** recorded meeting (e.g. after changing the
model or prompt), without re-transcribing:

```bash
./scripts/redo_all_notes.sh              # default model
./scripts/redo_all_notes.sh llama3.1:8b  # a specific model
```

Output lands in `recordings/<date>_<title>/`:

| File | Contents |
|------|----------|
| `audio.wav` | the raw recording |
| `transcript.txt` | timestamped transcript |
| `notes.md` | generated summary, key points, decisions, action items |
| `meeting.json` | metadata |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--title` | `meeting` | Meeting title (used in the folder name and notes) |
| `--model` | `small` | Whisper size: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `--llm` | `llama3.2:3b` | Ollama model used for notes (try `llama3.1:8b` for better quality) |
| `--no-mic` | off | Capture system audio only (don't record your microphone) |
| `--mic-index` | auto | Specific input-device index to use as the mic (see below) |
| `--no-notes` | off | Transcribe only, skip note generation |
| `--output` | `recordings` | Base directory for saved meetings |

Your default microphone is used automatically. To list devices and their
indices (e.g. if you have several mics):

```bash
python scripts/list_audio_devices.py
```

## Notes on models

- **Whisper size** trades speed for accuracy. `small` is a good CPU default;
  `medium` is more accurate but slower.
- **Note quality** improves with a larger Ollama model. `llama3.2:3b` is fast;
  `llama3.1:8b` or `mistral` reason better about decisions vs. discussion.
- On first run, models download from Hugging Face / Ollama and are cached for
  offline use afterwards.

## Privacy

Audio, transcripts, and notes never leave your computer. The `recordings/`
folder is git-ignored so your meetings are never committed.

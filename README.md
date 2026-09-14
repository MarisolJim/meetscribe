# meetscribe

Record your online meetings, transcribe them locally, and automatically generate
notes when the meeting ends — all running **100% on your machine** with
open-source models. Nothing is uploaded anywhere.

- 🎙️ **Capture** system audio (everyone in the meeting) via Windows WASAPI loopback
- 📝 **Transcribe** locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (an OpenAI Whisper model from Hugging Face)
- 🧠 **Summarize** into structured notes with a local LLM via [Ollama](https://ollama.com)
- 💾 **Store** each meeting in its own folder on your disk

## How it works

```
meeting audio ─▶ record (loopback) ─▶ transcribe (Whisper) ─▶ notes (local LLM)
                     audio.wav            transcript.txt          notes.md
```

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
| `--no-notes` | off | Transcribe only, skip note generation |
| `--output` | `recordings` | Base directory for saved meetings |

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

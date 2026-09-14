"""Generate meeting notes from a transcript using a local LLM via Ollama.

Ollama runs an open-source model (default: llama3.2:3b) entirely on your
machine -- the transcript never leaves your computer. Ollama must be running;
it normally starts automatically after install and listens on localhost:11434.
"""

from __future__ import annotations

import ollama

DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a meticulous meeting-notes assistant. You are given \
the raw transcript of a meeting. Produce clear, well-structured notes in \
Markdown with exactly these sections:

## Summary
A 2-4 sentence overview of what the meeting was about.

## Key Points
- The main topics discussed, as concise bullets.

## Decisions
- Decisions that were made. Write "None recorded." if there were none.

## Action Items
- [ ] Each task, with the owner in **bold** if a name was mentioned, and any due date.
  Write "None recorded." if there were none.

Only use information present in the transcript. Do not invent names, dates, or \
tasks. Keep it concise and skimmable."""


def generate_notes(transcript_text: str, model: str = DEFAULT_MODEL) -> str:
    """Return Markdown meeting notes generated from the transcript text."""
    if not transcript_text.strip():
        return "_(No speech was transcribed, so no notes could be generated.)_"

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript_text}"},
        ],
        options={"temperature": 0.2},  # low temp -> factual, less embellishment
    )
    return response["message"]["content"].strip()

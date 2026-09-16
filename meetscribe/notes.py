"""Generate detailed meeting notes from a transcript using a local LLM via Ollama.

Ollama runs an open-source model (default: llama3.2:3b) entirely on your
machine -- the transcript never leaves your computer. Ollama must be running;
it normally starts automatically after install and listens on localhost:11434.

Long meetings don't fit in a model's context window, and Ollama silently
truncates anything that overflows -- which would make notes reflect only a
fraction of the meeting. To avoid that, long transcripts are summarized in a
map-reduce fashion: each chunk is summarized on its own (the "map" step), then
the section summaries are merged into one detailed set of notes (the "reduce"
step). This way the whole meeting is actually read.
"""

from __future__ import annotations

import ollama

DEFAULT_MODEL = "llama3.1:8b"

# Transcript characters per chunk (~4 chars/token, so ~2000 tokens/chunk).
_CHUNK_CHARS = 8000
# Context windows to request from Ollama (must be set explicitly or it defaults
# to a few thousand tokens and quietly truncates the input).
_MAP_CTX = 8192
_REDUCE_CTX = 16384
_NUM_PREDICT = 2048  # allow a long, detailed answer

# What the final notes should look like -- shared by the single-pass and reduce steps.
_FINAL_FORMAT = """Produce detailed, readable Markdown notes with these sections:

## Overview
A short paragraph (3-6 sentences) capturing what the meeting/lecture was about \
and its main thrust -- enough that someone who missed it understands the gist.

## Main Topics & Ideas
A `###` subsection for each major topic. Under each, write 2-4 full sentences \
explaining the idea, the reasoning, and any important details or examples -- not \
one-line bullets. The reader should understand *what was actually discussed*.

## Decisions Made
- Each decision that was reached. Write "None recorded." if there were none.

## Who Does What
- **Name** -- the task or responsibility they own, with a deadline if one was \
mentioned. Group by person. Write "No owners identified." if nobody was named.

## Pending To-Dos & Open Questions
- [ ] Outstanding tasks not yet assigned, and questions left unresolved. Write \
"None recorded." if there were none.

Be specific and use concrete details from the content. Do not invent names, \
dates, numbers, or tasks that are not supported by the material.

If lines are prefixed with a speaker ("You:" is the person keeping these notes; \
"Others:" is everyone else in the meeting), use that to attribute tasks and \
commitments correctly in "Who Does What" -- distinguish what *you* committed to \
from what others own."""

_MAP_SYSTEM = """You are summarizing ONE PART of a longer meeting/lecture \
transcript. Capture, in detail and using only this excerpt:
- the main ideas and topics discussed, with enough explanation to understand them,
- any decisions made,
- any tasks or action items, noting WHO is responsible and any deadline,
- any open questions or pending to-dos.
Write concise structured Markdown under the headings: Topics, Decisions, \
Action Items, Open Questions. Do not invent anything not in the excerpt."""

_REDUCE_SYSTEM = (
    "You are compiling the FINAL notes for a single meeting from summaries of its "
    "consecutive sections. Merge related points, remove duplicates, and keep the "
    "detail. " + _FINAL_FORMAT
)

_SINGLE_SYSTEM = (
    "You are a meticulous meeting-notes assistant. You are given the transcript "
    "of a meeting. " + _FINAL_FORMAT
)


def _chat(model: str, system: str, user: str, num_ctx: int) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={
            "temperature": 0.2,  # low temp -> factual, less embellishment
            "num_ctx": num_ctx,  # actually read the whole chunk (no silent truncation)
            "num_predict": _NUM_PREDICT,
            "repeat_penalty": 1.15,  # guard against runaway repetition loops
        },
    )
    return response["message"]["content"].strip()


def _chunk(text: str, limit: int) -> list[str]:
    """Split transcript into chunks of <= limit chars, on line boundaries."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.splitlines():
        if current and length + len(line) > limit:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def generate_notes(transcript_text: str, model: str = DEFAULT_MODEL) -> str:
    """Return detailed Markdown meeting notes generated from the transcript."""
    if not transcript_text.strip():
        return "_(No speech was transcribed, so no notes could be generated.)_"

    chunks = _chunk(transcript_text, _CHUNK_CHARS)

    # Short meeting: one pass straight to the final format.
    if len(chunks) <= 1:
        return _chat(
            model, _SINGLE_SYSTEM, f"Transcript:\n\n{transcript_text}", _REDUCE_CTX
        )

    # Long meeting: map each section, then reduce into one detailed document.
    partials: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"      notes: summarizing section {i}/{len(chunks)}...", flush=True)
        partials.append(
            _chat(model, _MAP_SYSTEM, f"Excerpt {i} of {len(chunks)}:\n\n{chunk}", _MAP_CTX)
        )

    combined = "\n\n---\n\n".join(
        f"Section {i} summary:\n{p}" for i, p in enumerate(partials, 1)
    )
    print("      notes: merging sections into final notes...", flush=True)
    return _chat(model, _REDUCE_SYSTEM, f"Section summaries:\n\n{combined}", _REDUCE_CTX)

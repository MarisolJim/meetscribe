#!/usr/bin/env bash
# Regenerate notes for every recorded meeting from its existing transcript
# (no re-transcribing). Useful after changing the notes prompt or model.
#
# Usage:
#     ./scripts/redo_all_notes.sh              # uses the default LLM
#     ./scripts/redo_all_notes.sh llama3.1:8b  # uses a specific LLM
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$PROJECT_DIR/.venv/Scripts/activate"
cd "$PROJECT_DIR"

LLM="${1:-llama3.2:3b}"

for d in recordings/*/; do
    if [ ! -f "$d/transcript.txt" ]; then
        echo "skip (no transcript yet): $d"
        continue
    fi
    echo ""
    echo "### Regenerating notes for: $d"
    python -m meetscribe process "$d" --notes-only --llm "$LLM"
done

echo ""
echo "Done - all notes regenerated with $LLM."

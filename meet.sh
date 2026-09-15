#!/usr/bin/env bash
# meetscribe launcher: activates the project's virtual environment and runs the
# CLI, so you don't have to cd into the project or activate the venv by hand.
#
# It resolves its own location, so it works no matter where you call it from:
#     ~/Documents/meetscribe/meet.sh record --title "Team Sync"
#
# Tip: add a shortcut to your ~/.bashrc so you can just type `meet ...`:
#     alias meet="$HOME/Documents/meetscribe/meet.sh"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the venv (Windows layout: .venv/Scripts).
if [ -f "$SCRIPT_DIR/.venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/Scripts/activate"
elif [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "meetscribe: could not find the virtual environment at $SCRIPT_DIR/.venv" >&2
    echo "Create it with:  python -m venv .venv && pip install -r requirements.txt" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
exec python -m meetscribe "$@"

#!/bin/zsh
set -e
cd "$(dirname "$0")"
PYTHON=$(command -v python3 || true)
if [[ -z "$PYTHON" ]]; then
  echo 'Python 3 is required. Install it from https://www.python.org/downloads/macos/'
  read '?Press Return to close.'
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then "$PYTHON" -m venv .venv; fi
if ! .venv/bin/python -c 'import rawpy, cv2, tifffile, numpy' 2>/dev/null; then
  .venv/bin/python -m pip install -r requirements.txt
fi
if ! command -v exiftool >/dev/null 2>&1; then
  echo 'ExifTool is required to verify Sony sequences. Install it from https://exiftool.org/'
  read '?Press Return to close.'
  exit 1
fi
.venv/bin/python app.py

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VIDEO_PATH="videos/Lapchole/Lapchole1.mp4"
VIDEO_FPS="15"

if [[ -z "$VIDEO_PATH" ]]; then
  echo "Set VIDEO_PATH in this script to a file (relative to repo root) or an absolute path." >&2
  exit 1
fi

if [[ "$VIDEO_PATH" =~ ^[0-9]+$ ]]; then
  SOURCE="$VIDEO_PATH"
else
  if [[ "$VIDEO_PATH" = /* ]]; then
    SOURCE="$VIDEO_PATH"
  else
    SOURCE="$ROOT_DIR/$VIDEO_PATH"
  fi
  if [[ ! -f "$SOURCE" ]]; then
    echo "Video not found: $SOURCE" >&2
    exit 1
  fi
fi

export VIDEO_SOURCE="$SOURCE"
export VIDEO_FPS="$VIDEO_FPS"

uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
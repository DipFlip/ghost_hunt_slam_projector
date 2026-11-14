#!/usr/bin/env bash
# Start the ROS bridge container, then the game on the projector (Ctrl-C or q quits).
# airy_slam's SLAM must be running: bash ../airy_slam/scripts/start_slam.sh
# GHOST_HEADLESS=1: no projector, one frame a second to out/frame.png.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker compose up -d
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec python3 -m ghost_hunt.game

#!/bin/bash
# Post-process one corrected (candidate filter 8) test run: frozen geometry snap,
# snap-off control (= frozen linker alone), and every frame-mAP convention.
# Usage: postprocess_c8.sh RUN_DIR [PYTHON]
set -u
D=$1; PY=${2:-python3}
cd "$(dirname "$0")/.." || exit 1
C=$D/tube_cache_test_ema_c8.pkl; F=$D/frames_test_ema.pkl
for f in "$C" "$F"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
echo "$(date -Is) start $D"
nice -n 10 $PY tools/snap_tubes_to_dense_geometry.py --tube-cache "$C" --frame-dump "$F" \
  --match-iou 0.3 --box-blend 0.75 --score-blend 0 --smooth-radius 2 \
  --out "$D/snap_frozen.json" > "$D/snap_frozen.log" 2>&1 || echo "snap frozen FAILED"
nice -n 10 $PY tools/snap_tubes_to_dense_geometry.py --tube-cache "$C" --frame-dump "$F" \
  --match-iou 0.3 --box-blend 0 --score-blend 0 --smooth-radius 0 \
  --out "$D/snap_control.json" > "$D/snap_control.log" 2>&1 || echo "snap control FAILED"
nice -n 10 $PY tools/score_frame_conventions.py --dump "$F" --out "$D/frame_conventions.json" \
  > "$D/frame_conventions.log" 2>&1 || echo "frame conventions FAILED"
echo "$(date -Is) done $D"

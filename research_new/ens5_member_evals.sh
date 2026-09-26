#!/bin/bash
# V2-ENS5: single-pass frozen-protocol test evaluations of the four held-out members (V2-WS2.0's already exists in
# FLIPTTA_RESULT/V2-WS2.0_test). KF64, TK1 and NOACTX start now; HO0GC starts when its training has finished.
cd "$(dirname "$0")/.." || exit 1
OUT=research_new/experiments/ENS5_RESULT
LINK="--moc_link_iou 0.45 --moc_tubelet_nms 0.6 --moc_top_k 10 --moc_split_gap 2 --moc_tube_nms 0.3"
run() {  # id exp gpu
  D=$OUT/$1; mkdir -p $D
  [ -f $D/frames_test_ema.pkl ] && [ -f $D/tube_cache_test_ema_c8.pkl ] && return 0
  CUDA_VISIBLE_DEVICES=$3 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 eval_tube_queries.py --config configs/yolost_${2}_evaltest.yaml \
    --checkpoint experiments/$2/ema_final.pt --clip_weighting hann_peak_norm --frame_map \
    --frame_dump $D/frames_test_ema.pkl --candidate_cache $D/tube_cache_test_ema_c8.pkl --min_length 8 $LINK \
    > $D/eval_test.log 2>&1
  status=$?
  echo "$(date +%FT%T) $1 test evaluation exit $status"
}
run V2HO-KF64-s17 v2ho_kf64_s17_30ep 0 &
run V2HO-TK1-s17 v2ho_tk1_s17_30ep 1 &
run V2HO-NOACTX-s17 v2ho_noactx_s17_30ep 2 &
until [ -f experiments/v2ho_ho0gc_s17_30ep/ema_final.pt ] && \
      ! pgrep -f "^([^ ]*/)?python3? train\.py --config configs/yolost_v2ho_ho0gc_s17_30ep\.yaml( |$)" > /dev/null; do
  sleep 60
done
run V2HO-HO0GC-s17 v2ho_ho0gc_s17_30ep 3 &
wait
echo "$(date +%FT%T) all member test evaluations finished"

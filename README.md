# YOLO-ST: code, configurations, protocol and ledger

This repository accompanies the paper "Decoupling Temporal Action Ownership and
Actor Geometry for Spatio-Temporal Action Localization". It holds
the training and evaluation code, the configuration of every model in the
experiment ledger, the held-out selection protocol, the ledger itself, and the
partition lists. No data, checkpoints or logs are included. `REPRODUCIBILITY.md`
maps every number in the paper to its configuration, command and ledger record.

## Layout

| Path | Contents |
|---|---|
| `train.py` | training entry point (`python train.py --config CONFIG`) |
| `eval_tube_queries.py` | UCF101-24 / JHMDB-21 inference: dense frame detections, tube-query candidates |
| `frame_map_protocol.py`, `video_map_protocol.py` | frame mAP (VOC, annotated frames) and MOC/ACT video mAP |
| `eval_cached_tube_protocol.py` | video mAP from a cached candidate file without re-running the model |
| `eval_ava.py`, `eval_multisports.py` | AVA v2.2 and MultiSports evaluation |
| `yolost/` | model (`model_videomae.py` is the reported model), losses, heads, clip weighting |
| `data/` | dataset loaders |
| `tools/` | protocol tools (below), split construction, the ledger CLI |
| `configs/` | every configuration registered in the ledger, plus evaluation variants |
| `research_new/V2HO_PROTOCOL_FREEZE.json` | frozen inference protocol (linker, clip weighting, snap) |
| `research_new/experiments/ledger.jsonl` | append-only experiment ledger |
| `research_new/experiments/*_FREEZE.json` | per-experiment registration records |
| `splits/` | video lists of every development / held-out partition |

## Environment

Python 3.9, PyTorch 2.8 (CUDA 12.8); see `requirements.txt`. The frozen frame
pyramid loads the COCO-pretrained `yolo11l.pt` through `ultralytics`; VideoMAE-L
is `MCG-NJU/videomae-large-finetuned-kinetics` from the Hugging Face hub.

## Data

UCF101-24 uses the corrected annotations of Singh et al. (ROAD,
`UCF101v2-GT.pkl`) under `data/ucf24/UCF101_v2/`; JHMDB-21 uses `JHMDB-GT.pkl`
under `data/ucf24/JHMDB/`; AVA v2.2 and AVA-Kinetics follow the official
releases under `data/ava/` and `data/ava_kinetics/`; MultiSports under
`data/multisports/`. `eval_ava.py` loads the official ActivityNet AVA
evaluation code from `references/YOWOv3/evaluator/Evaluation/`, which is not
redistributed here.

Partitions:

* UCF101-24 held-out groups: `python tools/make_v2ho_splits.py` writes the
  training list without capture groups g22-g25 and the *tune* (g24/g25) and
  *confirm* (g22/g23) annotation files. `splits/ucf101_24_v2ho_partitions.json`
  lists the resulting videos.
* JHMDB-21, AVA and MultiSports development partitions: `tools/make_dev_splits.py`
  (deterministic, class-stratified); the resulting lists are in `splits/`.

## Reproducing the UCF101-24 numbers

1. Motion trunk: 150 epochs on split-1 train in three resumed stages
   (`configs/yolost_phase3a_boundary.yaml`, `yolost_phase3a_boundary_100ep.yaml`,
   `yolost_phase3a_150ep.yaml`; see `REPRODUCIBILITY.md`). The reported model loads
   it frozen from `experiments/phase3a_150ep/final.pt`.
2. Reported model: `python train.py --config configs/yolost_v2_ws2_0_tubequeries_30ep.yaml`.
   Held-out control seeds: `configs/yolost_v2ho_ho0_s{17,29,41}_30ep.yaml`;
   module ablations: `configs/yolost_v2ho_m{1,2,3,4}_*_s17_30ep.yaml`.
3. Test inference (one pass writes the frame dump and the tube candidates):

   ```
   python eval_tube_queries.py --config CONFIG --checkpoint EXP/ema_final.pt \
     --clip_weighting hann_peak_norm --frame_map --frame_dump OUT/frames_test_ema.pkl \
     --candidate_cache OUT/tube_cache_test_ema_c8.pkl --min_length 8 \
     --moc_link_iou 0.45 --moc_tubelet_nms 0.6 --moc_top_k 10 --moc_split_gap 2 --moc_tube_nms 0.3
   ```

   For held-out models use the `_evaltest.yaml` variant of the config, which
   points the evaluation at split-1 test.
4. Protocol scoring: `bash research_new/postprocess_c8.sh OUT` runs
   `tools/snap_tubes_to_dense_geometry.py` with the frozen snap (match IoU 0.3,
   blend 0.75, smoothing radius 2) and with the snap disabled, both on the frozen
   linker (0.45 / 0.6 / 10 / 16 / 2, tube NMS 0.3), and
   `tools/score_frame_conventions.py`, which scores the frame dump under every
   frame-mAP convention in Table 1 of the paper.

`--min_length` of `eval_tube_queries.py` is both the candidate filter and the
length used by the linker inside that script. The protocol filters candidates
at 8 and links at 16, so the video metrics that `eval_tube_queries.py` prints
after a `--min_length 8` run are not the reported ones; use step 4.

`train.backward_loss_divisor: 1` in the configs sums the two accumulated
micro-batch losses instead of averaging them. AdamW is invariant to that scale up
to its epsilon, but gradient clipping (1.0) acts on the summed gradient.

## Ledger

Every row of `research_new/experiments/ledger.jsonl` is one JSON event:
`register` (configuration, parent, control, frozen one-variable change, gate),
`result` (tag, partition, evaluator, metrics), `verdict`, or `note`.
`python tools/experiment_ledger.py show --id V2-WS2.0` prints one experiment;
`query` filters them. Partitions name the evaluation set (`ucf-test`,
`ucf-v2ho-g24g25` = tune, `ucf-v2ho-g22g23` = confirm, `jhmdb-test-splitN`,
`ava-official-val-64videos`, `multisports-official-val-555videos`, and the
development partitions); evaluators name the exact scoring rule.

The held-out runs KF64, TK1 and NOACTX, and their control HO0GC, were trained
with their registered configurations plus two memory-only flags
(`configs/*_gc.yaml`, `configs/yolost_v2ho_ho0gc_s17_30ep.yaml`), which
recompute activations in the backward pass and leave the computation unchanged.

The test-time variants of Table 1 (+flip and Ens.) are reproduced with
`eval_tube_queries.py --hflip`, `tools/fuse_flip_views.py` and
`tools/fuse_model_views.py`, driven by `research_new/fliptta_frame_pipeline.py`
and `research_new/ens5_finish.py`; every-frame numbers under ROAD's exact code
come from `tools/road_exact_frame_map.py` (ROAD's Python script) and
`tools/road_matlab_frame_map.py` (a port of ROAD's MATLAB `frameAp.m`).

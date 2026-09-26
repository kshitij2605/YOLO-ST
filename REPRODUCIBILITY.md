# Reproducing the paper

This file maps every result in the paper to the configuration, command, and
ledger record that produced it. Numbers quoted here are the ones in the paper;
GPU nondeterminism can move a re-run by a few hundredths of a point.

## 0. Setup

```
pip install -r requirements.txt        # Python 3.9, PyTorch 2.8 + CUDA 12.8
```

* Pretrained weights: VideoMAE-L is `MCG-NJU/videomae-large-finetuned-kinetics`
  (Hugging Face hub, fetched automatically); the frozen frame pyramid loads
  Ultralytics' COCO `yolo11l.pt` from `pretrained/yolo11l.pt`; the frozen motion
  trunk is trained in step 2.1.
* Memory: a UCF101-24 model trains on one 96 GB GPU. To fit in about 40 GB, add
  `backbone_gradient_checkpointing: true` and `class_ctx_grad_checkpoint: true`
  under `model:` (see `configs/*_gc.yaml`); they recompute activations and do
  not change the computation.

## 1. Data

| Dataset | Location expected by the configs | Source |
|---|---|---|
| UCF101-24 | `data/ucf24/UCF101_v2/rgb-images/<video>/<frame:05d>.jpg`, `UCF101v2-GT.pkl` | frames and the corrected annotations of Singh et al. (2017), as packaged by ACT/MOC: https://github.com/gurkirt/corrected-UCF101-Annots, https://github.com/MCG-NJU/MOC-Detector |
| JHMDB-21 | `data/ucf24/JHMDB/` with `JHMDB-GT.pkl` | http://jhmdb.is.tue.mpg.de/, packaged as in MOC |
| AVA v2.2, AVA-Kinetics | `data/ava/`, `data/ava_kinetics/` | https://research.google.com/ava/ |
| MultiSports | `data/multisports/` | https://github.com/MCG-NJU/MultiSports |

Partitions (lists of the resulting videos are in `splits/`):

* UCF101-24 held-out groups: `python tools/make_v2ho_splits.py` writes
  `UCF101v2-GT-v2ho-tune-g24g25.pkl` (tune) and `UCF101v2-GT-v2ho-confirm-g22g23.pkl`
  (confirm) next to `UCF101v2-GT.pkl`; compare with
  `splits/ucf101_24_v2ho_partitions.json`.
* JHMDB-21, AVA and MultiSports development partitions: `python tools/make_dev_splits.py`.

## 2. Training

All runs: `python train.py --config CONFIG` (add `--resume CKPT` where stated).
Checkpoints land in the config's `output.exp_dir`; the evaluated weights are
`ema_final.pt`.

### 2.1 Frozen motion trunk (UCF101-24 split-1 train, 150 epochs in three stages)

```
python train.py --config configs/yolost_phase3a_boundary.yaml                     # epochs 1-50
python train.py --config configs/yolost_phase3a_boundary_100ep.yaml \
    --resume experiments/phase3a_boundary/final.pt                               # to epoch 100
python train.py --config configs/yolost_phase3a_150ep.yaml                        # to epoch 150 (resume set in the config)
```

The reported models load `experiments/phase3a_150ep/final.pt` frozen.

### 2.2 UCF101-24 models

| Paper item | Ledger id | Config |
|---|---|---|
| Reported model (Table 1, video mAP) | `V2-WS2.0` | `configs/yolost_v2_ws2_0_tubequeries_30ep.yaml` |
| Held-out control and seeds (Section 4.2, Table 2) | `V2HO-HO0-s17`, `-s29`, `-s41` | `configs/yolost_v2ho_ho0_s{17,29,41}_30ep.yaml` |
| Module ablations (Table 2) | `V2HO-M1-s17` ... `V2HO-M4-s17` | `configs/yolost_v2ho_m{1,2,3,4}_*_s17_30ep.yaml` |
| VideoMAE-H, trainable keyframe stream (Table 2) | `V2HO-BBH-s17`, `V2HO-A1a-s17`, `V2HO-A1b-s17` | `configs/yolost_v2ho_bbh_*`, `yolost_v2ho_a1a_*`, `yolost_v2ho_a1b_*` |
| Action-free clips, boundary negatives (Table 2, App. B) | `V2HO-E3-s17`, `V2HO-BN*-s17` | `configs/yolost_v2ho_e3_*`, `yolost_v2ho_bn*_*` |
| Temporal-processing variants and their control (App. B) | `V2HO-KF64-s17`, `V2HO-TK1-s17`, `V2HO-NOACTX-s17`, `V2HO-HO0GC-s17` | `configs/yolost_v2ho_{kf64,tk1,noactx}_s17_30ep_gc.yaml`, `yolost_v2ho_ho0gc_s17_30ep.yaml` |

Held-out models are evaluated with the same config (tune), its `_evalconfirm`
variant (confirm) or its `_evaltest` variant (split-1 test).

### 2.3 Other benchmarks

| Paper item | Ledger id | Config |
|---|---|---|
| JHMDB-21, Table 3 | `JFULL-split{1,2,3}-s17` | `configs/yolost_jhmdb_jfull_split{1,2,3}_s17_40ep.yaml` |
| JHMDB-21 288x288 variant, App. B | `JR288-split{1,2,3}-s17` | `configs/yolost_jhmdb_jr288_split*_s17_40ep.yaml` |
| AVA v2.2, AVA subset, Table 4 | `AVA-F0-s17` | `configs/yolost_ava_f0_full_s17_6ep.yaml` |
| AVA v2.2, + AVA-Kinetics, Table 4 and App. C | `AVA-KFULL-s17` | `configs/yolost_ava_kfull_s17_6ep.yaml` |
| MultiSports 224, 24 epochs, Table 5 | `MS-E24-s17` | `configs/yolost_ms_e24_224_s17_24ep.yaml` |
| MultiSports 288x512, 12 epochs, Table 5 | `MS-R2E12-s17` | `configs/yolost_ms_r2e12_288x512_s17_12ep.yaml` |

## 3. UCF101-24 evaluation (Table 1, Table 2, Section 4.2)

Inference writes the frame dump and the tube candidates in one pass with the
frozen settings of `research_new/V2HO_PROTOCOL_FREEZE.json`:

```
python eval_tube_queries.py --config CONFIG --checkpoint EXP/ema_final.pt \
  --clip_weighting hann_peak_norm --frame_map --frame_dump OUT/frames_test_ema.pkl \
  --candidate_cache OUT/tube_cache_test_ema_c8.pkl --min_length 8 \
  --moc_link_iou 0.45 --moc_tubelet_nms 0.6 --moc_top_k 10 --moc_split_gap 2 --moc_tube_nms 0.3
```

Scoring:

| Table 1 entry | Command | Reported checkpoint |
|---|---|---|
| Every frame, VOC, ROAD MATLAB | `python tools/road_matlab_frame_map.py OUT/frames_test_ema.pkl` (key `reported`) | 88.47 |
| Every frame, VOC, ROAD Python | `python tools/road_exact_frame_map.py OUT/frames_test_ema.pkl` (line "ROAD exact (exclusive IoU, raw float boxes)") | 89.30 |
| Every frame trapezoid; annotated VOC; annotated trapezoid; YOWO-family (and YOWO list 93.32) | `python tools/score_frame_conventions.py --dump OUT/frames_test_ema.pkl --out conv.json` (keys `allframe_trapz`, `annotated_voc`, `annotated_trapz`, `yowoformer_style`, `yowoformer_exact`, `yowo_list_voc`) | 88.92, 93.21, 92.99, 95.10 (95.30) |
| Video AP20 / AP50 / AP50:95 (frozen snap) | `bash research_new/postprocess_c8.sh OUT` (runs `tools/snap_tubes_to_dense_geometry.py` with match IoU 0.3, blend 0.75, smoothing 2, and with the snap off) | 89.82 / 73.39 / 34.97 |

`--min_length` of `eval_tube_queries.py` is both the candidate filter (8) and the
length its internal linker uses; the protocol links at 16, so the video numbers
that `eval_tube_queries.py` prints are not the reported ones: use the snap script.

Test-time variants (Table 1, +flip and Ens.):

* +flip: run the inference above a second time with `--hflip`, then
  `python tools/fuse_flip_views.py --dumps PLAIN FLIPPED --caches PLAIN FLIPPED --out-dump F --out-cache C`
  and score `F` as above (frame metrics only). Driver:
  `research_new/fliptta_frame_pipeline.py`; ledger `V2HO-FLIPTTA-FRAME`.
* Ens.: single-pass test dumps of `V2-WS2.0`, `V2HO-HO0GC-s17`, `V2HO-KF64-s17`,
  `V2HO-TK1-s17` and `V2HO-NOACTX-s17` (each with its own config; the held-out
  ones with their `_evaltest` variant), fused by
  `python tools/fuse_model_views.py --dumps D1 ... D5 --caches C1 ... C5 --out-dump F --out-cache C`.
  Driver: `research_new/ens5_member_evals.sh` then `research_new/ens5_finish.py`;
  ledger `V2-ENS5`.

Held-out numbers (Table 2 Tune / Confirm) use the same commands on the tune and
confirm annotation files; the selection pipelines that produced the rejected
rows are `research_new/e3_pipeline.py`, `bn_pipeline.py`, `tr_pipeline_v3.py`,
`tr_pipeline_v4.py`, `fliptta_pipeline.py` and `swa_pipeline.py`.

## 4. Other benchmarks

* JHMDB-21 (Table 3): frame mAP from `eval_tube_queries.py` frame dumps scored
  on all frames; video mAP by linking dense detections
  (`tools/link_dense_frame_tubes.py`) with the setting frozen on the split-1
  development partition.
* AVA v2.2 (Table 4, App. C): `eval_ava.py` with the official ActivityNet AVA
  evaluator, which is not redistributed: place the `Evaluation/` code of
  https://github.com/activitynet/ActivityNet at
  `references/YOWOv3/evaluator/Evaluation/`, the path `eval_ava.py` loads. It is
  run on the 34 development videos and on the official 64-video validation set.
* MultiSports (Table 5): `eval_multisports.py`, our re-implementation of the
  official `evaluate_multisports.py`; video AP by linking dense detections with
  `tools/link_dense_frame_tubes.py` pinned to the development winner
  (per-frame 5, link IoU 0.3, maximum gap 2 for the 288x512 model and 5 for the
  224 model, minimum length 8).

## 5. The ledger

`research_new/experiments/ledger.jsonl` records every registration, result,
verdict and note; `python tools/experiment_ledger.py show --id ID` prints one
experiment with all its results, and `query` filters them. The exact settings
of each evaluation (partition, evaluator, linker) are in the result rows and
notes of the ids above.

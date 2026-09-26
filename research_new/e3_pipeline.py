"""Post-training pipeline for V2HO-E3-s17 (held-out gate) and V2-WS2.0E3 (test).

Stage 1  wait for experiments/v2ho_e3_s17_30ep/ema_final.pt and the training process
         to exit; run eval_tube_queries.py on tune (g24/g25) and confirm (g22/g23)
         with the frozen protocol (clip weighting hann_peak_norm, candidate filter 8),
         then the frozen linker with the snap (and snap-off control) and every
         frame-mAP convention. Record the held-out results in the ledger.
Stage 2  apply the gate registered in V2HO-E3-s17_FREEZE.json against V2HO-HO0-s17
         scored by the same scorer; record the verdict.
Stage 3  only if the gate passed: wait for experiments/v2_ws2_0e3_30ep/ema_final.pt,
         evaluate split-1 test once, score, record. If the gate failed, record that
         V2-WS2.0E3 is not evaluated on test.

Every stage writes a marker so a restart never repeats a finished stage.
Usage: nohup python3 research_new/e3_pipeline.py > research_new/experiments/V2HO-E3-s17_RESULT/pipeline.log 2>&1 &
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable
HO_ID, FULL_ID = "V2HO-E3-s17", "V2-WS2.0E3"
HO_CFG = "configs/yolost_v2ho_e3_s17_30ep.yaml"
HO_CFG_CONFIRM = "configs/yolost_v2ho_e3_s17_30ep_evalconfirm.yaml"
FULL_CFG = "configs/yolost_v2_ws2_0e3_30ep.yaml"
HO_CKPT = "experiments/v2ho_e3_s17_30ep/ema_final.pt"
FULL_CKPT = "experiments/v2_ws2_0e3_30ep/ema_final.pt"
OUT = "research_new/experiments/V2HO-E3-s17_RESULT"
FULL_OUT = "research_new/experiments/V2-WS2.0E3_RESULT"
CONTROL_DIR = OUT + "/control_ho0s17"
CONTROL_VIDEO_AP50_CONFIRM = 78.53  # V2HO-HO0-s17 ema_final-video-b2snap-confirm
LINKER = ["--moc_link_iou", "0.45", "--moc_tubelet_nms", "0.6", "--moc_top_k", "10",
          "--moc_split_gap", "2", "--moc_tube_nms", "0.3"]


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), message, flush=True)


def run(cmd, logfile):
    log("run: " + " ".join(cmd))
    with open(logfile, "w") as handle:
        code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT)
    if code != 0:
        log("FAILED (%d): see %s" % (code, logfile))
        raise SystemExit(1)


def training_running(config):
    out = subprocess.run(["pgrep", "-f", "train.py --config " + config],
                         capture_output=True, text=True).stdout.split()
    return [pid for pid in out if pid != str(os.getpid())]


def wait_for(ckpt, config):
    while True:
        running = bool(training_running(config))
        if os.path.isfile(ckpt) and not running:
            break
        if not running:
            time.sleep(600)  # re-check once before declaring a crash
            if not training_running(config) and not os.path.isfile(ckpt):
                log("ERROR: training of %s is not running and %s does not exist" % (config, ckpt))
                raise SystemExit(1)
            continue
        time.sleep(300)
    time.sleep(60)  # let the final torch.save flush
    log("checkpoint ready: " + ckpt)


def free_gpu(preferred):
    while True:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        used = {int(r.split(",")[0]): int(r.split(",")[1]) for r in rows}
        for gpu in [preferred] + sorted(used):
            if used.get(gpu, 10**9) < 2000:
                return gpu
        time.sleep(300)


def evaluate(config, ckpt, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    frames = "%s/frames_%s_ema.pkl" % (outdir, tag)
    cache = "%s/tube_cache_%s_ema_c8.pkl" % (outdir, tag)
    if not (os.path.isfile(frames) and os.path.isfile(cache)):
        gpu = free_gpu(0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        cmd = [PY, "eval_tube_queries.py", "--config", config, "--checkpoint", ckpt,
               "--clip_weighting", "hann_peak_norm", "--frame_map", "--frame_dump", frames,
               "--candidate_cache", cache, "--min_length", "8"] + LINKER
        log("GPU %d: %s" % (gpu, " ".join(cmd)))
        with open("%s/eval_%s.log" % (outdir, tag), "w") as handle:
            if subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env):
                raise SystemExit("evaluation failed: %s/eval_%s.log" % (outdir, tag))
    for name, blend, smooth in (("snap_frozen", "0.75", "2"), ("snap_control", "0", "0")):
        target = "%s/%s_%s.json" % (outdir, name, tag)
        if not os.path.isfile(target):
            run(["nice", "-n", "10", PY, "tools/snap_tubes_to_dense_geometry.py", "--tube-cache", cache,
                 "--frame-dump", frames, "--match-iou", "0.3", "--box-blend", blend,
                 "--score-blend", "0", "--smooth-radius", smooth, "--out", target],
                "%s/%s_%s.log" % (outdir, name, tag))
    conv = "%s/frame_conventions_%s.json" % (outdir, tag)
    if not os.path.isfile(conv):
        run(["nice", "-n", "10", PY, "tools/score_frame_conventions.py", "--dump", frames,
             "--out", conv], "%s/frame_conventions_%s.log" % (outdir, tag))
    snap = json.load(open("%s/snap_frozen_%s.json" % (outdir, tag)))["rows"][0]
    ctrl = json.load(open("%s/snap_control_%s.json" % (outdir, tag)))["rows"][0]
    return json.load(open(conv)), snap, ctrl


def ledger(*args):
    out = subprocess.run([PY, "tools/experiment_ledger.py"] + list(args),
                         capture_output=True, text=True)
    log("ledger: " + (out.stdout.strip() or out.stderr.strip()))


def record(run_id, partition, prefix, conv, snap, ctrl):
    ledger("result", "--id", run_id, "--tag", prefix + "-video-snap", "--partition", partition,
           "--evaluator", "moc-act-geometry-snap-c8", "--metric", "ap20=%.2f" % snap["ap20"],
           "--metric", "ap50=%.2f" % snap["ap50"], "--metric", "strict=%.2f" % snap["strict"])
    ledger("result", "--id", run_id, "--tag", prefix + "-video-b0", "--partition", partition,
           "--evaluator", "moc-act-b0linker-c8", "--metric", "ap20=%.2f" % ctrl["ap20"],
           "--metric", "ap50=%.2f" % ctrl["ap50"], "--metric", "strict=%.2f" % ctrl["strict"])
    for key, evaluator in (("annotated_voc", "moc-act-corrected-frame-hann_peak_norm"),
                           ("yowo_list_voc", "yowo-frame-hann_peak_norm"),
                           ("annotated_trapz", "moc-act-annotated-trapz-hann_peak_norm"),
                           ("allframe_voc", "road-allframe-voc-hann_peak_norm"),
                           ("allframe_trapz", "moc-act-allframe-trapz-hann_peak_norm"),
                           ("yowoformer_style", "yowoformer-released-reimpl-hann_peak_norm"),
                           ("yowoformer_exact", "yowoformer-released-exact-hann_peak_norm")):
        ledger("result", "--id", run_id, "--tag", prefix + "-frame", "--partition", partition,
               "--evaluator", evaluator, "--metric", "frame=%.2f" % conv[key])


os.makedirs(OUT, exist_ok=True)
gate_file = OUT + "/gate.json"
if not os.path.isfile(gate_file):
    log("stage 1: waiting for " + HO_CKPT)
    wait_for(HO_CKPT, HO_CFG)
    results = {}
    for tag, config, partition in (("tune", HO_CFG, "ucf-v2ho-g24g25"),
                                   ("confirm", HO_CFG_CONFIRM, "ucf-v2ho-g22g23")):
        conv, snap, ctrl = evaluate(config, HO_CKPT, OUT, tag)
        results[tag] = {"conv": conv, "snap": snap, "ctrl": ctrl, "partition": partition}
    if not os.path.isfile(OUT + "/.recorded_heldout"):
        for tag, r in results.items():
            record(HO_ID, r["partition"], "ema_final-%s-c8" % tag, r["conv"], r["snap"], r["ctrl"])
        open(OUT + "/.recorded_heldout", "w").write("recorded\n")

    control = {tag: json.load(open("%s/frame_conventions_%s.json" % (CONTROL_DIR, tag)))
               for tag in ("tune", "confirm")}
    checks = {
        "confirm_allframe_+1.0": results["confirm"]["conv"]["allframe_voc"] >= control["confirm"]["allframe_voc"] + 1.0,
        "tune_allframe_+0.5": results["tune"]["conv"]["allframe_voc"] >= control["tune"]["allframe_voc"] + 0.5,
        "tune_annotated_-0.5": results["tune"]["conv"]["annotated_voc"] >= control["tune"]["annotated_voc"] - 0.5,
        "confirm_annotated_-0.5": results["confirm"]["conv"]["annotated_voc"] >= control["confirm"]["annotated_voc"] - 0.5,
        "confirm_video_ap50_-2.0": results["confirm"]["snap"]["ap50"] >= CONTROL_VIDEO_AP50_CONFIRM - 2.0,
    }
    passed = all(checks.values())
    summary = {
        "passed": passed, "checks": checks,
        "e3": {tag: {"annotated_voc": r["conv"]["annotated_voc"], "allframe_voc": r["conv"]["allframe_voc"],
                     "video_snap": [r["snap"]["ap20"], r["snap"]["ap50"], r["snap"]["strict"]]}
               for tag, r in results.items()},
        "control": {tag: {"annotated_voc": c["annotated_voc"], "allframe_voc": c["allframe_voc"]}
                    for tag, c in control.items()},
        "control_video_ap50_confirm": CONTROL_VIDEO_AP50_CONFIRM,
    }
    json.dump(summary, open(gate_file, "w"), indent=2, sort_keys=True)
    text = ("Pre-registered gate %s. E3 every-frame VOC tune %.2f (control %.2f), confirm %.2f "
            "(control %.2f); annotated tune %.2f (%.2f), confirm %.2f (%.2f); confirm video snap "
            "%.2f/%.2f/%.2f (control AP50 %.2f). Checks: %s." % (
                "PASSED" if passed else "FAILED",
                results["tune"]["conv"]["allframe_voc"], control["tune"]["allframe_voc"],
                results["confirm"]["conv"]["allframe_voc"], control["confirm"]["allframe_voc"],
                results["tune"]["conv"]["annotated_voc"], control["tune"]["annotated_voc"],
                results["confirm"]["conv"]["annotated_voc"], control["confirm"]["annotated_voc"],
                results["confirm"]["snap"]["ap20"], results["confirm"]["snap"]["ap50"],
                results["confirm"]["snap"]["strict"], CONTROL_VIDEO_AP50_CONFIRM,
                ", ".join("%s %s" % (k, "ok" if v else "FAIL") for k, v in checks.items())))
    ledger("verdict", "--id", HO_ID, "--verdict", "promoted" if passed else "rejected",
           "--rationale", text)
    log("stage 2: " + text)

summary = json.load(open(gate_file))
if not summary["passed"]:
    marker = FULL_OUT + "/.not_tested"
    os.makedirs(FULL_OUT, exist_ok=True)
    if not os.path.isfile(marker):
        ledger("note", "--id", FULL_ID, "--text",
               "Not evaluated on test: V2HO-E3-s17 failed its pre-registered held-out gate "
               "(see its verdict). V2-WS2.0 stays the reported UCF101-24 model.")
        open(marker, "w").write("gate failed\n")
    log("stage 3 skipped: gate failed")
    raise SystemExit(0)

marker = FULL_OUT + "/.recorded_test"
if os.path.isfile(marker):
    log("stage 3 already recorded")
    raise SystemExit(0)
log("stage 3: gate passed; waiting for " + FULL_CKPT)
wait_for(FULL_CKPT, FULL_CFG)
conv, snap, ctrl = evaluate(FULL_CFG, FULL_CKPT, FULL_OUT, "test")
record(FULL_ID, "ucf-test", "ema_final-test-c8", conv, snap, ctrl)
ledger("note", "--id", FULL_ID, "--text",
       "Single test evaluation after V2HO-E3-s17 passed its gate: frame %.2f annotated, %.2f every "
       "frame (VOC); video with the frozen snap %.2f/%.2f/%.2f. Per the registered test-use clause "
       "this model replaces V2-WS2.0 as the reported UCF101-24 model." % (
           conv["annotated_voc"], conv["allframe_voc"], snap["ap20"], snap["ap50"], snap["strict"]))
open(marker, "w").write("recorded\n")
log("stage 3 done")

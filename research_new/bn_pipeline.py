"""Post-training pipeline for the boundary-negative runs (patch 0031).

Stage 1  as each held-out variant finishes, evaluate tune and confirm with the frozen
         protocol, score (snap, snap-off control, every frame-mAP convention) and
         record in the ledger.
Stage 2  pre-registered selection: among variants whose tune annotated VOC is >=
         control - 0.5, the one with the highest tune every-frame VOC is gated on
         confirm (same gate as V2HO-E3-s17). Verdicts go to the ledger.
Stage 3  if the selected variant passes: its full-data model (V2-WS2.0BN16w4 is
         already training; any other variant's is derived from V2-WS2.0, registered
         and trained now) is evaluated on split-1 test once and recorded. If it
         fails, nothing is evaluated on test.
Markers make every stage restart-safe.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable
OUT = "research_new/experiments/BN_RESULT"
CONTROL_DIR = "research_new/experiments/V2HO-E3-s17_RESULT/control_ho0s17"
CONTROL_VIDEO_AP50_CONFIRM = 78.53
VARIANTS = {  # id: (exp dir name, window, weight)
    "V2HO-BN16w4-s17": ("v2ho_bn16w4_s17_30ep", 16, 4.0),
    "V2HO-BN8w8-s17": ("v2ho_bn8w8_s17_30ep", 8, 8.0),
    "V2HO-BN32w4-s17": ("v2ho_bn32w4_s17_30ep", 32, 4.0),
}
FULL_OF = {"V2HO-BN16w4-s17": ("V2-WS2.0BN16w4", "v2_ws2_0bn16w4_30ep"),
           "V2HO-BN8w8-s17": ("V2-WS2.0BN8w8", "v2_ws2_0bn8w8_30ep"),
           "V2HO-BN32w4-s17": ("V2-WS2.0BN32w4", "v2_ws2_0bn32w4_30ep")}
LINKER = ["--moc_link_iou", "0.45", "--moc_tubelet_nms", "0.6", "--moc_top_k", "10",
          "--moc_split_gap", "2", "--moc_tube_nms", "0.3"]


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), message, flush=True)


def ledger(*args):
    out = subprocess.run([PY, "tools/experiment_ledger.py"] + list(args), capture_output=True, text=True)
    log("ledger: " + (out.stdout.strip() or out.stderr.strip())[:300])


def training_running(config):
    pattern = "train.py --config " + config
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [pid for pid in out if pid != str(os.getpid())]


def done_training(ckpt, config):
    """True when trained, False while training, raises if training died without a checkpoint."""
    running = bool(training_running(config))
    if os.path.isfile(ckpt) and not running:
        return True
    if not running and not os.path.isfile(ckpt):
        time.sleep(600)
        if not training_running(config) and not os.path.isfile(ckpt):
            raise SystemExit("ERROR: %s not running and %s missing" % (config, ckpt))
    return False


def free_gpu():
    while True:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        for row in rows:
            index, used = (int(x) for x in row.split(","))
            if used < 2000:
                return index
        time.sleep(300)


def run(cmd, logfile, env=None):
    log("run: " + " ".join(cmd))
    with open(logfile, "w") as handle:
        if subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env):
            raise SystemExit("FAILED: see " + logfile)


def evaluate(config, ckpt, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    frames, cache = "%s/frames_%s_ema.pkl" % (outdir, tag), "%s/tube_cache_%s_ema_c8.pkl" % (outdir, tag)
    if not (os.path.isfile(frames) and os.path.isfile(cache)):
        gpu = free_gpu()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        run([PY, "eval_tube_queries.py", "--config", config, "--checkpoint", ckpt, "--clip_weighting",
             "hann_peak_norm", "--frame_map", "--frame_dump", frames, "--candidate_cache", cache,
             "--min_length", "8"] + LINKER, "%s/eval_%s.log" % (outdir, tag), env)
    for name, blend, smooth in (("snap_frozen", "0.75", "2"), ("snap_control", "0", "0")):
        target = "%s/%s_%s.json" % (outdir, name, tag)
        if not os.path.isfile(target):
            run(["nice", "-n", "10", PY, "tools/snap_tubes_to_dense_geometry.py", "--tube-cache", cache,
                 "--frame-dump", frames, "--match-iou", "0.3", "--box-blend", blend, "--score-blend", "0",
                 "--smooth-radius", smooth, "--out", target], "%s/%s_%s.log" % (outdir, name, tag))
    conv = "%s/frame_conventions_%s.json" % (outdir, tag)
    if not os.path.isfile(conv):
        run(["nice", "-n", "10", PY, "tools/score_frame_conventions.py", "--dump", frames, "--out", conv],
            "%s/frame_conventions_%s.log" % (outdir, tag))
    return (json.load(open(conv)), json.load(open("%s/snap_frozen_%s.json" % (outdir, tag)))["rows"][0],
            json.load(open("%s/snap_control_%s.json" % (outdir, tag)))["rows"][0])


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
control = {tag: json.load(open("%s/frame_conventions_%s.json" % (CONTROL_DIR, tag))) for tag in ("tune", "confirm")}

# Stage 1: evaluate each held-out variant as soon as it has finished training.
results = {}
pending = dict(VARIANTS)
log("stage 1: waiting for %d held-out variants" % len(pending))
while pending:
    for run_id, (exp, window, weight) in list(pending.items()):
        config = "configs/yolost_%s.yaml" % exp
        ckpt = "experiments/%s/ema_final.pt" % exp
        try:
            if not done_training(ckpt, config):
                continue
        except SystemExit as error:  # this variant crashed; keep going with the others
            log("%s dropped: %s" % (run_id, error))
            ledger("note", "--id", run_id, "--text",
                   "Training ended without ema_final.pt; excluded from the pre-registered selection.")
            del pending[run_id]
            continue
        time.sleep(60)
        outdir = "%s/%s" % (OUT, run_id)
        res = {}
        for tag, cfg, partition in (("tune", config, "ucf-v2ho-g24g25"),
                                    ("confirm", "configs/yolost_%s_evalconfirm.yaml" % exp, "ucf-v2ho-g22g23")):
            res[tag] = evaluate(cfg, ckpt, outdir, tag) + (partition,)
        if not os.path.isfile(outdir + "/.recorded"):
            for tag, (conv, snap, ctrl, partition) in res.items():
                record(run_id, partition, "ema_final-%s-c8" % tag, conv, snap, ctrl)
            open(outdir + "/.recorded", "w").write("recorded\n")
        results[run_id] = res
        log("%s evaluated: tune every-frame %.2f annotated %.2f | confirm every-frame %.2f annotated %.2f" % (
            run_id, res["tune"][0]["allframe_voc"], res["tune"][0]["annotated_voc"],
            res["confirm"][0]["allframe_voc"], res["confirm"][0]["annotated_voc"]))
        del pending[run_id]
    if pending:
        time.sleep(300)

# Stage 2: selection on tune, gate on confirm.
selection_file = OUT + "/selection.json"
if not os.path.isfile(selection_file):
    eligible = {rid: r for rid, r in results.items()
                if r["tune"][0]["annotated_voc"] >= control["tune"]["annotated_voc"] - 0.5}
    ranking = sorted(eligible, key=lambda rid: -eligible[rid]["tune"][0]["allframe_voc"])
    selected = ranking[0] if ranking else None
    checks, passed = {}, False
    if selected:
        r = results[selected]
        checks = {
            "confirm_allframe_+1.0": r["confirm"][0]["allframe_voc"] >= control["confirm"]["allframe_voc"] + 1.0,
            "tune_allframe_+0.5": r["tune"][0]["allframe_voc"] >= control["tune"]["allframe_voc"] + 0.5,
            "tune_annotated_-0.5": r["tune"][0]["annotated_voc"] >= control["tune"]["annotated_voc"] - 0.5,
            "confirm_annotated_-0.5": r["confirm"][0]["annotated_voc"] >= control["confirm"]["annotated_voc"] - 0.5,
            "confirm_video_ap50_-2.0": r["confirm"][1]["ap50"] >= CONTROL_VIDEO_AP50_CONFIRM - 2.0,
        }
        passed = all(checks.values())
    summary = {"selected": selected, "passed": passed, "checks": checks, "ranking": ranking,
               "control": {t: {"allframe_voc": c["allframe_voc"], "annotated_voc": c["annotated_voc"]}
                           for t, c in control.items()},
               "variants": {rid: {t: {"allframe_voc": r[t][0]["allframe_voc"], "annotated_voc": r[t][0]["annotated_voc"],
                                      "video_snap": [r[t][1]["ap20"], r[t][1]["ap50"], r[t][1]["strict"]]}
                                  for t in ("tune", "confirm")} for rid, r in results.items()}}
    json.dump(summary, open(selection_file, "w"), indent=2, sort_keys=True)
    for rid, r in results.items():
        line = ("tune every-frame %.2f (control %.2f), annotated %.2f (%.2f); confirm every-frame %.2f (%.2f), "
                "annotated %.2f (%.2f); confirm video snap %.2f/%.2f/%.2f" % (
                    r["tune"][0]["allframe_voc"], control["tune"]["allframe_voc"], r["tune"][0]["annotated_voc"],
                    control["tune"]["annotated_voc"], r["confirm"][0]["allframe_voc"], control["confirm"]["allframe_voc"],
                    r["confirm"][0]["annotated_voc"], control["confirm"]["annotated_voc"],
                    r["confirm"][1]["ap20"], r["confirm"][1]["ap50"], r["confirm"][1]["strict"]))
        if rid == selected:
            verdict = "promoted" if passed else "rejected"
            text = "Selected on tune; pre-registered gate %s: %s. Checks: %s." % (
                "PASSED" if passed else "FAILED", line,
                ", ".join("%s %s" % (k, "ok" if v else "FAIL") for k, v in checks.items()))
        elif selected is None:
            verdict = "rejected"
            text = "No variant kept tune annotated-frame VOC within 0.5 of the control: %s." % line
        else:
            verdict = "rejected"
            text = "Not selected (selection on tune picked %s): %s." % (selected, line)
        ledger("verdict", "--id", rid, "--verdict", verdict, "--rationale", text)
    log("stage 2: selected %s, gate %s" % (selected, "PASSED" if passed else "FAILED"))

summary = json.load(open(selection_file))
if not summary["passed"]:
    marker = OUT + "/.not_tested"
    if not os.path.isfile(marker):
        ledger("note", "--id", "V2-WS2.0BN16w4", "--text",
               "Not evaluated on test: the selected boundary-negative variant (%s) failed its gate, so no "
               "BN model is evaluated on test (registered test-use clause)." % summary["selected"])
        open(marker, "w").write("gate failed\n")
    log("stage 3 skipped: gate failed")
    raise SystemExit(0)

# Stage 3: the selected variant's full-data model, tested once.
full_id, full_exp = FULL_OF[summary["selected"]]
full_cfg = "configs/yolost_%s.yaml" % full_exp
full_ckpt = "experiments/%s/ema_final.pt" % full_exp
if not os.path.isfile(full_cfg):
    exp, window, weight = VARIANTS[summary["selected"]]
    text = open("configs/yolost_v2_ws2_0_tubequeries_30ep.yaml", encoding="utf-8").read()
    for old, new in (("  filter_empty_clips: true\n", "  filter_empty_clips: true\n  boundary_negative_window: %d\n"
                      "  boundary_negative_weight: %.1f\n" % (window, weight)),
                     ("exp_dir: ./experiments/v2_ws2_0_tubequeries_30ep", "exp_dir: ./experiments/" + full_exp)):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    open(full_cfg, "w", encoding="utf-8").write(
        "# %s: V2-WS2.0 with patch 0031 at window %d, weight %g (selected on held-out). One variable.\n" % (
            full_id, window, weight) + text)
    freeze = dict(id=full_id, parent="V2-WS2.0", config=full_cfg,
                  config_sha256=hashlib.sha256(open(full_cfg, "rb").read()).hexdigest(),
                  one_variable="data.boundary_negative_window 0 -> %d, weight 1 -> %g" % (window, weight),
                  purpose="Full-data model of %s, which was selected and passed its gate." % summary["selected"],
                  gate="Evaluated on split-1 test once; replaces V2-WS2.0 whatever its test result.")
    path = "research_new/experiments/%s_FREEZE.json" % full_id
    json.dump(freeze, open(path, "w"), indent=2, sort_keys=True)
    ledger("register", "--freeze", path, "--control", "V2-WS2.0", "--workstream", "WS2", "--component", "loss", "--running")
    gpu = free_gpu()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
    subprocess.Popen([PY, "train.py", "--config", full_cfg], stdout=open("logs/yolost_%s.log" % full_exp, "w"),
                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
    log("stage 3: launched %s on GPU %d" % (full_id, gpu))
    time.sleep(600)

marker = OUT + "/.recorded_test"
if os.path.isfile(marker):
    raise SystemExit(0)
log("stage 3: waiting for " + full_ckpt)
while not done_training(full_ckpt, full_cfg):
    time.sleep(300)
time.sleep(60)
conv, snap, ctrl = evaluate(full_cfg, full_ckpt, "research_new/experiments/%s_RESULT" % full_id, "test")
record(full_id, "ucf-test", "ema_final-test-c8", conv, snap, ctrl)
ledger("note", "--id", full_id, "--text",
       "Single test evaluation after %s was selected and passed its gate: frame %.2f annotated, %.2f every frame "
       "(VOC); video with the frozen snap %.2f/%.2f/%.2f. Replaces V2-WS2.0 as the reported UCF101-24 model "
       "per the registered test-use clause." % (summary["selected"], conv["annotated_voc"], conv["allframe_voc"],
                                                snap["ap20"], snap["ap50"], snap["strict"]))
open(marker, "w").write("recorded\n")
log("stage 3 done: test every-frame %.2f annotated %.2f" % (conv["allframe_voc"], conv["annotated_voc"]))

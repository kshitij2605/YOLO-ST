"""Seven-candidate held-out selection, then (only on a pass) full-data training and one test.

Candidates (registered in research_new/experiments/TR_SELECTION_v2.json before any of their
every-frame numbers existed): KF64, MR100, TK1, NOACTX (held-out runs training now) and the
already-trained module ablations M2, M3, M4.

Stage 1  evaluate each candidate on tune and confirm as soon as it is trained (frozen protocol:
         clip weighting hann_peak_norm, candidate filter 8, frozen linker and snap); score every
         frame-mAP convention; record in the ledger. Evaluation may share a GPU with training.
Stage 2  among candidates whose tune annotated VOC >= control - 0.5, the highest tune every-frame
         VOC is gated on confirm (same gate as E3/BN). Verdicts (notes for M2-M4) to the ledger.
Stage 3  only if it passes: derive its full-data config from V2-WS2.0 with the same one-variable
         change, train it on a free GPU, evaluate split-1 test once, record.
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
OUT = "research_new/experiments/TR_RESULT_v3"
CONTROL_DIR = "research_new/experiments/V2HO-E3-s17_RESULT/control_ho0s17"
CONTROL_VIDEO_AP50_CONFIRM = 78.53
LINKER = ["--moc_link_iou", "0.45", "--moc_tubelet_nms", "0.6", "--moc_top_k", "10",
          "--moc_split_gap", "2", "--moc_tube_nms", "0.3"]
# id: (held-out exp dir, one-variable change for the full-data config, full-data id, full-data exp dir, new?)
CANDIDATES = {
    "V2HO-KF64-s17": ("v2ho_kf64_s17_30ep", ("  keyframe_frames: 16\n", "  keyframe_frames: 64\n"),
                      "V2-WS2.0KF64", "v2_ws2_0kf64_30ep", True),
    "V2HO-MR100-s17": ("v2ho_mr100_s17_30ep", ("  native_motion_channel_ratio: 0.25\n",
                                              "  native_motion_channel_ratio: 1.0\n"),
                       "V2-WS2.0MR100", "v2_ws2_0mr100_30ep", True),
    "V2HO-TK1-s17": ("v2ho_tk1_s17_30ep", ("  temporal_adapter_kernel: 5\n", "  temporal_adapter_kernel: 1\n"),
                     "V2-WS2.0TK1", "v2_ws2_0tk1_30ep", True),
    "V2HO-NOACTX-s17": ("v2ho_noactx_s17_30ep", ("  apt_actor_context: true\n", "  apt_actor_context: false\n"),
                        "V2-WS2.0NOACTX", "v2_ws2_0noactx_30ep", True),
    "V2HO-M2-s17": ("v2ho_m2_notaskadapt_s17_30ep", ("  apt_tube_query_task_adapters: true\n",
                                                     "  apt_tube_query_task_adapters: false\n"),
                    "V2-WS2.0M2", "v2_ws2_0m2_30ep", False),
    "V2HO-M3-s17": ("v2ho_m3_nomemory_s17_30ep", ("  apt_cross_clip_memory: true\n", "  apt_cross_clip_memory: false\n"),
                    "V2-WS2.0M3", "v2_ws2_0m3_30ep", False),
    "V2HO-M4-s17": ("v2ho_m4_nopyramidadapt_s17_30ep", ("  apt_pyramid_adapter: true\n", "  apt_pyramid_adapter: false\n"),
                    "V2-WS2.0M4", "v2_ws2_0m4_30ep", False),
}


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), message, flush=True)


def ledger(*args):
    out = subprocess.run([PY, "tools/experiment_ledger.py"] + list(args), capture_output=True, text=True)
    log("ledger: " + (out.stdout.strip() or out.stderr.strip())[:300])


def training_running(config):
    out = subprocess.run(["pgrep", "-f", "train.py --config " + config], capture_output=True, text=True).stdout.split()
    return [pid for pid in out if pid != str(os.getpid())]


def done_training(ckpt, config):
    running = bool(training_running(config))
    if os.path.isfile(ckpt) and not running:
        return True
    if not running and not os.path.isfile(ckpt):
        time.sleep(600)
        if not training_running(config) and not os.path.isfile(ckpt):
            raise SystemExit("ERROR: %s not running and %s missing" % (config, ckpt))
    return False


def gpu_with_free(need_mb):
    """Index of the GPU with the most free memory, once it has at least need_mb free."""
    while True:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
        free = {int(r.split(",")[0]): int(r.split(",")[2]) - int(r.split(",")[1]) for r in rows}
        best = max(free, key=free.get)
        if free[best] >= need_mb:
            return best
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
        gpu = gpu_with_free(20000)
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


def gate_checks(r):
    return {
        "confirm_allframe_+1.0": r["confirm"][0]["allframe_voc"] >= control["confirm"]["allframe_voc"] + 1.0,
        "tune_allframe_+0.5": r["tune"][0]["allframe_voc"] >= control["tune"]["allframe_voc"] + 0.5,
        "tune_annotated_-0.5": r["tune"][0]["annotated_voc"] >= control["tune"]["annotated_voc"] - 0.5,
        "confirm_annotated_-0.5": r["confirm"][0]["annotated_voc"] >= control["confirm"]["annotated_voc"] - 0.5,
        "confirm_video_ap50_-2.0": r["confirm"][1]["ap50"] >= CONTROL_VIDEO_AP50_CONFIRM - 2.0,
    }


# Stage 1
results, pending = {}, dict(CANDIDATES)
log("stage 1: %d candidates" % len(pending))
while pending:
    for run_id, (exp, change, full_id, full_exp, new) in list(pending.items()):
        config = "configs/yolost_%s.yaml" % exp
        ckpt = "experiments/%s/ema_final.pt" % exp
        try:
            if not done_training(ckpt, config):
                continue
        except SystemExit as error:
            log("%s dropped: %s" % (run_id, error))
            ledger("note", "--id", run_id, "--text", "Training ended without ema_final.pt; excluded from the selection.")
            del pending[run_id]
            continue
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

# Stage 2
selection_file = OUT + "/selection.json"
if not os.path.isfile(selection_file):
    eligible = {rid: r for rid, r in results.items()
                if r["tune"][0]["annotated_voc"] >= control["tune"]["annotated_voc"] - 0.5}
    ranking = sorted(eligible, key=lambda rid: -eligible[rid]["tune"][0]["allframe_voc"])
    selected = ranking[0] if ranking else None
    checks = gate_checks(results[selected]) if selected else {}
    passed = bool(selected) and all(checks.values())
    summary = {"selected": selected, "passed": passed, "checks": checks, "ranking": ranking,
               "control": {t: {"allframe_voc": c["allframe_voc"], "annotated_voc": c["annotated_voc"]}
                           for t, c in control.items()},
               "candidates": {rid: {t: {"allframe_voc": r[t][0]["allframe_voc"], "annotated_voc": r[t][0]["annotated_voc"],
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
            text = "2026-09-25 selection: selected on tune; pre-registered gate %s: %s. Checks: %s." % (
                "PASSED" if passed else "FAILED", line,
                ", ".join("%s %s" % (k, "ok" if v else "FAIL") for k, v in checks.items()))
            verdict = "promoted" if passed else "rejected"
        else:
            text = "2026-09-25 selection: not selected (top on tune: %s): %s." % (selected, line)
            verdict = "rejected"
        if CANDIDATES[rid][4]:
            ledger("verdict", "--id", rid, "--verdict", verdict, "--rationale", text)
        else:
            ledger("note", "--id", rid, "--text", text)
    log("stage 2: selected %s, gate %s" % (selected, "PASSED" if passed else "FAILED"))

summary = json.load(open(selection_file))
if not summary["passed"]:
    log("stage 3 skipped: gate failed; nothing is evaluated on test")
    raise SystemExit(0)

# Stage 3: full-data model of the selected candidate, trained now, tested once.
exp, change, full_id, full_exp, new = CANDIDATES[summary["selected"]]
full_cfg, full_ckpt = "configs/yolost_%s.yaml" % full_exp, "experiments/%s/ema_final.pt" % full_exp
launched_marker = OUT + "/.full_launched"
if not os.path.isfile(launched_marker):
    if not os.path.isfile(full_cfg):
        text = open("configs/yolost_v2_ws2_0_tubequeries_30ep.yaml", encoding="utf-8").read()
        for old, repl in (change, ("exp_dir: ./experiments/v2_ws2_0_tubequeries_30ep", "exp_dir: ./experiments/" + full_exp)):
            assert text.count(old) == 1, old
            text = text.replace(old, repl)
        open(full_cfg, "w", encoding="utf-8").write(
            "# %s: V2-WS2.0 with %s -> %s (selected on held-out). One variable.\n" % (
                full_id, change[0].strip(), change[1].strip()) + text)
        freeze = dict(id=full_id, parent="V2-WS2.0", config=full_cfg,
                      config_sha256=hashlib.sha256(open(full_cfg, "rb").read()).hexdigest(),
                      one_variable="%s -> %s" % (change[0].strip(), change[1].strip()),
                      purpose="Full-data model of %s, which was selected and passed its gate." % summary["selected"],
                      gate="Evaluated on split-1 test once; replaces V2-WS2.0 whatever its test result.")
        path = "research_new/experiments/%s_FREEZE.json" % full_id
        json.dump(freeze, open(path, "w"), indent=2, sort_keys=True)
        ledger("register", "--freeze", path, "--control", "V2-WS2.0", "--workstream", "WS2", "--component", "model", "--running")
    else:
        ledger("note", "--id", full_id, "--text", "Relaunched from scratch after %s passed its gate." % summary["selected"])
    gpu = gpu_with_free(80000)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
    subprocess.Popen([PY, "train.py", "--config", full_cfg], stdout=open("logs/yolost_%s.log" % full_exp, "w"),
                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
    open(launched_marker, "w").write("GPU %d\n" % gpu)
    log("stage 3: launched %s on GPU %d" % (full_id, gpu))
    time.sleep(900)

marker = OUT + "/.recorded_test"
if os.path.isfile(marker):
    raise SystemExit(0)
while not done_training(full_ckpt, full_cfg):
    time.sleep(300)
time.sleep(60)
conv, snap, ctrl = evaluate(full_cfg, full_ckpt, "research_new/experiments/%s_RESULT" % full_id, "test")
record(full_id, "ucf-test", "ema_final-test-c8", conv, snap, ctrl)
ledger("note", "--id", full_id, "--text",
       "Single test evaluation after %s was selected and passed its gate: frame %.2f annotated, %.2f every frame "
       "(VOC, inclusive extents); video with the frozen snap %.2f/%.2f/%.2f. Replaces V2-WS2.0 as the reported "
       "UCF101-24 model per the registered test-use clause." % (
           summary["selected"], conv["annotated_voc"], conv["allframe_voc"], snap["ap20"], snap["ap50"], snap["strict"]))
open(marker, "w").write("recorded\n")
log("stage 3 done: test every-frame %.2f annotated %.2f" % (conv["allframe_voc"], conv["annotated_voc"]))

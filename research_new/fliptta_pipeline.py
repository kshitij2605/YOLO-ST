"""Flip test-time augmentation: held-out gate on NOACTX and TK1, then (only on a pass) one test of V2-WS2.0.

Pre-registered in research_new/experiments/V2HO-FLIPTTA_FREEZE.json before any held-out TTA number existed.
For each model and partition the plain view is the frozen-protocol evaluation (reused from tr_pipeline_v4.py
for its candidates once their .recorded marker exists, else run here with identical arguments) and the second
view is the same evaluation with --hflip; tools/fuse_flip_views.py fuses them with fixed settings. Plain and
fused are scored identically (frozen snap, no-snap control, every frame-mAP convention).

Order: the decision models (NOACTX, TK1) as they finish -> gate -> single-pass V2-WS2.0 test reproduction
(always; the reported numbers should come back) -> fused test only on a pass. KF64 and HO0GC are reported when
they finish; they do not decide. One evaluation at a time from this script, at most two for the test pair.
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable
OUT = "research_new/experiments/FLIPTTA_RESULT"
V4 = "research_new/experiments/TR_RESULT_v4"
RUN_ID = "V2HO-FLIPTTA"
LINKER = ["--moc_link_iou", "0.45", "--moc_tubelet_nms", "0.6", "--moc_top_k", "10",
          "--moc_split_gap", "2", "--moc_tube_nms", "0.3"]
EVAL_FREE_MB = 25000
DECIDE = [("V2HO-NOACTX-s17", "v2ho_noactx_s17_30ep"), ("V2HO-TK1-s17", "v2ho_tk1_s17_30ep")]
REPORT = [("V2HO-KF64-s17", "v2ho_kf64_s17_30ep"), ("V2HO-HO0GC-s17", "v2ho_ho0gc_s17_30ep")]
V4_IDS = {"V2HO-NOACTX-s17", "V2HO-TK1-s17", "V2HO-KF64-s17"}
PARTITIONS = {"tune": "ucf-v2ho-g24g25", "confirm": "ucf-v2ho-g22g23"}


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), message, flush=True)


def ledger(*args):
    out = subprocess.run([PY, "tools/experiment_ledger.py"] + list(args), capture_output=True, text=True)
    log("ledger: " + (out.stdout.strip() or out.stderr.strip())[:300])


def gpu_with_free(need_mb):
    while True:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
        free = {int(r.split(",")[0]): int(r.split(",")[2]) - int(r.split(",")[1]) for r in rows}
        best = max(free, key=free.get)
        if free[best] >= need_mb:
            return best
        time.sleep(120)


def training_done(exp):
    config = "configs/yolost_%s%s.yaml" % (exp, "" if "ho0gc" in exp else "_gc")
    pattern = r"^([^ ]*/)?python3? train\.py --config %s( |$)" % config.replace(".", r"\.")
    running = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return os.path.isfile("experiments/%s/ema_final.pt" % exp) and not running


def eval_cmd(config, ckpt, frames, cache, hflip):
    return ([PY, "eval_tube_queries.py", "--config", config, "--checkpoint", ckpt, "--clip_weighting",
             "hann_peak_norm", "--frame_map", "--frame_dump", frames, "--candidate_cache", cache,
             "--min_length", "8"] + LINKER + (["--hflip"] if hflip else []))


def launch(cmd, logfile):
    gpu = gpu_with_free(EVAL_FREE_MB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log("run on GPU %d: %s" % (gpu, " ".join(cmd[1:4] + cmd[-3:])))
    return subprocess.Popen(cmd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT, env=env)


def wait(proc, logfile):
    if proc.wait():
        raise SystemExit("FAILED: see " + logfile)


def run_eval(config, ckpt, frames, cache, hflip, logfile):
    if os.path.isfile(frames) and os.path.isfile(cache):
        return
    wait(launch(eval_cmd(config, ckpt, frames, cache, hflip), logfile), logfile)


def score(frames, cache, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    result = {}
    for kind, blend, smooth in (("snap_frozen", "0.75", "2"), ("snap_control", "0", "0")):
        target = "%s/%s_%s.json" % (outdir, kind, name)
        if not os.path.isfile(target):
            cmd = ["nice", "-n", "10", PY, "tools/snap_tubes_to_dense_geometry.py", "--tube-cache", cache,
                   "--frame-dump", frames, "--match-iou", "0.3", "--box-blend", blend, "--score-blend", "0",
                   "--smooth-radius", smooth, "--out", target]
            if subprocess.call(cmd, stdout=open(target + ".log", "w"), stderr=subprocess.STDOUT):
                raise SystemExit("FAILED: " + target)
        result[kind] = json.load(open(target))["rows"][0]
    conv = "%s/frame_conventions_%s.json" % (outdir, name)
    if not os.path.isfile(conv):
        cmd = ["nice", "-n", "10", PY, "tools/score_frame_conventions.py", "--dump", frames, "--out", conv]
        if subprocess.call(cmd, stdout=open(conv + ".log", "w"), stderr=subprocess.STDOUT):
            raise SystemExit("FAILED: " + conv)
    result["conv"] = json.load(open(conv))
    return result


def summary(s):
    return {"annotated_voc": s["conv"]["annotated_voc"], "allframe_voc": s["conv"]["allframe_voc"],
            "yowo_list_voc": s["conv"]["yowo_list_voc"], "ap20": s["snap_frozen"]["ap20"],
            "ap50": s["snap_frozen"]["ap50"], "strict": s["snap_frozen"]["strict"]}


def fuse(plain, flipped, outdir, tag):
    frames, cache = "%s/frames_%s_fused.pkl" % (outdir, tag), "%s/tube_cache_%s_c8_fused.pkl" % (outdir, tag)
    if not (os.path.isfile(frames) and os.path.isfile(cache)):
        cmd = [PY, "tools/fuse_flip_views.py", "--dumps", plain[0], flipped[0], "--caches", plain[1], flipped[1],
               "--out-dump", frames, "--out-cache", cache]
        if subprocess.call(cmd, stdout=open("%s/fuse_%s.log" % (outdir, tag), "w"), stderr=subprocess.STDOUT):
            raise SystemExit("FAILED: fuse " + tag)
    return frames, cache


def record(run_id, tag, partition, s, label):
    ledger("result", "--id", run_id, "--tag", "%s-video-snap" % tag, "--partition", partition,
           "--evaluator", "moc-act-geometry-snap-c8" + label, "--metric", "ap20=%.2f" % s["snap_frozen"]["ap20"],
           "--metric", "ap50=%.2f" % s["snap_frozen"]["ap50"], "--metric", "strict=%.2f" % s["snap_frozen"]["strict"])
    for key, evaluator in (("annotated_voc", "moc-act-corrected-frame-hann_peak_norm"),
                           ("yowo_list_voc", "yowo-frame-hann_peak_norm"),
                           ("allframe_voc", "road-allframe-voc-hann_peak_norm"),
                           ("yowoformer_exact", "yowoformer-released-exact-hann_peak_norm")):
        ledger("result", "--id", run_id, "--tag", "%s-frame" % tag, "--partition", partition,
               "--evaluator", evaluator + label, "--metric", "frame=%.2f" % s["conv"][key])


def heldout(run_id, exp):
    """Plain and fused summaries on tune and confirm for one held-out model."""
    outdir = "%s/%s" % (OUT, run_id)
    os.makedirs(outdir, exist_ok=True)
    marker = outdir + "/summary.json"
    if os.path.isfile(marker):
        return json.load(open(marker))
    ckpt = "experiments/%s/ema_final.pt" % exp
    configs = {tag: "configs/yolost_%s%s.yaml" % (exp, "" if tag == "tune" else "_evalconfirm") for tag in PARTITIONS}
    views = {}
    for tag in PARTITIONS:  # the flipped view first: it does not wait for tr_pipeline_v4
        flipped = ("%s/frames_%s_ema_hflip.pkl" % (outdir, tag), "%s/tube_cache_%s_ema_c8_hflip.pkl" % (outdir, tag))
        run_eval(configs[tag], ckpt, flipped[0], flipped[1], True, "%s/eval_%s_hflip.log" % (outdir, tag))
        views[tag] = flipped
    res = {}
    for tag, partition in PARTITIONS.items():
        if run_id in V4_IDS:
            while not os.path.isfile("%s/%s/.recorded" % (V4, run_id)):
                time.sleep(120)
            plain = ("%s/%s/frames_%s_ema.pkl" % (V4, run_id, tag), "%s/%s/tube_cache_%s_ema_c8.pkl" % (V4, run_id, tag))
        else:
            plain = ("%s/frames_%s_ema.pkl" % (outdir, tag), "%s/tube_cache_%s_ema_c8.pkl" % (outdir, tag))
            run_eval(configs[tag], ckpt, plain[0], plain[1], False, "%s/eval_%s.log" % (outdir, tag))
        flipped = views[tag]
        s_plain = score(plain[0], plain[1], outdir, tag + "_plain")
        s_fused = score(*fuse(plain, flipped, outdir, tag), outdir=outdir, name=tag + "_fused")
        res[tag] = {"plain": summary(s_plain), "fused": summary(s_fused)}
        if run_id not in V4_IDS:
            record(run_id, "ema_final-%s-c8" % tag, partition, s_plain, "")
        record(RUN_ID, "%s-ema_final-%s-c8-fliptta" % (run_id, tag), partition, s_fused, "-fliptta")
    json.dump(res, open(marker, "w"), indent=2, sort_keys=True)
    line = "; ".join("%s annotated %.2f->%.2f, every-frame %.2f->%.2f, video %.2f/%.2f/%.2f->%.2f/%.2f/%.2f" % (
        tag, r["plain"]["annotated_voc"], r["fused"]["annotated_voc"], r["plain"]["allframe_voc"],
        r["fused"]["allframe_voc"], r["plain"]["ap20"], r["plain"]["ap50"], r["plain"]["strict"],
        r["fused"]["ap20"], r["fused"]["ap50"], r["fused"]["strict"]) for tag, r in res.items())
    log("%s: %s" % (run_id, line))
    ledger("note", "--id", RUN_ID, "--text", "%s, plain -> flip TTA: %s." % (run_id, line))
    return res


def delta(r, tag, key):
    return r[tag]["fused"][key] - r[tag]["plain"][key]


os.makedirs(OUT, exist_ok=True)
log("waiting for %s" % ", ".join(rid for rid, _ in DECIDE))
results = {}
pending = list(DECIDE)
while pending:
    for run_id, exp in list(pending):
        if training_done(exp):
            if not os.path.isfile("%s/%s/summary.json" % (OUT, run_id)):
                time.sleep(300)  # let tr_pipeline_v4 place its own evaluation first
            results[run_id] = heldout(run_id, exp)
            pending.remove((run_id, exp))
    if pending:
        time.sleep(120)

verdict_file = OUT + "/gate.json"
if not os.path.isfile(verdict_file):
    checks = {}
    for crit, key, threshold in (("a_annotated", "annotated_voc", 0.10), ("b_ap5095", "strict", 0.30)):
        checks[crit] = all(delta(results[rid], "confirm", key) >= threshold for rid, _ in DECIDE) and \
            all(delta(results[rid], "tune", key) >= 0 for rid, _ in DECIDE)
    guard = all(delta(results[rid], "confirm", key) >= -0.20 for rid, _ in DECIDE
                for key in ("allframe_voc", "ap20", "ap50"))
    passed = (checks["a_annotated"] or checks["b_ap5095"]) and guard
    gate = {"passed": passed, "criteria": checks, "confirm_guard": guard,
            "deltas": {rid: {tag: {k: round(delta(results[rid], tag, k), 4) for k in results[rid][tag]["plain"]}
                             for tag in PARTITIONS} for rid, _ in DECIDE}}
    json.dump(gate, open(verdict_file, "w"), indent=2, sort_keys=True)
    text = "Pre-registered gate %s: criterion (a) annotated +0.10 %s, (b) video AP50:95 +0.30 %s, confirm guard %s. " \
           "Deltas (fused - plain): %s" % ("PASSED" if passed else "FAILED",
                                            "met" if checks["a_annotated"] else "not met",
                                            "met" if checks["b_ap5095"] else "not met",
                                            "ok" if guard else "FAILED", json.dumps(gate["deltas"], sort_keys=True))
    ledger("verdict", "--id", RUN_ID, "--verdict", "promoted" if passed else "rejected", "--rationale", text[:1500])
    log("gate: %s" % ("PASSED" if passed else "FAILED"))
gate = json.load(open(verdict_file))

# Single-pass V2-WS2.0 test reproduction (always) and, only on a pass, the fused test.
test_dir = OUT + "/V2-WS2.0_test"
os.makedirs(test_dir, exist_ok=True)
config, ckpt = "configs/yolost_v2_ws2_0_tubequeries_30ep.yaml", "experiments/v2_ws2_0_tubequeries_30ep/ema_final.pt"
plain = ("%s/frames_test_ema.pkl" % test_dir, "%s/tube_cache_test_ema_c8.pkl" % test_dir)
flipped = ("%s/frames_test_ema_hflip.pkl" % test_dir, "%s/tube_cache_test_ema_c8_hflip.pkl" % test_dir)
procs = []
if not (os.path.isfile(plain[0]) and os.path.isfile(plain[1])):
    procs.append((launch(eval_cmd(config, ckpt, plain[0], plain[1], False), test_dir + "/eval_test.log"),
                  test_dir + "/eval_test.log"))
    time.sleep(180)
if gate["passed"] and not (os.path.isfile(flipped[0]) and os.path.isfile(flipped[1])):
    procs.append((launch(eval_cmd(config, ckpt, flipped[0], flipped[1], True), test_dir + "/eval_test_hflip.log"),
                  test_dir + "/eval_test_hflip.log"))
for proc, logfile in procs:
    wait(proc, logfile)
s_plain = score(plain[0], plain[1], test_dir, "test_plain")
if not os.path.isfile(test_dir + "/.recorded_plain"):
    record("V2-WS2.0", "ema_final-test-c8-repro0926", "ucf-test", s_plain, "")
    p = summary(s_plain)
    ledger("note", "--id", "V2-WS2.0", "--text", "Reproduction of the reported numbers "
           "(checkpoint md5 296f1473...): single-pass frozen protocol on split-1 test gives "
           "annotated %.2f, YOWO list %.2f, every-frame %.2f, video %.2f/%.2f/%.2f; reported 93.21, 93.32, 89.54, "
           "89.82/73.39/34.97." % (p["annotated_voc"], p["yowo_list_voc"], p["allframe_voc"], p["ap20"], p["ap50"],
                                    p["strict"]))
    open(test_dir + "/.recorded_plain", "w").write("recorded\n")
log("V2-WS2.0 test reproduction: %s" % json.dumps(summary(s_plain), sort_keys=True))
if gate["passed"] and not os.path.isfile(test_dir + "/.recorded_fused"):
    s_fused = score(*fuse(plain, flipped, test_dir, "test"), outdir=test_dir, name="test_fused")
    record(RUN_ID, "V2-WS2.0-ema_final-test-c8-fliptta", "ucf-test", s_fused, "-fliptta")
    f = summary(s_fused)
    ledger("note", "--id", RUN_ID, "--text", "Single test evaluation after the held-out gate passed: V2-WS2.0 with flip "
           "TTA gives annotated %.2f, YOWO list %.2f, every-frame %.2f, video %.2f/%.2f/%.2f (single pass in the same "
           "run: %s). Reported as a separate '+ flip TTA' row." % (
               f["annotated_voc"], f["yowo_list_voc"], f["allframe_voc"], f["ap20"], f["ap50"], f["strict"],
               json.dumps(summary(s_plain), sort_keys=True)))
    open(test_dir + "/.recorded_fused", "w").write("recorded\n")
    log("V2-WS2.0 test with flip TTA: %s" % json.dumps(f, sort_keys=True))

# Report-only models, as they finish.
for run_id, exp in REPORT:
    while not training_done(exp):
        time.sleep(300)
    heldout(run_id, exp)
log("done")

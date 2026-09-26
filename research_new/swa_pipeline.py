"""V2HO-SWA: averaged EMA snapshots (epochs 24 and 29) on held-out, then (only on a pass) V2-WS2.0 on test once.

Pre-registered in research_new/experiments/V2HO-SWA_FREEZE.json. The six held-out evaluations run in parallel;
the plain ema_final numbers of the same models come from tr_pipeline_v4.py (TR_RESULT_v4), scored identically.
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable
OUT = "research_new/experiments/SWA_RESULT"
V4 = "research_new/experiments/TR_RESULT_v4"
RUN_ID = "V2HO-SWA"
LINKER = ["--moc_link_iou", "0.45", "--moc_tubelet_nms", "0.6", "--moc_top_k", "10",
          "--moc_split_gap", "2", "--moc_tube_nms", "0.3"]
DECIDE = [("V2HO-KF64-s17", "v2ho_kf64_s17_30ep"), ("V2HO-TK1-s17", "v2ho_tk1_s17_30ep"),
          ("V2HO-NOACTX-s17", "v2ho_noactx_s17_30ep")]
PARTITIONS = {"tune": "ucf-v2ho-g24g25", "confirm": "ucf-v2ho-g22g23"}
GPUS = [0, 1, 2]


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), message, flush=True)


def ledger(*args):
    out = subprocess.run([PY, "tools/experiment_ledger.py"] + list(args), capture_output=True, text=True)
    log("ledger: " + (out.stdout.strip() or out.stderr.strip())[:300])


def average(exp_dir):
    out = exp_dir + "/ema_avg_24_final.pt"
    if not os.path.isfile(out):
        subprocess.check_call([PY, "tools/average_checkpoints.py", "--inputs", exp_dir + "/ema_epoch_24.pt",
                               exp_dir + "/ema_final.pt", "--out", out])
    return out


def eval_proc(config, ckpt, frames, cache, logfile, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [PY, "eval_tube_queries.py", "--config", config, "--checkpoint", ckpt, "--clip_weighting", "hann_peak_norm",
           "--frame_map", "--frame_dump", frames, "--candidate_cache", cache, "--min_length", "8"] + LINKER
    return subprocess.Popen(cmd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT, env=env)


def score(frames, cache, outdir, name):
    result = {}
    for kind, blend, smooth in (("snap_frozen", "0.75", "2"), ("snap_control", "0", "0")):
        target = "%s/%s_%s.json" % (outdir, kind, name)
        if not os.path.isfile(target):
            subprocess.check_call(["nice", "-n", "10", PY, "tools/snap_tubes_to_dense_geometry.py", "--tube-cache", cache,
                                   "--frame-dump", frames, "--match-iou", "0.3", "--box-blend", blend, "--score-blend",
                                   "0", "--smooth-radius", smooth, "--out", target],
                                  stdout=open(target + ".log", "w"), stderr=subprocess.STDOUT)
        result[kind] = json.load(open(target))["rows"][0]
    conv = "%s/frame_conventions_%s.json" % (outdir, name)
    if not os.path.isfile(conv):
        subprocess.check_call(["nice", "-n", "10", PY, "tools/score_frame_conventions.py", "--dump", frames, "--out", conv],
                              stdout=open(conv + ".log", "w"), stderr=subprocess.STDOUT)
    result["conv"] = json.load(open(conv))
    return result


def summary(s):
    return {"annotated_voc": s["conv"]["annotated_voc"], "allframe_voc": s["conv"]["allframe_voc"],
            "yowo_list_voc": s["conv"]["yowo_list_voc"], "yowoformer_exact": s["conv"]["yowoformer_exact"],
            "ap20": s["snap_frozen"]["ap20"], "ap50": s["snap_frozen"]["ap50"], "strict": s["snap_frozen"]["strict"]}


def record(run_id, tag, partition, s, label):
    ledger("result", "--id", run_id, "--tag", "%s-video-snap" % tag, "--partition", partition,
           "--evaluator", "moc-act-geometry-snap-c8" + label, "--metric", "ap20=%.2f" % s["snap_frozen"]["ap20"],
           "--metric", "ap50=%.2f" % s["snap_frozen"]["ap50"], "--metric", "strict=%.2f" % s["snap_frozen"]["strict"])
    for key, evaluator in (("annotated_voc", "moc-act-corrected-frame-hann_peak_norm"),
                           ("yowo_list_voc", "yowo-frame-hann_peak_norm"),
                           ("annotated_trapz", "moc-act-annotated-trapz-hann_peak_norm"),
                           ("allframe_voc", "road-allframe-voc-hann_peak_norm"),
                           ("allframe_trapz", "moc-act-allframe-trapz-hann_peak_norm"),
                           ("yowoformer_style", "yowoformer-released-reimpl-hann_peak_norm"),
                           ("yowoformer_exact", "yowoformer-released-exact-hann_peak_norm")):
        ledger("result", "--id", run_id, "--tag", "%s-frame" % tag, "--partition", partition,
               "--evaluator", evaluator + label, "--metric", "frame=%.2f" % s["conv"][key])


os.makedirs(OUT, exist_ok=True)
jobs, index = [], 0
for rid, exp in DECIDE:
    ckpt = average("experiments/" + exp)
    outdir = "%s/%s" % (OUT, rid)
    os.makedirs(outdir, exist_ok=True)
    for tag in PARTITIONS:
        config = "configs/yolost_%s%s.yaml" % (exp, "" if tag == "tune" else "_evalconfirm")
        frames, cache = "%s/frames_%s_avg.pkl" % (outdir, tag), "%s/tube_cache_%s_avg_c8.pkl" % (outdir, tag)
        if not (os.path.isfile(frames) and os.path.isfile(cache)):
            jobs.append((eval_proc(config, ckpt, frames, cache, "%s/eval_%s.log" % (outdir, tag), GPUS[index % 3]),
                         outdir, tag))
            index += 1
log("held-out: %d evaluations launched" % len(jobs))
for proc, outdir, tag in jobs:
    if proc.wait():
        raise SystemExit("FAILED: %s/eval_%s.log" % (outdir, tag))

results = {}
for rid, exp in DECIDE:
    outdir = "%s/%s" % (OUT, rid)
    res = {}
    for tag, partition in PARTITIONS.items():
        avg = score("%s/frames_%s_avg.pkl" % (outdir, tag), "%s/tube_cache_%s_avg_c8.pkl" % (outdir, tag), outdir,
                    tag + "_avg")
        plain = score("%s/%s/frames_%s_ema.pkl" % (V4, rid, tag), "%s/%s/tube_cache_%s_ema_c8.pkl" % (V4, rid, tag),
                      outdir, tag + "_plain")
        res[tag] = {"plain": summary(plain), "avg": summary(avg)}
        if not os.path.isfile("%s/.recorded_%s" % (outdir, tag)):
            record(RUN_ID, "%s-ema_avg_24_final-%s-c8" % (rid, tag), partition, avg, "-swa")
            open("%s/.recorded_%s" % (outdir, tag), "w").write("recorded\n")
    results[rid] = res
    log("%s plain -> averaged: %s" % (rid, "; ".join(
        "%s annotated %.2f->%.2f every-frame %.2f->%.2f video %.2f/%.2f/%.2f->%.2f/%.2f/%.2f" % (
            tag, r["plain"]["annotated_voc"], r["avg"]["annotated_voc"], r["plain"]["allframe_voc"],
            r["avg"]["allframe_voc"], r["plain"]["ap20"], r["plain"]["ap50"], r["plain"]["strict"],
            r["avg"]["ap20"], r["avg"]["ap50"], r["avg"]["strict"]) for tag, r in res.items())))


def d(rid, tag, key):
    return results[rid][tag]["avg"][key] - results[rid][tag]["plain"][key]


def mean(tag, key):
    return sum(d(rid, tag, key) for rid, _ in DECIDE) / len(DECIDE)


gate_file = OUT + "/gate.json"
if not os.path.isfile(gate_file):
    checks = {"confirm_annotated_mean_+0.10": mean("confirm", "annotated_voc") >= 0.10,
              "confirm_annotated_each_>=0": all(d(rid, "confirm", "annotated_voc") >= 0 for rid, _ in DECIDE),
              "tune_annotated_mean_>=0": mean("tune", "annotated_voc") >= 0,
              "confirm_allframe_mean_>=0": mean("confirm", "allframe_voc") >= 0,
              "confirm_ap20_mean_>=-0.5": mean("confirm", "ap20") >= -0.5,
              "confirm_ap50_mean_>=-0.5": mean("confirm", "ap50") >= -0.5}
    passed = all(checks.values())
    deltas = {rid: {tag: {k: round(d(rid, tag, k), 4) for k in results[rid][tag]["plain"]} for tag in PARTITIONS}
              for rid, _ in DECIDE}
    json.dump({"passed": passed, "checks": checks, "deltas": deltas}, open(gate_file, "w"), indent=2, sort_keys=True)
    ledger("verdict", "--id", RUN_ID, "--verdict", "promoted" if passed else "rejected", "--rationale",
           ("Pre-registered gate %s. Checks: %s. Deltas (averaged - ema_final): %s" % (
               "PASSED" if passed else "FAILED", json.dumps(checks, sort_keys=True), json.dumps(deltas, sort_keys=True)))[:1800])
    log("gate: %s %s" % ("PASSED" if passed else "FAILED", json.dumps(checks, sort_keys=True)))
gate = json.load(open(gate_file))
if not gate["passed"]:
    log("done: no test evaluation")
    raise SystemExit(0)

test_dir = OUT + "/V2-WS2.0_test"
os.makedirs(test_dir, exist_ok=True)
ckpt = average("experiments/v2_ws2_0_tubequeries_30ep")
frames, cache = test_dir + "/frames_test_avg.pkl", test_dir + "/tube_cache_test_avg_c8.pkl"
if not (os.path.isfile(frames) and os.path.isfile(cache)):
    proc = eval_proc("configs/yolost_v2_ws2_0_tubequeries_30ep.yaml", ckpt, frames, cache, test_dir + "/eval_test.log", 0)
    if proc.wait():
        raise SystemExit("FAILED: %s/eval_test.log" % test_dir)
s = score(frames, cache, test_dir, "test_avg")
if not os.path.isfile(test_dir + "/.recorded"):
    record(RUN_ID, "V2-WS2.0-ema_avg_24_final-test-c8", "ucf-test", s, "-swa")
    t = summary(s)
    ledger("note", "--id", RUN_ID, "--text", "Single test evaluation after the held-out gate passed: V2-WS2.0 with "
           "averaged EMA snapshots (epochs 24 and 29) gives annotated %.2f, YOWO list %.2f, every-frame %.2f, "
           "YOWOFormer-exact %.2f, video %.2f/%.2f/%.2f (single model, single pass). Replaces V2-WS2.0 as the "
           "single-model result per the registered test-use clause." % (
               t["annotated_voc"], t["yowo_list_voc"], t["allframe_voc"], t["yowoformer_exact"], t["ap20"], t["ap50"],
               t["strict"]))
    open(test_dir + "/.recorded", "w").write("recorded\n")
log("test (averaged V2-WS2.0): %s" % json.dumps(summary(s), sort_keys=True))
log("done")

"""V2HO-TUBEFUSE2 check on HO0GC's flip views, then the V2-ENS5 test ensemble (both pre-registered 2026-09-26).

1. As soon as HO0GC's plain and --hflip held-out evaluations exist, fuse them with --tube-frames top (and union,
   reported) and compare video AP with the plain evaluation under the frozen snap: PASS if on tune and confirm
   AP20, AP50 and AP50:95 each change by >= -0.20.
2. When all five single-pass test evaluations exist (V2-WS2.0 + four V2HO members), fuse them. Frame metrics are
   always reported; video metrics only if step 1 passed (top rule), else the video numbers stay V2-WS2.0's.
Also scores V2-WS2.0 single and the ensemble with ROAD's exact every-frame code for the paper's Table 1 row.
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable
OUT = "research_new/experiments/ENS5_RESULT"
HO = "research_new/experiments/FLIPTTA_RESULT/V2HO-HO0GC-s17"
WS = "research_new/experiments/FLIPTTA_RESULT/V2-WS2.0_test"
MEMBERS = ["V2HO-HO0GC-s17", "V2HO-KF64-s17", "V2HO-TK1-s17", "V2HO-NOACTX-s17"]


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), message, flush=True)


def ledger(*args):
    out = subprocess.run([PY, "tools/experiment_ledger.py"] + list(args), capture_output=True, text=True)
    log("ledger: " + (out.stdout.strip() or out.stderr.strip())[:300])


def run(cmd, logfile):
    if subprocess.call(cmd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT):
        raise SystemExit("FAILED: see " + logfile)


def snap(frames, cache, target):
    if not os.path.isfile(target):
        run(["nice", "-n", "10", PY, "tools/snap_tubes_to_dense_geometry.py", "--tube-cache", cache, "--frame-dump",
             frames, "--match-iou", "0.3", "--box-blend", "0.75", "--score-blend", "0", "--smooth-radius", "2",
             "--out", target], target + ".log")
    return json.load(open(target))["rows"][0]


def conventions(frames, target):
    if not os.path.isfile(target):
        run(["nice", "-n", "10", PY, "tools/score_frame_conventions.py", "--dump", frames, "--out", target],
            target + ".log")
    return json.load(open(target))


def road(frames, target):
    if not os.path.isfile(target):
        run(["nice", "-n", "10", PY, "tools/road_exact_frame_map.py", frames, target], target + ".log")
    return json.load(open(target))["ROAD exact (exclusive IoU, raw float boxes)"]


def fuse(dumps, caches, out_dump, out_cache, rule, logfile):
    if not (os.path.isfile(out_dump) and os.path.isfile(out_cache)):
        run([PY, "tools/fuse_model_views.py", "--dumps"] + dumps + ["--caches"] + caches +
            ["--out-dump", out_dump, "--out-cache", out_cache, "--tube-frames", rule], logfile)


def exists(*paths):
    return all(os.path.isfile(p) for p in paths)


os.makedirs(OUT + "/tubefuse2", exist_ok=True)
views = {tag: ("%s/frames_%s_ema.pkl" % (HO, tag), "%s/tube_cache_%s_ema_c8.pkl" % (HO, tag),
               "%s/frames_%s_ema_hflip.pkl" % (HO, tag), "%s/tube_cache_%s_ema_c8_hflip.pkl" % (HO, tag))
         for tag in ("tune", "confirm")}
log("waiting for HO0GC flip views")
while not all(exists(*v) for v in views.values()):
    time.sleep(120)
time.sleep(60)  # let the last pickle finish writing

gate_file = OUT + "/tubefuse2/gate.json"
if not os.path.isfile(gate_file):
    table, checks = {}, {}
    for tag, (fp, cp, fh, ch) in views.items():
        d = OUT + "/tubefuse2"
        plain = snap(fp, cp, "%s/snap_%s_plain.json" % (d, tag))
        row = {"plain": plain}
        for rule in ("top", "union"):
            fd, fc = "%s/frames_%s_%s.pkl" % (d, tag, rule), "%s/tube_cache_%s_%s.pkl" % (d, tag, rule)
            fuse([fp, fh], [cp, ch], fd, fc, rule, "%s/fuse_%s_%s.log" % (d, tag, rule))
            row[rule] = snap(fd, fc, "%s/snap_%s_%s.json" % (d, tag, rule))
        table[tag] = {k: {m: v[m] for m in ("ap20", "ap50", "strict")} for k, v in row.items()}
        for m in ("ap20", "ap50", "strict"):
            checks["%s_%s" % (tag, m)] = row["top"][m] - plain[m] >= -0.20
    passed = all(checks.values())
    json.dump({"passed": passed, "checks": checks, "video": table}, open(gate_file, "w"), indent=2, sort_keys=True)
    ledger("verdict", "--id", "V2HO-TUBEFUSE2", "--verdict", "promoted" if passed else "rejected", "--rationale",
           ("Pre-registered check %s on V2HO-HO0GC-s17 flip views (video AP20/AP50/AP50:95, frozen snap): %s" % (
               "PASSED" if passed else "FAILED", json.dumps(table, sort_keys=True)))[:1800])
    log("TUBEFUSE2 %s: %s" % ("PASSED" if passed else "FAILED", json.dumps(table, sort_keys=True)))
tubefuse_passed = json.load(open(gate_file))["passed"]

members = [(WS + "/frames_test_ema.pkl", WS + "/tube_cache_test_ema_c8.pkl")] + [
    ("%s/%s/frames_test_ema.pkl" % (OUT, m), "%s/%s/tube_cache_test_ema_c8.pkl" % (OUT, m)) for m in MEMBERS]
log("waiting for the five test evaluations")
while not all(exists(f, c) for f, c in members):
    time.sleep(120)
time.sleep(60)
rule = "top" if tubefuse_passed else "union"
fd, fc = OUT + "/frames_test_ens5.pkl", OUT + "/tube_cache_test_ens5_%s.pkl" % rule
fuse([f for f, _ in members], [c for _, c in members], fd, fc, rule, OUT + "/fuse_test.log")
conv = conventions(fd, OUT + "/frame_conventions_test_ens5.json")
road_ens = road(fd, OUT + "/road_exact_test_ens5.json")
road_single = road(WS + "/frames_test_ema.pkl", WS + "/road_exact_test_plain.json")
single = json.load(open(WS + "/frame_conventions_test_plain.json"))
video = snap(fd, fc, OUT + "/snap_frozen_test_ens5_%s.json" % rule) if tubefuse_passed else None
if not os.path.isfile(OUT + "/.recorded"):
    for key, evaluator in (("annotated_voc", "moc-act-corrected-frame-hann_peak_norm"),
                           ("yowo_list_voc", "yowo-frame-hann_peak_norm"),
                           ("annotated_trapz", "moc-act-annotated-trapz-hann_peak_norm"),
                           ("allframe_voc", "road-allframe-voc-hann_peak_norm"),
                           ("allframe_trapz", "moc-act-allframe-trapz-hann_peak_norm"),
                           ("yowoformer_style", "yowoformer-released-reimpl-hann_peak_norm"),
                           ("yowoformer_exact", "yowoformer-released-exact-hann_peak_norm")):
        ledger("result", "--id", "V2-ENS5", "--tag", "ens5-test-c8-frame", "--partition", "ucf-test",
               "--evaluator", evaluator + "-ens5", "--metric", "frame=%.2f" % conv[key])
    ledger("result", "--id", "V2-ENS5", "--tag", "ens5-test-c8-frame", "--partition", "ucf-test",
           "--evaluator", "road-exact-allframe-voc-exclusive-iou-hann_peak_norm-ens5", "--metric", "frame=%.2f" % road_ens)
    if video:
        ledger("result", "--id", "V2-ENS5", "--tag", "ens5-test-c8-video-snap", "--partition", "ucf-test",
               "--evaluator", "moc-act-geometry-snap-c8-ens5-tubetop", "--metric", "ap20=%.2f" % video["ap20"],
               "--metric", "ap50=%.2f" % video["ap50"], "--metric", "strict=%.2f" % video["strict"])
    ledger("note", "--id", "V2-ENS5", "--text",
           "Single test evaluation of the fixed 5-model ensemble: annotated %.2f (V2-WS2.0 alone %.2f), YOWO list %.2f "
           "(%.2f), every-frame inclusive %.2f (%.2f), ROAD-exact every-frame %.2f (%.2f), YOWOFormer-exact %.2f (%.2f); "
           "video %s. Reported as a separate 'ensemble of 5' row." % (
               conv["annotated_voc"], single["annotated_voc"], conv["yowo_list_voc"], single["yowo_list_voc"],
               conv["allframe_voc"], single["allframe_voc"], road_ens, road_single, conv["yowoformer_exact"],
               single["yowoformer_exact"],
               ("%.2f/%.2f/%.2f with the top tube rule (V2-WS2.0 alone 89.82/73.39/34.97)" % (
                   video["ap20"], video["ap50"], video["strict"])) if video else
               "not reported (V2HO-TUBEFUSE2 failed; single-model video stays V2-WS2.0's)"))
    open(OUT + "/.recorded", "w").write("recorded\n")
log("ENS5 test: frame %s | ROAD-exact %.2f (single %.2f) | video %s" % (
    json.dumps({k: round(conv[k], 2) for k in ("annotated_voc", "yowo_list_voc", "allframe_voc", "yowoformer_exact")}),
    road_ens, road_single, json.dumps(video) if video else "n/a"))
log("done")

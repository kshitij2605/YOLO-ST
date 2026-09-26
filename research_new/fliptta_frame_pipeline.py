"""V2HO-FLIPTTA-FRAME: gate on KF64 and HO0GC, then (only on a pass) V2-WS2.0 test with frame-level flip TTA.

Pre-registered in research_new/experiments/V2HO-FLIPTTA-FRAME_FREEZE.json. The per-model fused frame numbers come
from research_new/fliptta_pipeline.py (FLIPTTA_RESULT/<id>/summary.json); frame-level fusion there is independent
of the tube fusion, so its frame numbers are exactly this variant's.
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable
SRC = "research_new/experiments/FLIPTTA_RESULT"
OUT = "research_new/experiments/FLIPTTA_FRAME_RESULT"
RUN_ID = "V2HO-FLIPTTA-FRAME"
DECIDE = ["V2HO-KF64-s17", "V2HO-HO0GC-s17"]
LINKER = ["--moc_link_iou", "0.45", "--moc_tubelet_nms", "0.6", "--moc_top_k", "10",
          "--moc_split_gap", "2", "--moc_tube_nms", "0.3"]


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


os.makedirs(OUT, exist_ok=True)
log("waiting for %s" % ", ".join(DECIDE))
while not all(os.path.isfile("%s/%s/summary.json" % (SRC, rid)) for rid in DECIDE):
    time.sleep(300)
summaries = {rid: json.load(open("%s/%s/summary.json" % (SRC, rid))) for rid in DECIDE}


def d(rid, tag, key):
    return summaries[rid][tag]["fused"][key] - summaries[rid][tag]["plain"][key]


gate_file = OUT + "/gate.json"
if not os.path.isfile(gate_file):
    checks = {rid: {"confirm_annotated_+0.10": d(rid, "confirm", "annotated_voc") >= 0.10,
                    "tune_annotated_>=0": d(rid, "tune", "annotated_voc") >= 0,
                    "confirm_allframe_-0.20": d(rid, "confirm", "allframe_voc") >= -0.20} for rid in DECIDE}
    passed = all(all(c.values()) for c in checks.values())
    deltas = {rid: {tag: {k: round(d(rid, tag, k), 4) for k in ("annotated_voc", "yowo_list_voc", "allframe_voc")}
                    for tag in ("tune", "confirm")} for rid in DECIDE}
    json.dump({"passed": passed, "checks": checks, "deltas": deltas}, open(gate_file, "w"), indent=2, sort_keys=True)
    ledger("verdict", "--id", RUN_ID, "--verdict", "promoted" if passed else "rejected", "--rationale",
           "Pre-registered gate %s on %s. Frame deltas (fused - plain): %s" % (
               "PASSED" if passed else "FAILED", " and ".join(DECIDE), json.dumps(deltas, sort_keys=True)))
    log("gate: %s %s" % ("PASSED" if passed else "FAILED", json.dumps(deltas, sort_keys=True)))
gate = json.load(open(gate_file))
if not gate["passed"]:
    log("done: no test evaluation")
    raise SystemExit(0)

test_src = SRC + "/V2-WS2.0_test"
plain = ("%s/frames_test_ema.pkl" % test_src, "%s/tube_cache_test_ema_c8.pkl" % test_src)
flipped = ("%s/frames_test_ema_hflip.pkl" % OUT, "%s/tube_cache_test_ema_c8_hflip.pkl" % OUT)
if not (os.path.isfile(flipped[0]) and os.path.isfile(flipped[1])):
    gpu = gpu_with_free(25000)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [PY, "eval_tube_queries.py", "--config", "configs/yolost_v2_ws2_0_tubequeries_30ep.yaml", "--checkpoint",
           "experiments/v2_ws2_0_tubequeries_30ep/ema_final.pt", "--clip_weighting", "hann_peak_norm", "--frame_map",
           "--frame_dump", flipped[0], "--candidate_cache", flipped[1], "--min_length", "8"] + LINKER + ["--hflip"]
    log("test --hflip evaluation on GPU %d" % gpu)
    if subprocess.call(cmd, stdout=open(OUT + "/eval_test_hflip.log", "w"), stderr=subprocess.STDOUT, env=env):
        raise SystemExit("FAILED: see %s/eval_test_hflip.log" % OUT)
fused = ("%s/frames_test_fused.pkl" % OUT, "%s/tube_cache_test_c8_fused.pkl" % OUT)
if not os.path.isfile(fused[0]):
    cmd = [PY, "tools/fuse_flip_views.py", "--dumps", plain[0], flipped[0], "--caches", plain[1], flipped[1],
           "--out-dump", fused[0], "--out-cache", fused[1]]
    if subprocess.call(cmd, stdout=open(OUT + "/fuse_test.log", "w"), stderr=subprocess.STDOUT):
        raise SystemExit("FAILED: fuse")
conv_path = OUT + "/frame_conventions_test_fused.json"
if not os.path.isfile(conv_path):
    if subprocess.call([PY, "tools/score_frame_conventions.py", "--dump", fused[0], "--out", conv_path],
                       stdout=open(conv_path + ".log", "w"), stderr=subprocess.STDOUT):
        raise SystemExit("FAILED: conventions")
conv = json.load(open(conv_path))
plain_conv = json.load(open(test_src + "/frame_conventions_test_plain.json"))
if not os.path.isfile(OUT + "/.recorded_test"):
    for key, evaluator in (("annotated_voc", "moc-act-corrected-frame-hann_peak_norm"),
                           ("yowo_list_voc", "yowo-frame-hann_peak_norm"),
                           ("annotated_trapz", "moc-act-annotated-trapz-hann_peak_norm"),
                           ("allframe_voc", "road-allframe-voc-hann_peak_norm"),
                           ("allframe_trapz", "moc-act-allframe-trapz-hann_peak_norm"),
                           ("yowoformer_style", "yowoformer-released-reimpl-hann_peak_norm"),
                           ("yowoformer_exact", "yowoformer-released-exact-hann_peak_norm")):
        ledger("result", "--id", RUN_ID, "--tag", "V2-WS2.0-ema_final-test-c8-fliptta-frame", "--partition",
               "ucf-test", "--evaluator", evaluator + "-fliptta", "--metric", "frame=%.2f" % conv[key])
    ledger("note", "--id", RUN_ID, "--text",
           "Single test evaluation after the held-out gate passed: V2-WS2.0 with frame-level flip TTA gives annotated "
           "%.2f (single pass %.2f), YOWO list %.2f (%.2f), every-frame %.2f (%.2f), YOWOFormer-exact %.2f (%.2f). "
           "Video numbers are the single-pass ones. Reported as a separate '+ flip TTA (frames)' row." % (
               conv["annotated_voc"], plain_conv["annotated_voc"], conv["yowo_list_voc"], plain_conv["yowo_list_voc"],
               conv["allframe_voc"], plain_conv["allframe_voc"], conv["yowoformer_exact"],
               plain_conv["yowoformer_exact"]))
    open(OUT + "/.recorded_test", "w").write("recorded\n")
log("test with frame-level flip TTA: %s" % json.dumps(
    {k: round(conv[k], 2) for k in ("annotated_voc", "yowo_list_voc", "allframe_voc", "yowoformer_exact")}))
log("done")

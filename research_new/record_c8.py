"""Record one corrected test evaluation (candidate filter 8) in the ledger.

Reads RUN_DIR/snap_frozen.json, snap_control.json, frame_conventions.json written
by research_new/postprocess_c8.sh and appends result rows plus one note. Refuses
to record twice (marker file RUN_DIR/.recorded_c8) or if the frame gate failed.

Usage: python3 record_c8.py LEDGER_ID RUN_DIR
"""
import json
import os
import subprocess
import sys

run_id, run_dir = sys.argv[1], sys.argv[2]
marker = os.path.join(run_dir, ".recorded_c8")
if os.path.exists(marker):
    raise SystemExit("already recorded: " + marker)


def load(name):
    with open(os.path.join(run_dir, name)) as handle:
        return json.load(handle)


frozen = load("snap_frozen.json")
control = load("snap_control.json")
frames = load("frame_conventions.json")
for data, setting in ((frozen, (0.3, 0.75, 0.0, 2)), (control, (0.3, 0.0, 0.0, 0))):
    if len(data["rows"]) != 1:
        raise SystemExit("expected exactly one snap row")
    row = data["rows"][0]
    got = (row["match_iou"], row["box_blend"], row["score_blend"], row["smooth_radius"])
    if got != setting:
        raise SystemExit("unexpected snap setting %r" % (got,))
    if data["protocol"]["min_length"] != 16:
        raise SystemExit("linker min_length is not the frozen 16")
if not frames.get("gate_pass"):
    raise SystemExit("frame gate failed")

PART = "ucf-test"


def ledger(*args):
    cmd = ["python3", "tools/experiment_ledger.py"] + list(args)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(out.stdout.strip())


def r2(value):
    return "%.2f" % value


f, c = frozen["rows"][0], control["rows"][0]
ledger("result", "--id", run_id, "--tag", "ema_final-test-c8-video-snap", "--partition", PART,
       "--evaluator", "moc-act-geometry-snap-c8",
       "--metric", "ap20=" + r2(f["ap20"]), "--metric", "ap50=" + r2(f["ap50"]),
       "--metric", "strict=" + r2(f["strict"]))
ledger("result", "--id", run_id, "--tag", "ema_final-test-c8-video-b0", "--partition", PART,
       "--evaluator", "moc-act-b0linker-c8",
       "--metric", "ap20=" + r2(c["ap20"]), "--metric", "ap50=" + r2(c["ap50"]),
       "--metric", "strict=" + r2(c["strict"]))
FRAME = (
    ("annotated_voc", "moc-act-corrected-frame-hann_peak_norm"),
    ("yowo_list_voc", "yowo-frame-hann_peak_norm"),
    ("annotated_trapz", "moc-act-annotated-trapz-hann_peak_norm"),
    ("allframe_voc", "road-allframe-voc-hann_peak_norm"),
    ("allframe_trapz", "moc-act-allframe-trapz-hann_peak_norm"),
    ("yowoformer_style", "yowoformer-released-reimpl-hann_peak_norm"),
    ("yowoformer_exact", "yowoformer-released-exact-hann_peak_norm"),
)
for key, evaluator in FRAME:
    ledger("result", "--id", run_id, "--tag", "ema_final-test-c8-frame", "--partition", PART,
           "--evaluator", evaluator, "--metric", "frame=" + r2(frames[key]))
ledger("note", "--id", run_id, "--text",
       "Corrected test evaluation 2026-09-24 (%s): tube candidates filtered at 8 visible "
       "frames as on the held-out caches (the first test caches used 16), frozen linker "
       "0.45/0.6/10/16/2 with tube NMS 0.3, frozen geometry snap 0.3/0.75/2; one frozen "
       "setting plus the snap-off control, scored once. Frame rows score the same dump "
       "under each published convention (tools/score_frame_conventions.py). The video "
       "metrics printed at the end of eval_test_c8.log link at min_length 8 and are not "
       "the protocol. Video %.2f/%.2f/%.2f (control %.2f/%.2f/%.2f); frame %.2f annotated, "
       "%.2f every frame." % (run_dir, f["ap20"], f["ap50"], f["strict"], c["ap20"],
                               c["ap50"], c["strict"], frames["annotated_voc"],
                               frames["allframe_voc"]))
open(marker, "w").write("recorded\n")
print("marker written:", marker)

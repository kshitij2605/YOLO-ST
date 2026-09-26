"""Uniform average of the model weights of several checkpoints of one run (weight averaging, one model).

Floating-point tensors of ckpt["model"] are averaged; other entries (e.g. BatchNorm num_batches_tracked) and
the config come from the last input. The result loads like any checkpoint of the run.

Usage: python3 tools/average_checkpoints.py --inputs A.pt B.pt [...] --out AVG.pt
"""
import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise SystemExit("need at least two checkpoints")
    ckpts = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    keys = list(ckpts[-1]["model"].keys())
    for path, ckpt in zip(args.inputs, ckpts):
        if list(ckpt["model"].keys()) != keys:
            raise SystemExit("%s has different model keys" % path)
    averaged = {}
    for key in keys:
        tensors = [ckpt["model"][key] for ckpt in ckpts]
        if tensors[-1].dtype.is_floating_point:
            averaged[key] = (sum(t.double() for t in tensors) / len(tensors)).to(tensors[-1].dtype)
        else:
            averaged[key] = tensors[-1].clone()
    out = dict(ckpts[-1])
    out["model"] = averaged
    out["averaged_from"] = [(path, ckpt.get("epoch")) for path, ckpt in zip(args.inputs, ckpts)]
    torch.save(out, args.out)
    print("averaged %d checkpoints (%d tensors) -> %s" % (len(ckpts), len(keys), args.out))


if __name__ == "__main__":
    main()

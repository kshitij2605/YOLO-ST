"""Verify that a VideoMAE backbone loads into YOLO-ST without silent damage.

Motivation
----------
The v1 record contains a direct VideoMAE-Large transplant that scored 20.42
frame mAP, while the same VideoMAE-Large weights reach 92.48 frame mAP inside
the reproduced YOWOFormer-L in this very repository. That gap is far too large
to be an architecture effect, so it is almost certainly a loading or
preprocessing failure that no assertion caught. This tool is the assertion.

It reports, for a given ``model_id`` and target geometry:

1. **Weight fidelity.** Every tensor of the wrapper's backbone is compared with
   a freshly loaded reference ``VideoMAEForVideoClassification.videomae``:
   missing, unexpected, shape-mismatched and value-differing tensors are
   counted and listed. Anything other than the position table differing is a
   failure.
2. **Position-embedding interpolation.** Whether the sinusoidal table was
   resized, from which shape to which, whether it stayed finite, and whether
   its per-token norm distribution is preserved. VideoMAE stores this table as
   a plain tensor rather than a parameter or buffer, so it is invisible to
   ``state_dict`` and to ``.to(device)``; it is checked explicitly.
3. **Forward equivalence.** At the reference geometry, where no interpolation is
   needed, the wrapper's backbone and the reference must produce identical
   hidden states. This separates "the weights loaded" from "the geometry
   change altered the features".
4. **Parameter accounting.** Total, trainable and frozen counts, plus which
   transformer blocks are unfrozen.

Exit status is non-zero if any check fails, so this can gate a training launch.

Usage::

    python tools/verify_backbone_load.py \
        --model-id MCG-NJU/videomae-base-finetuned-kinetics \
        --img-size 224 --backbone-frames 32 --unfreeze-last-n 6

CPU only by default; pass ``--device cuda`` to check device placement too.
"""

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import VideoMAEForVideoClassification  # noqa: E402

from yolost.model_videomae import YOLOST_VideoMAE  # noqa: E402


DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


def _count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def compare_state_dicts(reference, candidate, atol=0.0):
    """Compare two state dicts tensor by tensor."""
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    report = {
        'missing': sorted(reference_keys - candidate_keys),
        'unexpected': sorted(candidate_keys - reference_keys),
        'shape_mismatch': [],
        'value_mismatch': [],
        'identical': 0,
    }
    for key in sorted(reference_keys & candidate_keys):
        left = reference[key]
        right = candidate[key]
        if tuple(left.shape) != tuple(right.shape):
            report['shape_mismatch'].append(
                {'key': key, 'reference': list(left.shape),
                 'candidate': list(right.shape)}
            )
            continue
        left_f = left.detach().float().cpu()
        right_f = right.detach().float().cpu()
        if torch.equal(left_f, right_f):
            report['identical'] += 1
        else:
            delta = (left_f - right_f).abs().max().item()
            if delta <= atol:
                report['identical'] += 1
            else:
                report['value_mismatch'].append(
                    {'key': key, 'max_abs_delta': delta}
                )
    return report


def check_position_embeddings(reference_model, wrapper):
    """Inspect the sinusoidal position table, which is not in the state dict."""
    reference_table = reference_model.videomae.embeddings.position_embeddings
    candidate_table = wrapper.backbone.embeddings.position_embeddings
    reference_table = reference_table.detach().float().cpu()
    candidate_table = candidate_table.detach().float().cpu()

    expected = wrapper.token_t * wrapper.grid_size * wrapper.grid_size
    result = {
        'reference_shape': list(reference_table.shape),
        'candidate_shape': list(candidate_table.shape),
        'expected_tokens': int(expected),
        'interpolated': list(reference_table.shape) != list(
            candidate_table.shape
        ),
        'candidate_is_finite': bool(torch.isfinite(candidate_table).all()),
        'candidate_token_count_correct':
            int(candidate_table.shape[1]) == int(expected),
        'reference_mean_token_norm':
            float(reference_table.norm(dim=-1).mean()),
        'candidate_mean_token_norm':
            float(candidate_table.norm(dim=-1).mean()),
        'is_parameter': isinstance(
            wrapper.backbone.embeddings.position_embeddings,
            torch.nn.Parameter,
        ),
        'in_state_dict': any(
            'position_embeddings' in key
            for key in wrapper.backbone.state_dict()
        ),
    }
    reference_norm = result['reference_mean_token_norm']
    candidate_norm = result['candidate_mean_token_norm']
    result['token_norm_ratio'] = (
        candidate_norm / reference_norm if reference_norm else float('nan')
    )
    return result


def check_forward_equivalence(reference_model, wrapper, device, dtype,
                              seed=0, tolerance=1e-4):
    """At the reference geometry the two backbones must agree."""
    config = reference_model.config
    frames = int(config.num_frames)
    size = int(config.image_size)

    reference_backbone = reference_model.videomae.to(device=device).eval()
    candidate_backbone = wrapper.backbone.to(device=device).eval()

    generator = torch.Generator(device='cpu').manual_seed(seed)
    pixel_values = torch.randn(
        1, frames, 3, size, size, generator=generator
    ).to(device=device, dtype=dtype)

    # The wrapper rewrites patch-embedding metadata for its own geometry; the
    # reference geometry must still be accepted.
    reference_state = {
        'image_size': reference_backbone.embeddings.patch_embeddings.image_size,
        'num_patches': reference_backbone.embeddings.patch_embeddings.num_patches,
    }
    candidate_patch = candidate_backbone.embeddings.patch_embeddings
    saved = {
        'image_size': candidate_patch.image_size,
        'num_patches': candidate_patch.num_patches,
        'position_embeddings':
            candidate_backbone.embeddings.position_embeddings,
        'num_patches_emb': candidate_backbone.embeddings.num_patches,
    }
    candidate_patch.image_size = reference_state['image_size']
    candidate_patch.num_patches = reference_state['num_patches']
    candidate_backbone.embeddings.num_patches = reference_state['num_patches']
    candidate_backbone.embeddings.position_embeddings = (
        reference_model.videomae.embeddings.position_embeddings
    )
    try:
        with torch.no_grad():
            reference_out = reference_backbone(
                pixel_values=pixel_values
            ).last_hidden_state.float()
            candidate_out = candidate_backbone(
                pixel_values=pixel_values
            ).last_hidden_state.float()
    finally:
        candidate_patch.image_size = saved['image_size']
        candidate_patch.num_patches = saved['num_patches']
        candidate_backbone.embeddings.num_patches = saved['num_patches_emb']
        candidate_backbone.embeddings.position_embeddings = (
            saved['position_embeddings']
        )

    delta = (reference_out - candidate_out).abs()
    return {
        'reference_shape': list(reference_out.shape),
        'candidate_shape': list(candidate_out.shape),
        'max_abs_delta': float(delta.max()),
        'mean_abs_delta': float(delta.mean()),
        'reference_std': float(reference_out.std()),
        'candidate_std': float(candidate_out.std()),
        'tolerance': tolerance,
        'passed': bool(delta.max() <= tolerance),
    }


def verify(model_id, img_size=224, clip_length=64, backbone_frames=32,
           unfreeze_last_n=6, dtype='float32', device='cpu',
           num_classes=24, tolerance=1e-4, run_forward=True):
    """Run every check and return a report dictionary."""
    torch_dtype = DTYPES[dtype]
    device = torch.device(device)

    reference_model = VideoMAEForVideoClassification.from_pretrained(model_id)
    reference_model.eval()

    wrapper = YOLOST_VideoMAE(
        num_classes=num_classes,
        img_size=img_size,
        clip_length=clip_length,
        model_id=model_id,
        freeze_backbone=False,
        unfreeze_last_n_blocks=unfreeze_last_n,
        backbone_frames=backbone_frames,
        dtype=dtype,
    )
    wrapper.eval()

    reference_backbone_state = {
        key: value for key, value in
        reference_model.videomae.state_dict().items()
    }
    candidate_backbone_state = wrapper.backbone.state_dict()

    report = {
        'model_id': model_id,
        'geometry': {
            'img_size': img_size,
            'clip_length': clip_length,
            'backbone_frames': backbone_frames,
            'token_t': int(wrapper.token_t),
            'grid_size': int(wrapper.grid_size),
            'patch_size': int(wrapper.patch_size),
            'tubelet_size': int(wrapper.tubelet_size),
            'hidden_size': int(reference_model.config.hidden_size),
            'reference_num_frames': int(reference_model.config.num_frames),
            'reference_image_size': int(reference_model.config.image_size),
        },
        'weights': compare_state_dicts(
            reference_backbone_state, candidate_backbone_state
        ),
        'position_embeddings': check_position_embeddings(
            reference_model, wrapper
        ),
    }

    total, trainable = _count_parameters(wrapper)
    backbone_total, backbone_trainable = _count_parameters(wrapper.backbone)
    unfrozen_blocks = sorted({
        int(name.split('.')[2])
        for name, parameter in wrapper.backbone.named_parameters()
        if parameter.requires_grad and name.startswith('encoder.layer.')
        and name.split('.')[2].isdigit()
    })
    report['parameters'] = {
        'model_total': total,
        'model_trainable': trainable,
        'backbone_total': backbone_total,
        'backbone_trainable': backbone_trainable,
        'unfrozen_encoder_blocks': list(unfrozen_blocks),
        'requested_unfreeze_last_n': int(unfreeze_last_n),
    }

    if run_forward:
        report['forward'] = check_forward_equivalence(
            reference_model, wrapper, device=device,
            dtype=torch.float32, tolerance=tolerance,
        )

    failures = []
    weights = report['weights']
    if weights['missing']:
        failures.append(f"{len(weights['missing'])} backbone tensors missing")
    if weights['shape_mismatch']:
        failures.append(
            f"{len(weights['shape_mismatch'])} backbone tensors "
            'changed shape'
        )
    if weights['value_mismatch']:
        failures.append(
            f"{len(weights['value_mismatch'])} backbone tensors "
            'changed value'
        )
    position = report['position_embeddings']
    if not position['candidate_is_finite']:
        failures.append('position embeddings contain non-finite values')
    if not position['candidate_token_count_correct']:
        failures.append('position embedding token count does not match geometry')
    if report['parameters']['backbone_trainable'] == 0 and unfreeze_last_n > 0:
        failures.append('no backbone parameter is trainable despite unfreezing')
    if (unfreeze_last_n > 0
            and len(report['parameters']['unfrozen_encoder_blocks'])
            != unfreeze_last_n):
        failures.append(
            'unfrozen block count '
            f"{len(report['parameters']['unfrozen_encoder_blocks'])} "
            f'!= requested {unfreeze_last_n}'
        )
    if run_forward and not report['forward']['passed']:
        failures.append(
            'forward outputs differ at the reference geometry '
            f"(max abs delta {report['forward']['max_abs_delta']:.3e})"
        )

    report['failures'] = failures
    report['passed'] = not failures
    return report


def _print_report(report):
    geometry = report['geometry']
    weights = report['weights']
    position = report['position_embeddings']
    parameters = report['parameters']

    print(f"model_id                : {report['model_id']}")
    print(
        'geometry                : '
        f"{geometry['backbone_frames']} frames @ {geometry['img_size']}px "
        f"-> tokens {geometry['token_t']}x{geometry['grid_size']}"
        f"x{geometry['grid_size']}, hidden {geometry['hidden_size']}"
    )
    print(
        'reference geometry      : '
        f"{geometry['reference_num_frames']} frames @ "
        f"{geometry['reference_image_size']}px"
    )
    print()
    print(f"tensors identical       : {weights['identical']}")
    print(f"tensors missing         : {len(weights['missing'])}")
    print(f"tensors unexpected      : {len(weights['unexpected'])}")
    print(f"tensors shape-mismatched: {len(weights['shape_mismatch'])}")
    print(f"tensors value-changed   : {len(weights['value_mismatch'])}")
    for entry in weights['shape_mismatch'][:10]:
        print(f"   shape {entry['key']}: {entry['reference']} -> "
              f"{entry['candidate']}")
    for entry in weights['value_mismatch'][:10]:
        print(f"   value {entry['key']}: max delta "
              f"{entry['max_abs_delta']:.3e}")
    for key in weights['missing'][:10]:
        print(f"   missing {key}")
    print()
    print(
        'position table          : '
        f"{position['reference_shape']} -> {position['candidate_shape']} "
        f"(interpolated={position['interpolated']})"
    )
    print(
        'position token norm     : '
        f"{position['reference_mean_token_norm']:.4f} -> "
        f"{position['candidate_mean_token_norm']:.4f} "
        f"(ratio {position['token_norm_ratio']:.4f})"
    )
    print(
        'position storage        : '
        f"parameter={position['is_parameter']}, "
        f"in_state_dict={position['in_state_dict']}"
    )
    print()
    print(f"params total            : {parameters['model_total'] / 1e6:.1f}M")
    print(
        'params trainable        : '
        f"{parameters['model_trainable'] / 1e6:.1f}M"
    )
    print(
        'backbone trainable      : '
        f"{parameters['backbone_trainable'] / 1e6:.1f}M of "
        f"{parameters['backbone_total'] / 1e6:.1f}M"
    )
    print(
        'unfrozen blocks         : '
        f"{parameters['unfrozen_encoder_blocks']}"
    )
    if 'forward' in report:
        forward = report['forward']
        print()
        print(
            'forward at reference    : max abs delta '
            f"{forward['max_abs_delta']:.3e} "
            f"(tolerance {forward['tolerance']:.1e}), "
            f"passed={forward['passed']}"
        )
    print()
    if report['passed']:
        print('RESULT: PASS')
    else:
        print('RESULT: FAIL')
        for failure in report['failures']:
            print(f'  - {failure}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '--model-id', default='MCG-NJU/videomae-base-finetuned-kinetics'
    )
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--clip-length', type=int, default=64)
    parser.add_argument('--backbone-frames', type=int, default=32)
    parser.add_argument('--unfreeze-last-n', type=int, default=6)
    parser.add_argument('--num-classes', type=int, default=24)
    parser.add_argument(
        '--dtype', default='float32', choices=sorted(DTYPES)
    )
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--skip-forward', action='store_true')
    parser.add_argument('--json-out', default=None)
    args = parser.parse_args(argv)

    report = verify(
        model_id=args.model_id,
        img_size=args.img_size,
        clip_length=args.clip_length,
        backbone_frames=args.backbone_frames,
        unfreeze_last_n=args.unfreeze_last_n,
        dtype=args.dtype,
        device=args.device,
        num_classes=args.num_classes,
        tolerance=args.tolerance,
        run_forward=not args.skip_forward,
    )
    _print_report(report)
    if args.json_out:
        with open(args.json_out, 'w') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f'wrote {args.json_out}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Export Depth Anything 3 (SMALL) to ONNX for the browser demo.

The exported graph takes a batch of views and returns per-view depth, depth
confidence, world-to-camera extrinsics and intrinsics. The number of views is a
dynamic axis, so the browser can send 2..N frames without re-exporting.

Usage (from the repo root, with the DA3 source and weights on disk):

    python tools/export_onnx.py \
        --da3-src /path/to/depth-anything-3/src \
        --config  /path/to/depth-anything-3/src/depth_anything_3/configs/da3-small.yaml \
        --weights /path/to/model.safetensors \
        --out     build/da3-small.onnx \
        --res     504 --fp16

--fp16 also writes build/da3-small-fp16.onnx, the single-file copy that the
page loads from model/.

--da3-src is the directory that contains the `depth_anything_3` package.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _patch_rope_for_export(grid_side: int):
    """Make the DA3 RoPE forward traceable.

    The upstream forward reads `int(positions.max())` to size the frequency
    table. That is data-dependent, so the tracer cannot record it. With a fixed
    input resolution the max position is always grid_side - 1, so use that
    constant instead. The rest of the forward is unchanged.
    """
    from depth_anything_3.model.dinov2.layers import rope as rope_mod

    def forward(self, tokens, positions):
        feature_dim = tokens.size(-1) // 2
        # Patch positions are shifted by +1 (0 is reserved for special tokens),
        # so the largest index is grid_side and the table needs grid_side + 1 rows.
        max_position = grid_side + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp
        )
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp
        )
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    rope_mod.RotaryPositionEmbedding2D.forward = forward


def _patch_sincos_float32():
    """Keep the DPT position embedding in float32.

    The upstream helper builds its frequencies in float64, which the ONNX
    graph then mixes with float32 tensors (Einsum type error). Only the
    frequency table changes precision, and it is a constant at a fixed size.
    """
    from depth_anything_3.model.utils import head_utils

    def make_sincos_pos_embed(embed_dim, pos, omega_0=100):
        omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
        omega = omega / (embed_dim / 2.0)
        omega = 1.0 / omega_0**omega
        out = torch.einsum("m,d->md", pos.reshape(-1), omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()

    head_utils.make_sincos_pos_embed = make_sincos_pos_embed


class ExportWrapper(nn.Module):
    """Expose only the tensors the browser needs, as a flat tuple."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        # images: (1, V, 3, H, W), ImageNet-normalised
        out = self.net(images)
        return out["depth"], out["depth_conf"], out["extrinsics"], out["intrinsics"]


def load_model(da3_src: str, config: str, weights: str):
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.model_loading import convert_general_state_dict

    model = create_object(OmegaConf.load(config)).eval()
    state = convert_general_state_dict(load_file(weights))
    # Same key handling as the upstream loader: strip the "model." prefix.
    state = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys, check the DA3 version: {unexpected[:5]}")
    print(f"loaded weights, {len(missing)} keys not in checkpoint: {missing}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--da3-src", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="build/da3-small.onnx")
    p.add_argument("--res", type=int, default=504, help="square input side, multiple of 14")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--fp16", action="store_true", help="also write a float16 copy")
    args = p.parse_args()

    assert args.res % 14 == 0, "DINOv2 patch size is 14"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.da3_src)
    _patch_rope_for_export(args.res // 14)
    _patch_sincos_float32()
    net = load_model(args.da3_src, args.config, args.weights)
    wrapper = ExportWrapper(net).eval()

    dummy = torch.rand(1, 3, 3, args.res, args.res)
    with torch.no_grad():
        ref = wrapper(dummy)

    views = torch.export.Dim("views", min=1, max=64)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["images"],
        output_names=["depth", "depth_conf", "extrinsics", "intrinsics"],
        dynamic_shapes={"images": {1: views}},
        opset_version=args.opset,
        dynamo=True,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Check the exported graph against PyTorch on a different view count, so the
    # dynamic axis is exercised too.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    probe = torch.rand(1, 4, 3, args.res, args.res)
    with torch.no_grad():
        torch_out = wrapper(probe)
    onnx_out = sess.run(None, {"images": probe.numpy()})
    for name, a, b in zip(["depth", "depth_conf", "extrinsics", "intrinsics"], torch_out, onnx_out):
        err = np.abs(a.numpy() - b).max()
        print(f"{name:12s} shape={tuple(b.shape)} max_abs_err={err:.2e}")
        if not np.isfinite(b).all():
            raise SystemExit(f"{name} has non-finite values")

    if args.fp16:
        import onnx
        from onnxconverter_common import float16

        fp16_path = out_path.with_name(out_path.stem + "-fp16.onnx")
        model_fp16 = float16.convert_float_to_float16(onnx.load(str(out_path)), keep_io_types=True)
        onnx.save(model_fp16, str(fp16_path))
        print(f"wrote {fp16_path} ({fp16_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

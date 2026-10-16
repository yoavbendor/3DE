"""Export Depth Anything 3 (SMALL) to ONNX for the browser demo.

The exported graph takes a batch of views and returns per-view depth, depth
confidence, world-to-camera extrinsics and intrinsics. The number of views, the
height and the width are all dynamic, so one model file serves every input
size. Height and width must be multiples of 14 (the DINOv2 patch size), and the
page feeds images at their own aspect ratio, as the upstream preprocessing does.

Usage (from the repo root, with the DA3 source and weights on disk):

    python tools/export_onnx.py \
        --da3-src /path/to/depth-anything-3/src \
        --config  /path/to/depth-anything-3/src/depth_anything_3/configs/da3-small.yaml \
        --weights /path/to/model.safetensors \
        --out     build/da3-small.onnx \
        --fp16

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

# Largest patch grid side the RoPE table covers (128 * 14 = 1792 px).
MAX_GRID = 128


def _patch_rope_for_export():
    """Make the DA3 RoPE traceable for any input size.

    The upstream forward sizes its frequency table from `int(positions.max())`,
    which is data-dependent. Row i of that table depends only on i, so a fixed
    table with MAX_GRID + 1 rows gives identical values for every grid up to
    MAX_GRID patches per side.

    The position grid is also cached by Python (height, width) keys upstream,
    which pins the traced shape. Build it from the symbolic sizes instead.
    """
    from depth_anything_3.model.dinov2.layers import rope as rope_mod

    def forward(self, tokens, positions):
        feature_dim = tokens.size(-1) // 2
        # Patch positions are shifted by +1 (0 is reserved for special tokens).
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, MAX_GRID + 1, tokens.device, tokens.dtype
        )
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp
        )
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp
        )
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    def get_positions(self, batch_size, height, width, device):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        positions = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)
        return positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    rope_mod.RotaryPositionEmbedding2D.forward = forward
    rope_mod.PositionGetter.__call__ = get_positions


def _patch_sincos_float32():
    """Keep the DPT position embedding in float32.

    The upstream helper builds its frequencies in float64, which the ONNX
    graph then mixes with float32 tensors (Einsum type error). The official
    GitHub version of this helper also uses float32.
    """
    from depth_anything_3.model.utils import head_utils

    def make_sincos_pos_embed(embed_dim, pos, omega_0=100):
        omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
        omega = omega / (embed_dim / 2.0)
        omega = 1.0 / omega_0**omega
        out = torch.einsum("m,d->md", pos.reshape(-1), omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()

    head_utils.make_sincos_pos_embed = make_sincos_pos_embed


def _patch_pos_encoding_dynamic():
    """Resize the DINOv2 position embedding with run-time sampling coordinates.

    Upstream calls F.interpolate(scale_factor=(sx, sy)) with Python floats
    derived from the input size, which ONNX export bakes in as constants, so
    any other input size fails. This samples the same bicubic positions as
    upstream (scale = (n + interpolate_offset) / M, source = (dst + 0.5) / scale
    - 0.5, edge pixels replicated), via grid_sample, with the coordinates
    computed from the symbolic grid size.
    """
    import math

    import torch.nn.functional as F
    from depth_anything_3.model.dinov2 import vision_transformer as vt

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        n_pos = self.pos_embed.shape[1] - 1
        m = int(math.sqrt(n_pos))
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        dim = pos_embed.shape[-1]
        patch = pos_embed[:, 1:].reshape(1, m, m, dim).permute(0, 3, 1, 2)
        rows = torch.arange(w // self.patch_size, dtype=torch.float32, device=x.device)
        cols = torch.arange(h // self.patch_size, dtype=torch.float32, device=x.device)
        n_rows = torch.ones_like(rows).sum()
        n_cols = torch.ones_like(cols).sum()
        offset = self.interpolate_offset or 0.0
        src_r = (rows + 0.5) / ((n_rows + offset) / m) - 0.5
        src_c = (cols + 0.5) / ((n_cols + offset) / m) - 0.5
        grid_y, grid_x = torch.meshgrid((src_r + 0.5) / m * 2 - 1, (src_c + 0.5) / m * 2 - 1, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)[None]
        out = F.grid_sample(patch, grid, mode="bicubic", padding_mode="border", align_corners=False)
        patch_pos_embed = out.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    vt.DinoVisionTransformer.interpolate_pos_encoding = interpolate_pos_encoding


def _uv_grid(width, height, dtype=None, device=None):
    """Tensor-only version of head_utils.create_uv_grid.

    Same values as upstream (linspace over +-span * (n - 1) / n, normalised by
    the diagonal), but built from aranges so symbolic sizes stay symbolic. The
    aspect ratio comes from the grid itself: every DPT feature map is a whole
    multiple of the 14 px patch grid, so width / height equals the image's W / H.
    """
    xs = torch.arange(width, device=device, dtype=torch.float32)
    ys = torch.arange(height, device=device, dtype=torch.float32)
    w = torch.ones_like(xs).sum()
    h = torch.ones_like(ys).sum()
    aspect = w / h
    diag = torch.sqrt(aspect * aspect + 1.0)
    x = (2.0 * xs - (w - 1.0)) / w * (aspect / diag)
    y = (2.0 * ys - (h - 1.0)) / h * (1.0 / diag)
    uu, vv = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((uu, vv), dim=-1).to(dtype)


def _patch_pos_embed_dynamic():
    """Give the DPT heads a UV position grid that works with symbolic sizes."""
    from depth_anything_3.model import dpt, dualdpt
    from depth_anything_3.model.utils.head_utils import position_grid_to_embed

    def _add_pos_embed(self, x, W, H, ratio=0.1):
        pw, ph = x.shape[-1], x.shape[-2]
        pe = _uv_grid(pw, ph, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    dpt.DPT._add_pos_embed = _add_pos_embed
    dualdpt.DualDPT._add_pos_embed = _add_pos_embed


def _patch_head_output_size():
    """Keep the heads' output size symbolic.

    The heads compute h_out = int(ph * patch_size / down_ratio). int() on a
    symbolic size freezes it to the example's value, so every other input size
    came out at the example resolution. With integer patch_size and down_ratio
    floor division gives the same number and stays symbolic.
    """
    import inspect
    import textwrap

    from depth_anything_3.model import dpt, dualdpt

    replacements = {
        "int(ph * self.patch_size / self.down_ratio)": "ph * self.patch_size // self.down_ratio",
        "int(pw * self.patch_size / self.down_ratio)": "pw * self.patch_size // self.down_ratio",
    }
    for module, cls in ((dpt, dpt.DPT), (dualdpt, dualdpt.DualDPT)):
        src = textwrap.dedent(inspect.getsource(cls._forward_impl))
        for old_expr, new_expr in replacements.items():
            if src.count(old_expr) != 1:
                raise RuntimeError(f"{cls.__name__}._forward_impl changed upstream; check {old_expr!r}")
            src = src.replace(old_expr, new_expr)
        namespace = {}
        exec(compile(src, f"<patched {cls.__name__}._forward_impl>", "exec"), module.__dict__, namespace)
        cls._forward_impl = namespace["_forward_impl"]


def _patch_no_view_chunking():
    """Run the DPT heads on all views in one pass.

    The heads split views into chunks of 8 with a Python branch on the view
    count, which pins the traced view count. The browser never sends more than
    a handful of views, so one pass is the same computation.
    """
    from depth_anything_3.model import dpt, dualdpt

    for cls in (dpt.DPT, dualdpt.DualDPT):
        original = cls.forward

        def forward(self, *args, _original=original, **kwargs):
            kwargs["chunk_size"] = None
            return _original(self, *args, **kwargs)

        cls.forward = forward


class ExportWrapper(nn.Module):
    """Expose only the tensors the browser needs, as a flat tuple."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):
        # images: (1, V, 3, H, W), ImageNet-normalised
        out = self.net(images)
        return out["depth"], out["depth_conf"], out["extrinsics"], out["intrinsics"]


def load_model(config: str, weights: str):
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object

    model = create_object(OmegaConf.load(config)).eval()
    # The Hugging Face checkpoint's keys already match the network, prefixed
    # with "model." (the API wrapper's attribute), which is how the official
    # from_pretrained path loads it. Do not run convert_general_state_dict: it
    # is meant for raw training checkpoints and drops the trained camera token,
    # which leaves multi-view poses random from run to run.
    raw = load_file(weights)
    state = {k[len("model.") :]: v for k, v in raw.items() if k.startswith("model.")}
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
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--fp16", action="store_true", help="also write a float16 copy")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.da3_src)
    _patch_rope_for_export()
    _patch_sincos_float32()
    _patch_no_view_chunking()
    _patch_pos_embed_dynamic()
    _patch_pos_encoding_dynamic()
    _patch_head_output_size()
    net = load_model(args.config, args.weights)
    wrapper = ExportWrapper(net).eval()

    # Example input: 3 portrait views at 336 x 252. All three axes are dynamic.
    dummy = torch.rand(1, 3, 3, 24 * 14, 18 * 14)
    views = torch.export.Dim("views", min=1, max=64)
    gh = torch.export.Dim("grid_h", min=4, max=MAX_GRID)
    gw = torch.export.Dim("grid_w", min=4, max=MAX_GRID)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["images"],
        output_names=["depth", "depth_conf", "extrinsics", "intrinsics"],
        dynamic_shapes={"images": {1: views, 3: 14 * gh, 4: 14 * gw}},
        opset_version=args.opset,
        dynamo=True,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Check the exported graph against PyTorch at shapes other than the example,
    # so every dynamic axis is exercised.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    for v, h, w in [(1, 504, 378), (2, 252, 336), (4, 224, 168)]:
        probe = torch.rand(1, v, 3, h, w)
        with torch.no_grad():
            torch_out = wrapper(probe)
        onnx_out = sess.run(None, {"images": probe.numpy()})
        for name, a, b in zip(["depth", "depth_conf", "extrinsics", "intrinsics"], torch_out, onnx_out):
            err = np.abs(a.numpy() - b).max()
            print(f"V={v} {h}x{w} {name:12s} shape={tuple(b.shape)} max_abs_err={err:.2e}")
            if not np.isfinite(b).all():
                raise SystemExit(f"{name} has non-finite values")

    if args.fp16:
        fp16_path = out_path.with_name(out_path.stem + "-fp16.onnx")
        write_fp16_weights(out_path, fp16_path)
        print(f"wrote {fp16_path} ({fp16_path.stat().st_size / 1e6:.1f} MB)")

def write_fp16_weights(src: Path, dst: Path, min_elements: int = 1024):
    """Store large float32 weights as float16, cast back to float32 on load.

    Halves the file size (about 55 MB for SMALL) while every operation still
    runs in float32. A whole-graph float16 conversion breaks this graph: it
    rewrites the Cast nodes in the dynamic-shape arithmetic and ONNX Runtime
    then rejects mismatched types.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src))
    casts, new_inits = [], []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and arr.size >= min_elements:
            half_name = init.name + "__fp16"
            new_inits.append(numpy_helper.from_array(arr.astype(np.float16), half_name))
            casts.append(helper.make_node("Cast", [half_name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "__to_fp32"))
        else:
            new_inits.append(init)
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(casts + nodes)
    onnx.save(model, str(dst))


if __name__ == "__main__":
    main()

# Object to 3D in the browser

A single static page that turns a few photos (or a short video) of an object into a 3D point cloud. Depth Anything 3 (SMALL) runs on the device through ONNX Runtime Web, so no images leave the phone.

Live page: `https://<your-user>.github.io/3DE/` once GitHub Pages is enabled (see below).

## Files

| Path | What it is |
| --- | --- |
| `index.html` | The whole app: capture, inference, point cloud, viewer, PLY export. |
| `model/da3-small-dyn-fp16.onnx` | DA3-SMALL exported to ONNX with dynamic views, height and width; float16 weights, float32 compute (55 MB). Served by Pages. |
| `tools/export_onnx.py` | Reproduces the ONNX export from the official PyTorch weights. |
| `tools/requirements.txt` | Python deps for the export. |

## Run it locally

```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

Any static server works. Camera access and WebGPU need `localhost` or HTTPS, and the page is HTTPS on Pages.

## Enable GitHub Pages

Settings → Pages → Build and deployment → Source: *Deploy from a branch*, branch `main`, folder `/ (root)`. The `.nojekyll` file stops Pages from running Jekyll.

## How it works

1. Photos or video frames are squashed to 504×504 and ImageNet-normalised. A video contributes evenly spaced frames. Up to 8 views are used.
2. All views go through the model together, so the network can relate them. For each view it returns depth, a depth confidence, a world-to-camera pose and intrinsics.
3. Each valid pixel is unprojected with its depth, intrinsics and pose into one shared frame, and coloured from the frame.
4. The viewer keeps the most confident points (slider). Download writes a binary PLY with colours.

The backend is WebGPU when the browser has it (recent iOS Safari), otherwise WebAssembly on the CPU. The model is cached in the browser's Cache Storage after the first load.

## Re-export the model

The export needs the upstream package and weights. Use the official repo and the Hugging Face checkpoint:

```bash
pip install -r tools/requirements.txt
git clone https://github.com/ByteDance-Seed/depth-anything-3 /tmp/da3
huggingface-cli download depth-anything/DA3-SMALL model.safetensors --local-dir /tmp/da3-weights

python tools/export_onnx.py \
  --da3-src /tmp/da3/src \
  --config  /tmp/da3/src/depth_anything_3/configs/da3-small.yaml \
  --weights /tmp/da3-weights/model.safetensors \
  --fp16
mv build/da3-small-fp16.onnx model/da3-small-dyn-fp16.onnx
```

The export makes the number of views, the height and the width dynamic, so one file serves every input size. To trace that, the script patches a few upstream spots without changing the maths: a fixed-size RoPE table, position grids and the DINOv2 position-embedding resize computed from the symbolic size, the DPT heads' output size kept symbolic, and one sin/cos table in float32. It checks the exported graph against PyTorch at three different shapes before writing the copy with float16 weights.

It loads the Hugging Face weights directly (as the official `from_pretrained` does), not through `convert_general_state_dict`, which is meant for raw training checkpoints and drops the trained camera token.

The page feeds each photo at its own aspect ratio (longest side rounded to a multiple of 14, as upstream does) and picks the longest side per run: `planRun()` in `index.html` keeps as many photos as the memory budget allows (up to 6), then uses the largest size that fits. Peak memory per size and view count is in `PEAK_MB`.

## Known limits

- **Speed on phones is not measured yet.** On a headless Linux CPU, 4 views took about 47 s on the WASM backend. An iPhone 15 with WebGPU should be much faster, but that needs a real test.
- **Memory.** The budget is 820 MB (`MEM_BUDGET_MB`); one photo runs at 504 px, six at 252 px. If Safari reloads the tab, lower the budget.
- **Single-threaded WASM.** GitHub Pages cannot send the cross-origin isolation headers, so threads are off.
- **Multi-view accuracy.** DA3-SMALL's depth maps for different views disagree by a few percent of the scene size (DA3-LARGE about 2%, measured on the same photos), so overlapping views show as slightly offset layers rather than one surface. This is the model, not the page: the official code gives the same numbers. The page compensates with a clean-up (on by default, checkbox under the viewer): it drops flying pixels at depth edges (`EDGE_REL`, 5% jump between neighbours) and drops a later view's points that land on a surface an earlier view already covers (`MERGE_TOL`, 15% depth tolerance), so each region is drawn once. It also treats all photos as one camera at one setting: one shared focal length with square pixels, self-calibrated by picking the value (0.7–1.4× the model's estimate) that makes the views' depths agree best, kept only if it improves agreement by at least 3%.
- **Object-only captures.** Background and table surfaces show up as points. Use the confidence slider, or shoot against a plain background.

## Licences

- Depth Anything 3 weights (`DA3-SMALL`): Apache-2.0, per the model card on Hugging Face.
- The Hugging Face *Space* that hosts the demo is CC-BY-NC-4.0. This project does not copy its UI. It only uses the model package for the export, and the upstream repo's licence applies to that code. Check it before redistributing.
- ONNX Runtime Web (MIT) and three.js (MIT) load from jsDelivr.

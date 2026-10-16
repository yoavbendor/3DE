# Object to 3D in the browser

A single static page that turns a few photos (or a short video) of an object into a 3D point cloud. Depth Anything 3 (SMALL) runs on the device through ONNX Runtime Web, so no images leave the phone.

Live page: `https://<your-user>.github.io/3DE/` once GitHub Pages is enabled (see below).

## Files

| Path | What it is |
| --- | --- |
| `index.html` | The whole app: capture, inference, point cloud, viewer, PLY export. |
| `model/da3-small-fp16.onnx` | DA3-SMALL exported to ONNX, float16 weights (56 MB). Served by Pages. |
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
  --res 504 --fp16
mv build/da3-small-fp16.onnx model/
```

The script patches three spots in the upstream model so `torch.onnx` can trace it. None of them change the maths at the fixed 504 input: the RoPE table size is a constant instead of `int(positions.max())`, and one sin/cos table is kept in float32. It checks the exported graph against PyTorch before it writes the fp16 copy.

The page picks the input size per run: `planRun()` in `index.html` keeps as many photos as the memory budget allows, then uses the largest size that fits. Sizes are 504, 336 and 252 px (multiples of 14), each with its own model file in `model/`. Peak memory per size and view count is in `PEAK_MB`. If you export a new size, add its model to `MODELS` in `worker.js` and its numbers to `PEAK_MB`.

## Known limits

- **Speed on phones is not measured yet.** On a headless Linux CPU, 4 views took about 47 s on the WASM backend. An iPhone 15 with WebGPU should be much faster, but that needs a real test.
- **Memory.** 8 views at 504² is the default cap. If Safari reloads the tab, lower `MAX_VIEWS` in `index.html`.
- **Single-threaded WASM.** GitHub Pages cannot send the cross-origin isolation headers, so threads are off.
- **Pose quality.** The checkpoint includes a `camera_token` for the multi-view path, but the upstream loader deletes it, so the model uses a randomly initialised token. This app loads the weights the same way upstream does. Check the reconstructions on real captures before relying on the poses.
- **Object-only captures.** Background and table surfaces show up as points. Use the confidence slider, or shoot against a plain background.

## Licences

- Depth Anything 3 weights (`DA3-SMALL`): Apache-2.0, per the model card on Hugging Face.
- The Hugging Face *Space* that hosts the demo is CC-BY-NC-4.0. This project does not copy its UI. It only uses the model package for the export, and the upstream repo's licence applies to that code. Check it before redistributing.
- ONNX Runtime Web (MIT) and three.js (MIT) load from jsDelivr.

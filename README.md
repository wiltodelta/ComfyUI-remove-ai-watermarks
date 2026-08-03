# ComfyUI-remove-ai-watermarks

ComfyUI custom nodes for [remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks):
remove visible AI watermarks, erase arbitrary regions, detect known marks, and
strip invisible SynthID watermarks by diffusion regeneration, all inside a
ComfyUI graph.

## Nodes

| Node | What it does |
| --- | --- |
| **Remove Visible Watermark (RAIW)** | Removes every detected registered AI-provenance mark when `mark = auto`, or forces one selected mark. Choose the fill backend and detection sensitivity. No GPU is required for detection or the default cv2 fill. |
| **Detect Visible Watermark (RAIW)** | Reports per-mark detection confidence on the input image. Outputs a text report, a `detected` boolean, the best confidence, and the detected mark key. |
| **Erase Region (RAIW)** | Inpaints whatever a `MASK` covers. Choose cv2 for the dependency-free path or LaMa for a heavier learned fill. |
| **Remove Invisible Watermark / SynthID (RAIW)** | Diffusion regeneration that defeats the SynthID pixel watermark. Choose ControlNet, SDXL, or the CUDA-only Qwen-Image plus Z-Image profile for the highest face fidelity. |

All pixel nodes operate in-memory on the ComfyUI image batch. The invisible node
writes each frame to a temp file, runs the diffusion engine, and reads it back.

## Install

### Via ComfyUI Manager

Search for "Remove AI Watermarks" in ComfyUI Manager and install.

### Manual

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/wiltodelta/ComfyUI-remove-ai-watermarks
pip install -r ComfyUI-remove-ai-watermarks/requirements.txt
```

The package installs `remove-ai-watermarks[qwen-zimage]`, including the normal
GPU diffusion stack and the separate DiffSynth runtime. Model weights are not
bundled and download on first use.

Optional backends: `pip install "remove-ai-watermarks[migan]"` for the
memory-conscious MI-GAN fill and `pip install "remove-ai-watermarks[lama]"` for
the heavier LaMa fill.

## Notes

- ComfyUI image tensors carry no file metadata, so the invisible node's
  vendor-adaptive strength default falls back to the unknown-vendor value
  defined by the installed library. Set the `strength` input above 0 to override
  it.
- Both profiles are CUDA-only: `qwen-zimage` (the default) and `sdxl-zimage`,
  the same recipe and face stage on an SDXL global pass. There is no CPU or MPS
  path, so the node has no device widget.
- The invisible node exposes no `steps`, `guidance_scale`, `model` or `device`
  input. Each profile pins its model stack, its per-stage distilled schedule and
  CFG 1.0, so a widget for any of them could only produce an error inside the
  run. Seed 0 is the certified default.
- `adaptive_polish` is three-way. `profile default` lets the library decide (off
  for `qwen-zimage`, whose output already matches the input's detail level; on
  for `sdxl-zimage`); `on`/`off` override it. A workflow saved when this input
  was a checkbox still loads, and its stored true/false counts as an override.
- Use `tile` for large inputs and `cpu_offload` to stream both stacks instead of
  pinning them in VRAM.
- Proprietary invisible-watermark verification is vendor-specific. Use the
  matching provider oracle. Higher `strength` removes more but drifts further
  from the original.
- `remove-ai-watermarks` depends on `opencv-python-headless`. If your ComfyUI
  install already ships `opencv-python`, both provide `cv2` and coexist; if you
  hit a cv2 conflict, keep a single OpenCV distribution in the environment.

## Release synchronization

The registry package has its own version. Each `remove-ai-watermarks` release
dispatches the synchronization workflow with its exact version. The workflow
updates the dependency floor, runs compatibility tests, bumps the node patch
version, and publishes only after those tests pass. A daily scheduled run is
the recovery path for an interrupted release dispatch. A failed compatibility
test blocks publication instead of exposing an incompatible node update.

## License

Apache-2.0, matching the upstream library.

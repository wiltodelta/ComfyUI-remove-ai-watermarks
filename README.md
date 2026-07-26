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
| **Remove Invisible Watermark / SynthID (RAIW)** | SDXL diffusion regeneration that defeats the SynthID pixel watermark while preserving text/face structure (canny ControlNet). Requires the GPU/ML extra. |

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

The base install (`remove-ai-watermarks`) covers the three pixel nodes. The
**invisible / SynthID** node additionally needs the diffusion stack (torch,
diffusers; multi-GB):

```sh
pip install "remove-ai-watermarks[gpu]"
```

Optional backends: `pip install "remove-ai-watermarks[migan]"` for the
memory-conscious MI-GAN fill, `pip install "remove-ai-watermarks[lama]"` for the
heavier LaMa fill, and `pip install "remove-ai-watermarks[esrgan]"` for the
Real-ESRGAN upscaler in the invisible node.

## Notes

- ComfyUI image tensors carry no file metadata, so the invisible node's
  vendor-adaptive strength default falls back to the unknown-vendor value
  defined by the installed library. Set the `strength` input above 0 to override
  it.
- Proprietary invisible-watermark verification is vendor-specific. Use the
  matching provider oracle. Higher `strength` removes more but drifts further
  from the original.
- `remove-ai-watermarks` depends on `opencv-python-headless`. If your ComfyUI
  install already ships `opencv-python`, both provide `cv2` and coexist; if you
  hit a cv2 conflict, keep a single OpenCV distribution in the environment.

## Release synchronization

The registry package has its own version. A scheduled GitHub Actions workflow
checks the latest `remove-ai-watermarks` release, updates the dependency floor,
runs compatibility tests, bumps the node patch version, and publishes only
after those tests pass. A failed compatibility test blocks publication instead
of exposing an incompatible node update.

## License

Apache-2.0, matching the upstream library.

# ComfyUI-remove-ai-watermarks

ComfyUI custom nodes for
[remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks):
identify provenance, remove visible and invisible AI watermarks, strip AI
metadata, run the complete image pipeline, and erase arbitrary regions inside a
ComfyUI graph.

## Nodes

| Node | What it does |
| --- | --- |
| **Remove Visible Watermark (RAIW)** | Removes every detected registered AI-provenance mark when `mark = auto`, or forces one selected mark. Choose the fill backend and detection sensitivity. No GPU is required for detection or the default cv2 fill. |
| **Detect Visible Watermark (RAIW)** | Reports per-mark detection confidence on the input image. Outputs a text report, a `detected` boolean, the best confidence, and the detected mark key. |
| **Erase Region (RAIW)** | Inpaints whatever a `MASK` covers. Choose cv2 for the dependency-free path or LaMa for a heavier learned fill. |
| **Remove Invisible Watermark / SynthID (RAIW)** | Diffusion regeneration through the CUDA-only `qwen-zimage`, `sdxl-zimage`, or `chroma-zimage` profile. |
| **Identify Provenance (RAIW)** | Reads an original file and returns the versioned provenance report, verdict, platform, and confidence. |
| **Strip AI Metadata (RAIW)** | Losslessly strips and verifies AI metadata from an original file, then returns the cleaned image and path. |
| **Remove All Watermarks (RAIW)** | Runs visible removal, conditional invisible removal, and verified metadata stripping against one original file. |

The four pixel nodes operate in-memory on ComfyUI image batches. The three
provenance-aware nodes accept a file path because ComfyUI `IMAGE` tensors do not
carry the original C2PA, EXIF, XMP, IPTC, or container data. Their source and
output paths must be accessible to the ComfyUI server. A blank output path writes
`<source>_clean.<ext>` beside the source without overwriting it.

## Install

### Via ComfyUI Manager

Search for "Remove AI Watermarks" in ComfyUI Manager and install.

### Manual

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/wiltodelta/ComfyUI-remove-ai-watermarks
pip install -r ComfyUI-remove-ai-watermarks/requirements.txt
```

The package installs `remove-ai-watermarks[qwen-zimage,heif]`, including the
pixel runtime, HEIC/HEIF/AVIF decoding, GPU diffusion stack, and DiffSynth
runtime. Model weights are not bundled and download on first use.

Optional backends: `pip install "remove-ai-watermarks[migan]"` for the
memory-conscious MI-GAN fill and `pip install "remove-ai-watermarks[lama]"` for
the heavier LaMa fill.

## Notes

- ComfyUI image tensors carry no file metadata. The in-memory invisible node
  therefore uses resolution-adaptive strength for `qwen-zimage` and the
  unknown-vendor strength for `sdxl-zimage`. `Remove All Watermarks` reads the
  original file and can use its vendor provenance. Set `strength` above 0 to
  override either default.
- All three profiles are CUDA-only: `qwen-zimage` (the default), `sdxl-zimage`, and `chroma-zimage`,
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
- The selected extras install `opencv-python-headless`. If your ComfyUI install
  already ships `opencv-python`, both provide `cv2` and coexist; if you hit a
  cv2 conflict, keep a single OpenCV distribution in the environment.

## Release synchronization

The registry package has its own version. Each `remove-ai-watermarks` release
dispatches the synchronization workflow with its exact version. The workflow
updates the dependency floor, runs compatibility tests, bumps the node patch
version, and publishes only after those tests pass. A daily scheduled run is
the recovery path for an interrupted release dispatch. A failed compatibility
test blocks publication instead of exposing an incompatible node update.

### The package name is the registry id

`name` in `pyproject.toml` is the ComfyUI Registry node id, not a local label.
Publishing under a different one creates a second listing and orphans every
existing install, so it is not something to change casually.

It deliberately matches the PyPI library this node depends on. uv reads that as
a self-dependency and refuses to resolve the project, which looks like a naming
mistake and is not one: this repository is installed with pip, through
`requirements.txt` and `scripts/test.sh` in CI, and pip resolves the dependency
from PyPI without complaint. There is no `uv.lock` here and `uv run` is not part
of the workflow.

Editing `pyproject.toml` on `main` triggers the publish workflow, so treat any
change to this file as a release action rather than an edit.

## License

Apache-2.0, matching the upstream library.

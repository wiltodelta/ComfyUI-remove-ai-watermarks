"""ComfyUI nodes wrapping the remove-ai-watermarks library.

ComfyUI passes images as torch tensors shaped ``(B, H, W, C)``, float32 in
``[0, 1]``, RGB. The remove-ai-watermarks library works on per-image BGR uint8
numpy arrays (OpenCV convention), so each node converts at the boundary and
processes the batch frame by frame.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

CATEGORY = "remove-ai-watermarks"


# --- tensor <-> numpy boundary helpers -------------------------------------


def _tensor_to_bgr_list(image: "torch.Tensor") -> list[np.ndarray[Any, Any]]:
    """ComfyUI IMAGE (B, H, W, C) float RGB [0, 1] -> list of BGR uint8 frames."""
    arr = image.detach().cpu().numpy()
    frames: list[np.ndarray[Any, Any]] = []
    for i in range(arr.shape[0]):
        rgb = np.clip(arr[i] * 255.0, 0, 255).astype(np.uint8)
        if rgb.ndim == 2:  # grayscale safety
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = rgb[..., :3]  # drop alpha if present
        frames.append(rgb[..., ::-1].copy())  # RGB -> BGR
    return frames


def _bgr_list_to_tensor(frames: list[np.ndarray[Any, Any]]) -> "torch.Tensor":
    """List of BGR uint8 frames -> ComfyUI IMAGE (B, H, W, C) float RGB [0, 1]."""
    tensors = []
    for bgr in frames:
        rgb = bgr[..., :3][..., ::-1].copy()  # BGR -> RGB
        tensors.append(torch.from_numpy(rgb.astype(np.float32) / 255.0))
    return torch.stack(tensors, dim=0)


def _mask_to_uint8_list(mask: "torch.Tensor", count: int) -> list[np.ndarray[Any, Any]]:
    """ComfyUI MASK (B, H, W) or (H, W) float [0, 1] -> list of uint8 masks (255 = erase).

    A single mask is broadcast across the whole image batch.
    """
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[None, ...]
    out: list[np.ndarray[Any, Any]] = []
    for i in range(count):
        m = arr[i] if i < arr.shape[0] else arr[-1]
        out.append((m > 0.5).astype(np.uint8) * 255)
    return out


# --- nodes -----------------------------------------------------------------


class RAIWRemoveVisibleWatermark:
    """Remove a known visible AI watermark (Gemini sparkle, Doubao, Jimeng, Samsung)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        from remove_ai_watermarks import watermark_registry

        marks = ["auto", *watermark_registry.mark_keys()]
        return {
            "required": {
                "image": ("IMAGE",),
                "mark": (marks, {"default": "auto"}),
            },
            "optional": {
                "inpaint": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "remove"
    CATEGORY = CATEGORY

    def remove(self, image: "torch.Tensor", mark: str, inpaint: bool = True) -> tuple[Any, str]:
        from remove_ai_watermarks import watermark_registry

        frames = _tensor_to_bgr_list(image)
        out: list[np.ndarray[Any, Any]] = []
        infos: list[str] = []
        for bgr in frames:
            if mark == "auto":
                best = watermark_registry.best_auto_mark(bgr)
                if best is None:
                    out.append(bgr)
                    infos.append("no visible mark detected")
                    continue
                known = watermark_registry.get_mark(best.key)
                result, _ = known.remove(bgr, inpaint=inpaint)
                infos.append(f"removed {best.key} (conf {best.confidence:.2f})")
            else:
                known = watermark_registry.get_mark(mark)
                result, _ = known.remove(bgr, inpaint=inpaint, force=True)
                infos.append(f"removed {mark} (forced)")
            out.append(result)
        return (_bgr_list_to_tensor(out), " | ".join(infos))


class RAIWDetectVisibleWatermark:
    """Detect known visible AI watermarks and report per-mark confidence."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING", "BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("report", "detected", "confidence", "mark")
    FUNCTION = "detect"
    CATEGORY = CATEGORY

    def detect(self, image: "torch.Tensor") -> tuple[str, bool, float, str]:
        from remove_ai_watermarks import watermark_registry

        bgr = _tensor_to_bgr_list(image)[0]  # report on the first frame
        detections = watermark_registry.detect_marks(bgr)
        lines = [
            f"{d.label}: {'YES' if d.detected else 'no'} ({d.confidence:.2f})" for d in detections
        ]
        fired = [d for d in detections if d.detected]
        best = max(detections, key=lambda d: d.confidence) if detections else None
        any_detected = bool(fired)
        confidence = best.confidence if best else 0.0
        mark_key = best.key if (best and best.detected) else ""
        return ("\n".join(lines), any_detected, confidence, mark_key)


class RAIWEraseRegion:
    """Erase an arbitrary region (given by a MASK) via inpainting (cv2 or LaMa)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "backend": (["cv2", "lama"], {"default": "cv2"}),
                "dilate": ("INT", {"default": 3, "min": 0, "max": 64}),
                "cv2_method": (["telea", "ns"], {"default": "telea"}),
                "cv2_radius": ("INT", {"default": 6, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "erase"
    CATEGORY = CATEGORY

    def erase(
        self,
        image: "torch.Tensor",
        mask: "torch.Tensor",
        backend: str = "cv2",
        dilate: int = 3,
        cv2_method: str = "telea",
        cv2_radius: int = 6,
    ) -> tuple[Any]:
        from remove_ai_watermarks.region_eraser import erase as erase_region

        frames = _tensor_to_bgr_list(image)
        masks = _mask_to_uint8_list(mask, len(frames))
        out = [
            erase_region(
                bgr,
                mask=m,
                backend=backend,
                dilate=dilate,
                cv2_method=cv2_method,  # type: ignore[arg-type]
                cv2_radius=cv2_radius,
            )
            for bgr, m in zip(frames, masks)
        ]
        return (_bgr_list_to_tensor(out),)


class RAIWRemoveInvisibleWatermark:
    """Remove invisible AI watermarks (SynthID) via SDXL diffusion regeneration.

    Requires the GPU/ML extra of the library:
        pip install "remove-ai-watermarks[gpu]"

    ComfyUI tensors carry no file metadata, so the vendor-adaptive strength
    default falls back to the unknown-vendor value. Set ``strength`` > 0 to
    override it explicitly.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "pipeline": (["controlnet", "sdxl"], {"default": "controlnet"}),
                "strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "controlnet_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "humanize": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "unsharp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "max_resolution": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "min_resolution": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "adaptive_polish": ("BOOLEAN", {"default": True}),
                "upscaler": (["lanczos", "esrgan"], {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove"
    CATEGORY = CATEGORY

    def remove(
        self,
        image: "torch.Tensor",
        pipeline: str = "controlnet",
        strength: float = 0.0,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 0,
        device: str = "auto",
        controlnet_scale: float = 1.0,
        humanize: float = 0.0,
        unsharp: float = 0.0,
        max_resolution: int = 0,
        min_resolution: int = 1024,
        adaptive_polish: bool = True,
        upscaler: str = "lanczos",
    ) -> tuple[Any]:
        try:
            from remove_ai_watermarks.invisible_engine import InvisibleEngine, is_available
        except Exception as exc:  # diffusers/torch not installed
            raise RuntimeError(
                "The invisible-watermark node needs the GPU/ML extra. Install it with: "
                "pip install 'remove-ai-watermarks[gpu]'"
            ) from exc
        if not is_available():
            raise RuntimeError(
                "Diffusion dependencies are unavailable. Install the GPU/ML extra: "
                "pip install 'remove-ai-watermarks[gpu]'"
            )

        engine = InvisibleEngine(
            device=None if device == "auto" else device,
            pipeline=pipeline,
            controlnet_conditioning_scale=controlnet_scale,
            progress_callback=lambda msg: log.info("invisible: %s", msg),
        )

        frames = _tensor_to_bgr_list(image)
        out: list[np.ndarray[Any, Any]] = []
        with tempfile.TemporaryDirectory(prefix="raiw_comfy_") as tmp:
            tmp_dir = Path(tmp)
            for idx, bgr in enumerate(frames):
                from remove_ai_watermarks import image_io

                src = tmp_dir / f"in_{idx}.png"
                dst = tmp_dir / f"out_{idx}.png"
                image_io.imwrite(str(src), bgr)
                result_path = engine.remove_watermark(
                    image_path=src,
                    output_path=dst,
                    strength=strength if strength > 0 else None,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    humanize=humanize,
                    unsharp=unsharp,
                    max_resolution=max_resolution,
                    min_resolution=min_resolution,
                    adaptive_polish=adaptive_polish,
                    upscaler=upscaler,
                )
                cleaned = image_io.imread(str(result_path))
                out.append(cleaned)
        return (_bgr_list_to_tensor(out),)


NODE_CLASS_MAPPINGS = {
    "RAIWRemoveVisibleWatermark": RAIWRemoveVisibleWatermark,
    "RAIWDetectVisibleWatermark": RAIWDetectVisibleWatermark,
    "RAIWEraseRegion": RAIWEraseRegion,
    "RAIWRemoveInvisibleWatermark": RAIWRemoveInvisibleWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RAIWRemoveVisibleWatermark": "Remove Visible Watermark (RAIW)",
    "RAIWDetectVisibleWatermark": "Detect Visible Watermark (RAIW)",
    "RAIWEraseRegion": "Erase Region (RAIW)",
    "RAIWRemoveInvisibleWatermark": "Remove Invisible Watermark / SynthID (RAIW)",
}

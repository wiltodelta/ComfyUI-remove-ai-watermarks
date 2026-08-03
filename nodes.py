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
from typing import TYPE_CHECKING, Any, get_args

import numpy as np

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

CATEGORY = "remove-ai-watermarks"

# The library's two remaining profiles. Duplicated as a literal rather than imported
# at module scope on purpose: INPUT_TYPES runs when ComfyUI loads the node pack, and a
# hard library import there would take the visible-mark nodes down with it whenever the
# diffusion dependencies are absent. _profile_choices() prefers the library's own list
# when it can be read cheaply, and tests/test_nodes.py asserts this fallback still
# matches it -- so a future profile change surfaces here rather than in a user's queue.
_FALLBACK_PROFILE_CHOICES = ("qwen-zimage", "sdxl-zimage")
DEFAULT_PROFILE = "qwen-zimage"

# The library resolves an unset adaptive polish per profile (off for qwen-zimage, whose
# output already matches the input's detail level; on for sdxl-zimage). A ComfyUI
# BOOLEAN cannot express "unset", so this is a three-way widget instead of a checkbox
# that would silently override the profile.
_POLISH_PROFILE_DEFAULT = "profile default"
_POLISH_CHOICES = [_POLISH_PROFILE_DEFAULT, "on", "off"]


def _profile_choices() -> tuple[str, ...]:
    """Pipeline names for the widget, from the library when it is importable."""
    try:
        from remove_ai_watermarks._internal.watermark_profiles import PROFILE_CHOICES
    except Exception:
        return _FALLBACK_PROFILE_CHOICES
    return tuple(PROFILE_CHOICES)


def _resolve_polish_widget(value: str | bool) -> bool | None:
    """Map the widget to the library's tri-state ``adaptive_polish``.

    Saved workflows from before this widget existed carry a raw bool, so accept one
    and treat it as an explicit choice rather than raising on an old graph.
    """
    if isinstance(value, bool):
        return value
    if value == _POLISH_PROFILE_DEFAULT:
        return None
    return value == "on"


# --- tensor <-> numpy boundary helpers -------------------------------------


def _tensor_to_bgr_list(image: torch.Tensor) -> list[np.ndarray[Any, Any]]:
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


def _bgr_list_to_tensor(frames: list[np.ndarray[Any, Any]]) -> torch.Tensor:
    """List of BGR uint8 frames -> ComfyUI IMAGE (B, H, W, C) float RGB [0, 1]."""
    import torch

    tensors = []
    for bgr in frames:
        rgb = bgr[..., :3][..., ::-1].copy()  # BGR -> RGB
        tensors.append(torch.from_numpy(rgb.astype(np.float32) / 255.0))
    return torch.stack(tensors, dim=0)


def _mask_to_uint8_list(mask: torch.Tensor, count: int) -> list[np.ndarray[Any, Any]]:
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
    """Remove a registered visible AI-provenance mark."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        from remove_ai_watermarks import watermark_registry

        marks = ["auto", *watermark_registry.mark_keys()]
        backends = list(get_args(watermark_registry.Backend))
        sensitivities = list(get_args(watermark_registry.Sensitivity))
        return {
            "required": {
                "image": ("IMAGE",),
                "mark": (marks, {"default": "auto"}),
            },
            "optional": {
                # Retained so existing workflows keep their widget layout. Since
                # library 0.16 every visible mark uses localize -> fill, so this
                # legacy switch no longer changes the removal path.
                "inpaint": ("BOOLEAN", {"default": True}),
                "backend": (backends, {"default": "auto"}),
                "sensitivity": (sensitivities, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "remove"
    CATEGORY = CATEGORY

    def remove(
        self,
        image: torch.Tensor,
        mark: str,
        inpaint: bool = True,
        backend: str = "auto",
        sensitivity: str = "auto",
    ) -> tuple[Any, str]:
        from remove_ai_watermarks import watermark_registry

        del inpaint
        frames = _tensor_to_bgr_list(image)
        out: list[np.ndarray[Any, Any]] = []
        infos: list[str] = []
        for bgr in frames:
            if mark == "auto":
                result, labels = watermark_registry.remove_auto_marks(
                    bgr,
                    backend=backend,
                    sensitivity=sensitivity,
                )
                infos.append(", ".join(labels) if labels else "no visible mark detected")
            else:
                known = watermark_registry.get_mark(mark)
                result, _ = known.remove(bgr, backend=backend, force=True)
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

    def detect(self, image: torch.Tensor) -> tuple[str, bool, float, str]:
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
    """Erase an arbitrary region through a cv2, MI-GAN, or LaMa fill."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        from remove_ai_watermarks import region_eraser

        backends = list(get_args(region_eraser.Backend))
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "backend": (backends, {"default": "cv2"}),
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
        image: torch.Tensor,
        mask: torch.Tensor,
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
    """Remove invisible AI watermarks (SynthID) via diffusion regeneration.

    The node package installs the full qwen-zimage dependency group, which also
    includes the normal GPU/ML dependencies. The qwen-zimage profile itself is
    CUDA-only.

    ComfyUI tensors carry no file metadata, so the vendor-adaptive strength
    default falls back to the unknown-vendor value. Set ``strength`` > 0 to
    override it explicitly.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "pipeline": (list(_profile_choices()), {"default": DEFAULT_PROFILE}),
                "strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "controlnet_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "humanize": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "unsharp": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "max_resolution": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "adaptive_polish": (_POLISH_CHOICES, {"default": _POLISH_PROFILE_DEFAULT}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "tile": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "tile_overlap": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove"
    CATEGORY = CATEGORY

    def __init__(self) -> None:
        self._engine: Any = None
        self._engine_config: tuple[str, float, bool] | None = None

    def remove(
        self,
        image: torch.Tensor,
        pipeline: str = DEFAULT_PROFILE,
        strength: float = 0.0,
        seed: int = 0,
        controlnet_scale: float = 1.0,
        humanize: float = 0.0,
        unsharp: float = 0.0,
        max_resolution: int = 0,
        adaptive_polish: str | bool = _POLISH_PROFILE_DEFAULT,
        cpu_offload: bool = False,
        tile: bool = False,
        tile_size: int = 1024,
        tile_overlap: int = 128,
    ) -> tuple[Any]:
        try:
            from remove_ai_watermarks.invisible_engine import (
                InvisibleEngine,
                is_available,
            )
        except Exception as exc:  # diffusers/torch not installed
            raise RuntimeError(
                "The invisible-watermark node needs its diffusion dependencies. "
                "Reinstall the node dependencies or run: "
                "pip install 'remove-ai-watermarks[qwen-zimage]'"
            ) from exc
        if not is_available():
            raise RuntimeError(
                "Diffusion dependencies are unavailable. Reinstall the node dependencies "
                "or run: pip install 'remove-ai-watermarks[qwen-zimage]'"
            )

        # No device widget: both profiles are CUDA-only, so the only value a user
        # could usefully pick is the one detection already returns.
        engine_config = (pipeline, controlnet_scale, cpu_offload)
        if self._engine_config != engine_config:
            self._engine = InvisibleEngine(
                pipeline=pipeline,
                controlnet_conditioning_scale=controlnet_scale,
                cpu_offload=cpu_offload,
                progress_callback=lambda msg: log.info("invisible: %s", msg),
            )
            self._engine_config = engine_config
        engine = self._engine

        profile_adaptive_polish = _resolve_polish_widget(adaptive_polish)

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
                    seed=seed,
                    humanize=humanize,
                    unsharp=unsharp,
                    max_resolution=max_resolution,
                    adaptive_polish=profile_adaptive_polish,
                    tile=tile,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
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

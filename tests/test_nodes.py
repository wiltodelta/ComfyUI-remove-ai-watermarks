"""Compatibility tests for the remove-ai-watermarks library boundary."""

from __future__ import annotations

import inspect
from typing import Any, get_args

import numpy as np

import nodes


def _stub_tensor_boundary(monkeypatch: Any) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(nodes, "_tensor_to_bgr_list", lambda image: [frame])
    monkeypatch.setattr(nodes, "_bgr_list_to_tensor", lambda frames: "tensor")


def test_visible_auto_uses_current_registry_api(monkeypatch: Any) -> None:
    from remove_ai_watermarks import watermark_registry

    _stub_tensor_boundary(monkeypatch)
    calls: list[tuple[str, str]] = []

    def remove_auto_marks(
        image: np.ndarray[Any, Any],
        *,
        backend: str,
        sensitivity: str,
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        calls.append((backend, sensitivity))
        return image, ["Google Gemini sparkle"]

    monkeypatch.setattr(watermark_registry, "remove_auto_marks", remove_auto_marks)

    output, info = nodes.RAIWRemoveVisibleWatermark().remove(
        object(),
        "auto",
        backend="cv2",
        sensitivity="strict",
    )

    assert output == "tensor"
    assert info == "Google Gemini sparkle"
    assert calls == [("cv2", "strict")]


def test_input_choices_follow_library_types() -> None:
    from remove_ai_watermarks import region_eraser, watermark_registry

    visible = nodes.RAIWRemoveVisibleWatermark.INPUT_TYPES()["optional"]
    erase = nodes.RAIWEraseRegion.INPUT_TYPES()["optional"]

    assert visible["backend"][0] == list(get_args(watermark_registry.Backend))
    assert visible["sensitivity"][0] == list(get_args(watermark_registry.Sensitivity))
    assert erase["backend"][0] == list(get_args(region_eraser.Backend))


def test_visible_forced_mark_uses_backend_api(monkeypatch: Any) -> None:
    from remove_ai_watermarks import watermark_registry

    _stub_tensor_boundary(monkeypatch)
    calls: list[tuple[str, bool]] = []

    class StubMark:
        def remove(
            self,
            image: np.ndarray[Any, Any],
            *,
            backend: str,
            force: bool,
        ) -> tuple[np.ndarray[Any, Any], tuple[int, int, int, int]]:
            calls.append((backend, force))
            return image, (0, 0, 1, 1)

    monkeypatch.setattr(watermark_registry, "get_mark", lambda key: StubMark())

    output, info = nodes.RAIWRemoveVisibleWatermark().remove(
        object(),
        "yuanbao",
        backend="migan",
    )

    assert output == "tensor"
    assert info == "removed yuanbao (forced)"
    assert calls == [("migan", True)]


def test_library_contract_contains_every_node_parameter() -> None:
    from remove_ai_watermarks import region_eraser, watermark_registry
    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    contracts = [
        (
            watermark_registry.remove_auto_marks,
            {"image", "sensitivity", "backend"},
        ),
        (
            watermark_registry.KnownMark.remove,
            {"self", "image", "backend", "force"},
        ),
        (
            region_eraser.erase,
            {"image_bgr", "mask", "backend", "dilate", "cv2_method", "cv2_radius"},
        ),
        (
            InvisibleEngine.__init__,
            {
                "self",
                "device",
                "pipeline",
                "progress_callback",
                "controlnet_conditioning_scale",
            },
        ),
        (
            InvisibleEngine.remove_watermark,
            {
                "self",
                "image_path",
                "output_path",
                "strength",
                "num_inference_steps",
                "guidance_scale",
                "seed",
                "humanize",
                "max_resolution",
                "min_resolution",
                "unsharp",
                "adaptive_polish",
                "upscaler",
            },
        ),
    ]

    for target, required in contracts:
        assert required <= set(inspect.signature(target).parameters)

"""Compatibility tests for the remove-ai-watermarks library boundary."""

from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path
from typing import Any, get_args

import numpy as np

import nodes


def test_root_init_loads_without_package_context() -> None:
    root_init = Path(__file__).parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("_comfy_root", root_init)

    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = ""
    spec.loader.exec_module(module)
    assert module.NODE_CLASS_MAPPINGS == nodes.NODE_CLASS_MAPPINGS


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


def test_invisible_node_offers_no_knob_the_profiles_cannot_honor() -> None:
    """Model id, step count, CFG and device are fixed by the profile.

    The library deleted every one of them; a widget for any of them can only produce
    an error several frames below the queue prompt. (Which list of pipelines the
    widget offers is asserted against the library in
    ``test_the_pipeline_widget_offers_exactly_the_library_profiles``.)
    """
    types = nodes.RAIWRemoveInvisibleWatermark.INPUT_TYPES()
    widgets = set(types["required"]) | set(types["optional"])

    assert not widgets & {"steps", "guidance_scale", "device", "model", "min_resolution", "upscaler"}


def test_package_installs_qwen_zimage_extra() -> None:
    root = Path(__file__).parents[1]
    project = (root / "pyproject.toml").read_text()
    requirements = (root / "requirements.txt").read_text().splitlines()
    test_script = (root / "scripts" / "test.sh").read_text()

    match = re.search(r'dependencies = \["([^"]+)"\]', project)
    assert match is not None
    dependency = match.group(1)
    assert dependency.startswith("remove-ai-watermarks[qwen-zimage]>=")
    assert requirements == [dependency]
    assert '"remove-ai-watermarks[qwen-zimage]==$LIBRARY_VERSION"' in test_script


def test_the_node_defers_every_fixed_setting_to_the_library(monkeypatch: Any) -> None:
    from remove_ai_watermarks import image_io, invisible_engine

    _stub_tensor_boundary(monkeypatch)
    constructor_calls: list[dict[str, Any]] = []
    removal_calls: list[dict[str, Any]] = []

    class StubEngine:
        def __init__(self, **kwargs: Any) -> None:
            constructor_calls.append(kwargs)

        def remove_watermark(self, **kwargs: Any) -> Path:
            removal_calls.append(kwargs)
            return kwargs["output_path"]

    monkeypatch.setattr(invisible_engine, "InvisibleEngine", StubEngine)
    monkeypatch.setattr(invisible_engine, "is_available", lambda: True)
    monkeypatch.setattr(image_io, "imwrite", lambda path, image: True)
    monkeypatch.setattr(
        image_io,
        "imread",
        lambda path: np.zeros((8, 8, 3), dtype=np.uint8),
    )

    node = nodes.RAIWRemoveInvisibleWatermark()
    outputs = [
        node.remove(
            object(),
            pipeline="qwen-zimage",
            strength=0.0,
            seed=0,
            adaptive_polish="profile default",
            cpu_offload=True,
            tile=True,
            tile_size=768,
            tile_overlap=96,
        )
        for _ in range(2)
    ]

    assert outputs == [("tensor",), ("tensor",)]
    # Built once for two prompts, and with no device: the engine detects CUDA itself.
    assert constructor_calls == [
        {
            "pipeline": "qwen-zimage",
            "controlnet_conditioning_scale": 1.0,
            "cpu_offload": True,
            "progress_callback": constructor_calls[0]["progress_callback"],
        }
    ]
    assert len(removal_calls) == 2
    for call in removal_calls:
        assert {
            key: value
            for key, value in call.items()
            if key not in {"image_path", "output_path"}
        } == {
            "strength": None,
            "seed": 0,
            "humanize": 0.0,
            "unsharp": 0.0,
            "max_resolution": 0,
            # None, not False: the library picks the profile's answer. Sending False
            # here would pin qwen-zimage's default and silently override sdxl-zimage's.
            "adaptive_polish": None,
            "tile": True,
            "tile_size": 768,
            "tile_overlap": 96,
        }


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
                "pipeline",
                "progress_callback",
                "controlnet_conditioning_scale",
                "cpu_offload",
            },
        ),
        (
            InvisibleEngine.remove_watermark,
            {
                "self",
                "image_path",
                "output_path",
                "strength",
                "seed",
                "humanize",
                "max_resolution",
                "unsharp",
                "adaptive_polish",
                "tile",
                "tile_size",
                "tile_overlap",
            },
        ),
    ]

    for target, required in contracts:
        assert required <= set(inspect.signature(target).parameters)


def test_the_node_never_passes_a_parameter_the_library_dropped() -> None:
    """A subset check cannot catch a parameter the node sends and the library removed.

    ``num_inference_steps``, ``guidance_scale``, ``min_resolution`` and ``upscaler``
    were all still being passed here after the library deleted them, and the contract
    above stayed green because it only asserts the library HAS what the node needs --
    never that the node sends nothing the library refuses.
    """
    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    accepted = set(inspect.signature(InvisibleEngine.remove_watermark).parameters)
    source = (Path(__file__).parents[1] / "nodes.py").read_text()
    call = source[source.index("result_path = engine.remove_watermark(") :]
    call = call[: call.index("\n                )")]
    sent = set(re.findall(r"^\s+(\w+)=", call, re.MULTILINE))

    assert sent, "could not parse the remove_watermark call site"
    assert sent <= accepted, f"node sends parameters the library does not accept: {sorted(sent - accepted)}"


def test_the_pipeline_widget_offers_exactly_the_library_profiles() -> None:
    """The widget list and its offline fallback must both track the library.

    The node shipped `controlnet`/`sdxl` with `controlnet` as the DEFAULT long after
    the library deleted those profiles, so every default-configured run raised
    "Unsupported pipeline" -- against the node's own pinned library floor.
    """
    from remove_ai_watermarks._internal.watermark_profiles import DEFAULT_PROFILE, PROFILE_CHOICES

    widget = nodes.RAIWRemoveInvisibleWatermark.INPUT_TYPES()["required"]["pipeline"]
    assert tuple(widget[0]) == tuple(PROFILE_CHOICES)
    assert widget[1]["default"] == DEFAULT_PROFILE
    assert nodes._FALLBACK_PROFILE_CHOICES == tuple(PROFILE_CHOICES)
    assert nodes.DEFAULT_PROFILE == DEFAULT_PROFILE


def test_the_polish_widget_maps_to_the_library_tri_state() -> None:
    assert nodes._resolve_polish_widget("profile default") is None
    assert nodes._resolve_polish_widget("on") is True
    assert nodes._resolve_polish_widget("off") is False
    # A graph saved before the widget became three-way carries a raw bool.
    assert nodes._resolve_polish_widget(True) is True
    assert nodes._resolve_polish_widget(False) is False

"""Typed ComfyUI nodes for the library's complete operation catalog."""
from __future__ import annotations

import json
from typing import Any, ClassVar

from remove_ai_watermarks.integration import operation_catalog, run_operation, widget_arguments


class CatalogOperation:
    operation: ClassVar[dict[str, Any]]
    CATEGORY = "remove-ai-watermarks/operations"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output_path", "report", "exit_code")
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        # File contents and existing output paths can change outside the graph.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        inputs = {}
        for parameter in cls.operation["parameters"]:
            name, schema = parameter["name"], parameter["schema"]
            default = schema.get("default")
            options = {"tooltip": schema.get("description", name)}
            if "enum" in schema:
                kind = list(schema["enum"])
                options["default"] = default if default is not None else kind[0]
            elif schema["type"] == "boolean":
                if default is None:
                    kind = ["default", "enabled", "disabled"]
                    options["default"] = "default"
                else:
                    kind = "BOOLEAN"
                    options["default"] = default
            elif schema["type"] in {"integer", "number"} and default is not None:
                kind = "INT" if schema["type"] == "integer" else "FLOAT"
                options["default"] = default
                if name == "seed":
                    options["control_after_generate"] = False
                if schema["type"] == "number":
                    options.update(round=0.0001, step=0.01)
                for bound, key in [("minimum", "min"), ("maximum", "max")]:
                    if bound in schema:
                        options[key] = schema[bound]
            else:
                kind = "STRING"
                options.update(default=default or "", multiline=schema["type"] == "array")
            inputs[name] = (kind, options)
        return {"required": inputs}

    def execute(self, **kwargs) -> tuple[str, str, int]:
        values = widget_arguments(self.operation, kwargs)
        result = run_operation(self.operation, values)
        report = json.dumps(result, ensure_ascii=False, indent=2)
        if result["exit_code"] not in {0, 2} and not result["output"]:
            raise RuntimeError(result["stderr"] or result["stdout"] or "Operation failed")
        return (result["output"] or "", report, result["exit_code"])


def node_mappings() -> tuple[dict, dict]:
    classes, names = {}, {}
    for operation in operation_catalog():
        key = "RAIWOperation_" + operation["id"]
        classes[key] = type(key, (CatalogOperation,), {"operation": operation, "__module__": __name__})
        names[key] = operation["title"].capitalize() + " (RAIW)"
    return classes, names

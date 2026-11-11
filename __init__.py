"""ComfyUI custom nodes for remove-ai-watermarks."""

if __package__:
    from . import nodes as _nodes
else:
    import nodes as _nodes

NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Keep tensor helpers and add every typed catalog operation.
if __package__:
    from .operation_nodes import node_mappings
else:
    from operation_nodes import node_mappings

operation_classes, operation_names = node_mappings()
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **operation_classes}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **operation_names}

"""ComfyUI custom nodes for remove-ai-watermarks."""

if __package__:
    from . import nodes as _nodes
else:
    import nodes as _nodes

NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

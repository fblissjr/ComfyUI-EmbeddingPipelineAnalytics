# NODE_CLASS_MAPPINGS = {
#     "EmbeddingPipelineCapture": HunyuanPipelineCapture,
#     "EmbeddingAnalyzer": EmbeddingAnalyzer
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "EmbeddingPipelineCapture": "Embedding Pipeline Capture",
#     "EmbeddingAnalyzer": "Embedding Analyzer"
# }

from .nodes import NODE_CLASS_MAPPINGS as NODES_CLASS, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY

NODE_CLASS_MAPPINGS = {**NODES_CLASS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# __all__ = [
#     "NODE_CLASS_MAPPINGS",
#     "NODE_DISPLAY_NAME_MAPPINGS",
#     "BasePipelineCapture",
#     "HunyuanPipelineCapture"
# ]
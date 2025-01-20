from typing import Dict, Any, Optional, List
import torch
import numpy as np
from pathlib import Path
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EmbeddingAnalyzer:
    """Core analysis functionality for embeddings."""
    
    @staticmethod
    def calculate_statistics(tensor: torch.Tensor) -> Dict[str, float]:
        """Calculate basic statistics for a tensor."""
        return {
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "shape": list(tensor.shape)
        }

    @staticmethod
    def reduce_dimensions(tensor: torch.Tensor, 
                         method: str = "umap",
                         n_components: int = 2) -> np.ndarray:
        """Reduce dimensionality of embeddings."""
        # Flatten if needed
        if tensor.ndim > 2:
            tensor_2d = tensor.view(tensor.size(0), -1)
        else:
            tensor_2d = tensor

        tensor_np = tensor_2d.cpu().numpy()

        if method == "umap":
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components)
        elif method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        return reducer.fit_transform(tensor_np)

    @staticmethod
    def compare_embeddings(emb1: torch.Tensor, 
                          emb2: torch.Tensor,
                          method: str = "cosine") -> float:
        """Compare two sets of embeddings."""
        if emb1.ndim > 2:
            emb1 = emb1.view(emb1.size(0), -1)
        if emb2.ndim > 2:
            emb2 = emb2.view(emb2.size(0), -1)

        if method == "cosine":
            return torch.nn.functional.cosine_similarity(
                emb1.mean(dim=0, keepdim=True),
                emb2.mean(dim=0, keepdim=True)
            ).item()
        else:
            raise ValueError(f"Unknown comparison method: {method}")

class PipelineAnalyzer:
    """Analyzes full pipeline runs."""
    
    @staticmethod
    def analyze_run(run_data: Dict[str, Any],
                   analysis_types: List[str] = ["statistics", "umap"]) -> Dict[str, Any]:
        """Analyze a complete pipeline run."""
        analyzer = EmbeddingAnalyzer()
        results = {
            "run_id": run_data.get("run_id"),
            "metadata": run_data.get("metadata", {}),
            "analyses": {}
        }

        # Analyze each stage's embeddings
        for stage, embeddings in run_data.get("embeddings", {}).items():
            stage_results = {}
            
            for name, tensor in embeddings.items():
                tensor_results = {}
                
                if "statistics" in analysis_types:
                    tensor_results["statistics"] = analyzer.calculate_statistics(tensor)
                
                if "umap" in analysis_types:
                    tensor_results["umap"] = analyzer.reduce_dimensions(tensor, "umap").tolist()
                
                if "pca" in analysis_types:
                    tensor_results["pca"] = analyzer.reduce_dimensions(tensor, "pca").tolist()
                
                stage_results[name] = tensor_results
            
            results["analyses"][stage] = stage_results

        return results
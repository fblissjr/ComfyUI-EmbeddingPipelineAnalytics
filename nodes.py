import json
import torch
from typing import Dict, Any, Optional

# analysis imports
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
    
from .core.pipeline_types.hunyuanvideo import HunyuanPipelineCapture

class EmbeddingPipelineCapture:
    """Base node for capturing pipeline data at any point."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("*",),  # Accept any input type
                "run_id": ("STRING", {"default": ""}),  # Optional - will generate if empty
                "stage_name": ("STRING", {}),
                "metadata": ("STRING", {"multiline": True, "default": "{}"}),
                "config_path": ("STRING", {"default": "config.json"}) # Add config path here
            }
        }
    
    RETURN_TYPES = ("*", "STRING")  # Pass through data + run_id
    RETURN_NAMES = ("data", "run_id")
    FUNCTION = "capture"
    CATEGORY = "EmbeddingAnalytics"
    
    def __init__(self):
        # self.data_store = HunyuanPipelineCapture()  # We can make this configurable later
        # Initialize with config path
        self.config_path = "config.json"
        self.data_store = HunyuanPipelineCapture(config_path=self.config_path)
        
    def capture(self, data: Any, run_id: str, stage_name: str, metadata: str, config_path: str) -> tuple:
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {"raw": metadata}
            
        metadata_dict["stage"] = stage_name
            
        # Generate run_id if not provided
        if not run_id:
            run_id = self.data_store.data_store.generate_run_id(metadata_dict)
            
        # Handle different types of data
        if isinstance(data, dict) and "samples" in data:  # LATENT
            self.data_store.data_store.save_outputs(run_id, {"latents": data["samples"]})
        elif isinstance(data, dict) and "prompt_embeds" in data:  # HYVIDEMBEDS
            self.data_store.data_store.save_embeddings(run_id, data, stage_name)
        else:
            # Save as generic output
            self.data_store.data_store.save_outputs(run_id, {stage_name: data})
            
        # Save/update metadata
        self.data_store.data_store.save_metadata(run_id, metadata_dict)
        
        return (data, run_id)

class EmbeddingAnalyzer:
    """Node for analyzing captured embeddings."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "run_id": ("STRING", {}),
                "analysis_type": (["umap", "pca", "tsne", "statistics"], {}),
                "config_path": ("STRING", {"default": "config.json"})
            },
            "optional": {
                "compare_run_id": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": "embedding_analysis"})
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("analysis_results", "visualization")
    FUNCTION = "analyze"
    CATEGORY = "EmbeddingAnalytics"
    
    def __init__(self):
        # self.data_store = HunyuanPipelineCapture()
        # Initialize with config path
        self.config_path = "config.json"
        self.data_store = HunyuanPipelineCapture(config_path=self.config_path)

    
    def analyze(self, run_id: str, analysis_type: str, config_path: str,
                compare_run_id: str = "", output_path: str = "embedding_analysis") -> tuple:
        # Load run data
        run_data = self.data_store.data_store.load_run(run_id)
        
        # Initialize analysis results
        results = {
            "run_id": run_id,
            "analysis_type": analysis_type,
            "stages": {}
        }
        
        # Get comparison data if specified
        compare_data = None
        if compare_run_id:
            compare_data = self.data_store.data_store.load_run(compare_run_id)
            results["compare_run_id"] = compare_run_id
            
        # Analyze embeddings for each stage
        for stage, embeddings in run_data["embeddings"].items():
            stage_results = analyze_embeddings(
                embeddings, 
                analysis_type,
                compare_embeddings=compare_data["embeddings"].get(stage) if compare_data else None
            )
            results["stages"][stage] = stage_results
            
        # Generate visualization
        visualization = generate_visualization(
            results, 
            output_path=output_path
        )
        
        return (json.dumps(results, indent=2), visualization)

def analyze_embeddings(embeddings: Dict[str, torch.Tensor], 
                      analysis_type: str,
                      compare_embeddings: Optional[Dict[str, torch.Tensor]] = None) -> Dict:
    """Analyze embeddings based on specified type."""
    results = {}
    
    for name, tensor in embeddings.items():
        # Basic statistics
        stats = {
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "shape": list(tensor.shape)
        }
        
        # Dimensionality reduction if requested
        if analysis_type in ["umap", "pca", "tsne"]:
            # Prepare 2D tensor
            if tensor.ndim > 2:
                tensor_2d = tensor.view(tensor.size(0), -1)
            else:
                tensor_2d = tensor
                
            if analysis_type == "umap":
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
                reduced = reducer.fit_transform(tensor_2d.cpu().numpy())
            elif analysis_type == "pca":
                reducer = PCA(n_components=2)
                reduced = reducer.fit_transform(tensor_2d.cpu().numpy())
            else:  # tsne
                reducer = TSNE(n_components=2)
                reduced = reducer.fit_transform(tensor_2d.cpu().numpy())
                
            stats["reduced_coords"] = reduced.tolist()
            
            # Compare if provided
            if compare_embeddings and name in compare_embeddings:
                comp_tensor = compare_embeddings[name]
                if comp_tensor.ndim > 2:
                    comp_tensor = comp_tensor.view(comp_tensor.size(0), -1)
                    
                similarity = torch.nn.functional.cosine_similarity(
                    tensor_2d.mean(dim=0, keepdim=True),
                    comp_tensor.mean(dim=0, keepdim=True)
                ).item()
                stats["comparison_similarity"] = similarity
                
        results[name] = stats
        
    return results

def generate_visualization(results: Dict, output_path: str) -> np.ndarray:
    """Generate visualization of analysis results."""
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots for each stage
    num_stages = len(results["stages"])
    fig, axes = plt.subplots(num_stages, 1, figsize=(12, 6*num_stages))
    if num_stages == 1:
        axes = [axes]
        
    for ax, (stage_name, stage_data) in zip(axes, results["stages"].items()):
        # Plot embeddings if reduced coordinates available
        for embed_name, embed_data in stage_data.items():
            if "reduced_coords" in embed_data:
                coords = np.array(embed_data["reduced_coords"])
                ax.scatter(coords[:, 0], coords[:, 1], label=embed_name, alpha=0.6)
                
        ax.set_title(f"{stage_name} - {results['analysis_type'].upper()}")
        ax.legend()
        
    plt.tight_layout()
    
    # Save plot
    fig.savefig(f"{output_path}/analysis_{results['run_id']}.png")
    
    # Convert to image for ComfyUI
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    plt.close(fig)
    
    return image

# Register nodes
NODE_CLASS_MAPPINGS = {
    "EmbeddingPipelineCapture": EmbeddingPipelineCapture,
    "EmbeddingAnalyzer": EmbeddingAnalyzer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan": "Embedding Pipeline Capture",
    "EmbeddingAnalyzer": "Embedding Analyzer"
}
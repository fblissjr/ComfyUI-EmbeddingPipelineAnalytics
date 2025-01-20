# core/pipeline_types/hunyuanvideo.py
from ..data_store import PipelineDataCapture
from typing import Dict, Any, Optional
import torch

class HunyuanPipelineCapture:
    """Specialized capture for HunyuanVideo pipeline."""
    
    def __init__(self):
        self.data_store = PipelineDataCapture("hunyuan_analytics")
        
    def capture_text_encode(self,
                          hyvid_embeds: Dict[str, torch.Tensor],
                          prompt: str,
                          template_used: Optional[str] = None,
                          cfg_settings: Optional[Dict] = None) -> str:
        """Capture embeddings and metadata from text encoding stage."""
        
        # Generate run ID from inputs
        metadata = {
            "stage": "text_encode",
            "prompt": prompt,
            "template": template_used,
            "cfg_settings": cfg_settings
        }
        run_id = self.data_store.generate_run_id(metadata)
        
        # Save embeddings
        embedding_paths = self.data_store.save_embeddings(
            run_id, hyvid_embeds, "text_encode"
        )
        metadata["embedding_paths"] = embedding_paths
        
        # Save metadata
        self.data_store.save_metadata(run_id, metadata)
        
        return run_id

    def capture_sampling(self,
                        run_id: str,
                        samples: torch.Tensor,
                        sampling_params: Dict[str, Any]) -> str:
        """Capture sampling parameters and outputs."""
        
        # Load existing metadata
        run_data = self.data_store.load_run(run_id)
        metadata = run_data["metadata"]
        
        # Add sampling metadata
        metadata["sampling_params"] = sampling_params
        
        # Save samples
        output_paths = self.data_store.save_outputs(
            run_id, {"samples": samples}
        )
        metadata["output_paths"] = output_paths
        
        # Update metadata
        self.data_store.save_metadata(run_id, metadata)
        
        return run_id
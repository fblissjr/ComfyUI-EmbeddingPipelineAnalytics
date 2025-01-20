# core/data_store.py
from typing import Dict, Any, Optional
from pathlib import Path
import json
import torch
import numpy as np
from datetime import datetime
import hashlib

class PipelineDataCapture:
    """Captures and stores pipeline execution data."""
    
    def __init__(self, base_path: str = "pipeline_analytics"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.embeddings_path = self.base_path / "embeddings"
        self.metadata_path = self.base_path / "metadata"
        self.outputs_path = self.base_path / "outputs"
        self.analytics_path = self.base_path / "analytics"
        
        for path in [self.embeddings_path, self.metadata_path, 
                    self.outputs_path, self.analytics_path]:
            path.mkdir(exist_ok=True)

    def generate_run_id(self, metadata: Dict) -> str:
        """Generate unique run ID based on timestamp and input hash."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_hash = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{timestamp}_{input_hash}"

    def save_embeddings(self, 
                       run_id: str,
                       embeddings: Dict[str, torch.Tensor],
                       stage: str) -> Dict[str, str]:
        """Save embeddings with metadata about stage in pipeline."""
        paths = {}
        for name, tensor in embeddings.items():
            if tensor is None:
                continue
                
            filename = f"{run_id}_{stage}_{name}.pt"
            path = self.embeddings_path / filename
            
            # Save tensor
            torch.save({
                "tensor": tensor,
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
                "stage": stage,
                "name": name
            }, path)
            
            paths[name] = str(path)
            
        return paths

    def save_metadata(self,
                     run_id: str,
                     metadata: Dict[str, Any]) -> str:
        """Save run metadata including parameters, prompt, etc."""
        filename = f"{run_id}_metadata.json"
        path = self.metadata_path / filename
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return str(path)

    def save_outputs(self,
                    run_id: str,
                    outputs: Dict[str, Any]) -> Dict[str, str]:
        """Save pipeline outputs (images, videos, etc)."""
        paths = {}
        for name, output in outputs.items():
            if isinstance(output, torch.Tensor):
                filename = f"{run_id}_{name}.pt"
                path = self.outputs_path / filename
                torch.save(output, path)
            elif isinstance(output, np.ndarray):
                filename = f"{run_id}_{name}.npy"
                path = self.outputs_path / filename
                np.save(path, output)
            else:
                filename = f"{run_id}_{name}.json"
                path = self.outputs_path / filename
                with open(path, 'w') as f:
                    json.dump(output, f)
                    
            paths[name] = str(path)
            
        return paths

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load all data for a specific run."""
        run_data = {
            "metadata": None,
            "embeddings": {},
            "outputs": {},
            "analytics": {}
        }
        
        # Load metadata
        metadata_path = self.metadata_path / f"{run_id}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                run_data["metadata"] = json.load(f)
        
        # Load embeddings
        for path in self.embeddings_path.glob(f"{run_id}_*.pt"):
            data = torch.load(path)
            stage = data["stage"]
            if stage not in run_data["embeddings"]:
                run_data["embeddings"][stage] = {}
            run_data["embeddings"][stage][data["name"]] = data["tensor"]
            
        # Load outputs
        for path in self.outputs_path.glob(f"{run_id}_*"):
            name = path.stem.replace(f"{run_id}_", "")
            if path.suffix == ".pt":
                run_data["outputs"][name] = torch.load(path)
            elif path.suffix == ".npy":
                run_data["outputs"][name] = np.load(path)
            else:
                with open(path) as f:
                    run_data["outputs"][name] = json.load(f)
                    
        return run_data
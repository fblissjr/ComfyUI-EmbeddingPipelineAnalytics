from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..data_store import PipelineDataCapture

class BasePipelineCapture(ABC):
    """Base class for pipeline-specific capture implementations."""
    
    def __init__(self, analytics_dir: str = "pipeline_analytics"):
        self.data_store = PipelineDataCapture(analytics_dir)
    
    @abstractmethod
    def capture_stage(self, 
                     stage_name: str,
                     data: Any,
                     metadata: Optional[Dict] = None) -> str:
        """Capture data from a pipeline stage."""
        pass
    
    @abstractmethod
    def get_stage_types(self) -> Dict[str, type]:
        """Return mapping of stage names to expected data types."""
        pass
    
    @abstractmethod
    def validate_stage_data(self, stage_name: str, data: Any) -> bool:
        """Validate data for a specific stage."""
        pass
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentResult:
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class BaseAIAgent(ABC):
    """
    Base class for all AI agents in the automated job application system.
    Provides common functionality and interface for all specialized agents.
    """
    
    def __init__(self, agent_name: str, model_config: Dict[str, Any] = None):
        self.agent_name = agent_name
        self.model_config = model_config or {}
        self.logger = logging.getLogger(f"ai_agent.{agent_name}")
        self.status = AgentStatus.IDLE
        self.metrics = {
            "requests_processed": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_processing_time": 0.0
        }
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Main processing method for the agent.
        Must be implemented by all subclasses.
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data before processing.
        Must be implemented by all subclasses.
        """
        pass
    
    async def process_batch(self, input_data: List[Any], **kwargs) -> List[AgentResult]:
        """
        Process multiple inputs in batch.
        Can be overridden for optimized batch processing.
        """
        self.status = AgentStatus.PROCESSING
        results = []
        
        try:
            tasks = [self.process(item, **kwargs) for item in input_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    results[i] = AgentResult(
                        success=False,
                        data=None,
                        error=str(result)
                    )
                    
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            self.status = AgentStatus.ERROR
        finally:
            self.status = AgentStatus.COMPLETED
            
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "metrics": self.metrics.copy()
        }
    
    def update_metrics(self, success: bool, processing_time: float):
        """Update agent performance metrics."""
        self.metrics["requests_processed"] += 1
        
        if success:
            self.metrics["successful_operations"] += 1
        else:
            self.metrics["failed_operations"] += 1
        
        # Update average processing time
        current_avg = self.metrics["average_processing_time"]
        total_ops = self.metrics["requests_processed"]
        self.metrics["average_processing_time"] = (
            (current_avg * (total_ops - 1) + processing_time) / total_ops
        )
    
    async def health_check(self) -> bool:
        """Perform health check on the agent."""
        try:
            # Test with simple validation
            test_result = await self.process({"test": True})
            return test_result.success
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources used by the agent."""
        self.status = AgentStatus.IDLE
        self.logger.info(f"Agent {self.agent_name} cleaned up")

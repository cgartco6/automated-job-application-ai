"""
Deep Learning Agents for Advanced AI Capabilities

This module contains deep learning models and agents that provide
advanced AI capabilities for the job application system.
"""

__version__ = "2.0.0"
__author__ = "Automated Job Application AI Team"

from .neural_networks.resume_analyzer import ResumeAnalyzer
from .neural_networks.job_matcher import JobMatcher
from .reinforcement_learning.agent_optimizer import AgentOptimizer
from .nlp_models.semantic_matcher import SemanticMatcher

__all__ = [
    "ResumeAnalyzer",
    "JobMatcher", 
    "AgentOptimizer",
    "SemanticMatcher",
]

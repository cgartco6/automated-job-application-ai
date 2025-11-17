"""
AI Agents for Automated Job Application System

This module contains various AI agents that handle different aspects
of the job application process using advanced AI technologies.
"""

__version__ = "2.0.0"
__author__ = "Automated Job Application AI Team"

from .base_agent import BaseAIAgent
from .job_search_agent import JobSearchAgent
from .application_agent import ApplicationAgent
from .resume_agent import ResumeAgent
from .interview_agent import InterviewAgent

__all__ = [
    "BaseAIAgent",
    "JobSearchAgent", 
    "ApplicationAgent",
    "ResumeAgent",
    "InterviewAgent",
]

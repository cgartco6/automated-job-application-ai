"""
Strategic Intelligence for Market Analysis and Decision Making

This module provides strategic analysis capabilities for the job market,
including trend analysis, competitor intelligence, and market predictions.
"""

__version__ = "2.0.0"
__author__ = "Automated Job Application AI Team"

from .market_analysis import MarketIntelligence
from .competitor_analysis import CompetitorAnalyzer
from .trend_prediction import TrendAnalyzer

__all__ = [
    "MarketIntelligence",
    "CompetitorAnalyzer",
    "TrendAnalyzer"
]

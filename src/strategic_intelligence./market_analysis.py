import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

@dataclass
class MarketTrend:
    skill: str
    demand: float
    growth_rate: float
    salary_trend: str
    market_volume: int

class MarketIntelligence:
    """
    Strategic market intelligence for job market analysis.
    Provides insights into skill demand, salary trends, and market dynamics.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.market_data = pd.DataFrame()
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        self.logger = logging.getLogger(__name__)
    
    async def analyze_market_trends(self, location: str = "global", 
                                  industry: str = "technology") -> Dict[str, Any]:
        """
        Analyze comprehensive market trends.
        
        Args:
            location: Geographic location for analysis
            industry: Industry sector for analysis
            
        Returns:
            Comprehensive market analysis
        """
        cache_key = f"market_trends_{location}_{industry}"
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            # Gather data from multiple sources
            trends_data = await asyncio.gather(
                self._get_skill_demand(location, industry),
                self._get_salary_trends(location, industry),
                self._get_industry_growth(industry),
                self._get_geographic_hotspots(),
                return_exceptions=True
            )
            
            # Process results
            skill_demand, salary_trends, industry_growth, geographic_hotspots = trends_data
            
            analysis = {
                'skill_demand': skill_demand if not isinstance(skill_demand, Exception) else {},
                'salary_trends': salary_trends if not isinstance(salary_trends, Exception) else {},
                'industry_growth': industry_growth if not isinstance(industry_growth, Exception) else {},
                'geographic_hotspots': geographic_hotspots if not isinstance(geographic_hotspots, Exception) else [],
                'strategic_recommendations': self._generate_strategic_recommendations(
                    skill_demand, salary_trends, industry_growth, geographic_hotspots
                ),
                'market_health_score': self._calculate_market_health(
                    skill_demand, salary_trends, industry_growth
                ),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self._cache_result(cache_key, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            return {
                'error': str(e),
                'skill_demand': {},
                'salary_trends': {},
                'industry_growth': {},
                'geographic_hotspots': [],
                'strategic_recommendations': []
            }
    
    async def get_salary_data(self, role: str, location: str, experience: int) -> Dict[str, Any]:
        """
        Get salary data for specific role and experience.
        
        Args:
            role: Job role
            location: Geographic location
            experience: Years of experience
            
        Returns:
            Salary data and trends
        """
        cache_key = f"salary_{role}_{location}_{experience}"
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            # Gather salary data from multiple sources
            salary_sources = await asyncio.gather(
                self._get_salary_from_glassdoor(role, location, experience),
                self._get_salary_from_payscale(role, location, experience),
                self._get_salary_from_linkedin(role, location, experience),
                return_exceptions=True
            )
            
            # Aggregate salary data
            aggregated_salary = self._aggregate_salary_data(salary_sources, role, experience)
            
            self._cache_result(cache_key, aggregated_salary)
            return aggregated_salary
            
        except Exception as e:
            self.logger.error(f"Salary data retrieval failed: {e}")
            return {
                'role': role,
                'location': location,
                'experience': experience,
                'median_salary': 0,
                'salary_range': (0, 0),
                'trend': 'unknown',
                'sources': []
            }
    
    async def analyze_skill_marketability(self, skills: List[str]) -> Dict[str, Any]:
        """
        Analyze marketability of specific skills.
        
        Args:
            skills: List of skills to analyze
            
        Returns:
            Skill marketability analysis
        """
        analysis = {}
        
        for skill in skills:
            try:
                skill_analysis = await self._analyze_single_skill(skill)
                analysis[skill] = skill_analysis
            except Exception as e:
                self.logger.warning(f"Skill analysis failed for {skill}: {e}")
                analysis[skill] = {'error': str(e)}
        
        # Generate overall recommendations
        analysis['overall_recommendations'] = self._generate_skill_recommendations(analysis)
        
        return analysis
    
    async def _get_skill_demand(self, location: str, industry: str) -> Dict[str, float]:
        """Get current skill demand from various sources."""
        # Implementation for skill demand analysis
        # This would integrate with job posting APIs and market data
        
        # Mock data for demonstration
        skill_demand = {
            'machine_learning': 0.85,
            'cloud_computing': 0.78,
            'cybersecurity': 0.72,
            'data_engineering': 0.68,
            'devops': 0.75,
            'python': 0.90,
            'javascript': 0.82,
            'react': 0.79,
            'aws': 0.88,
            'docker': 0.76,
            'kubernetes': 0.71,
            'tensorflow': 0.65,
            'pytorch': 0.63,
            'sql': 0.80,
            'nosql': 0.58
        }
        
        return skill_demand
    
    async def _get_salary_trends(self, location: str, industry: str) -> Dict[str, Any]:
        """Get salary trends across different roles."""
        # Implementation for salary trend analysis
        
        salary_trends = {
            'ai_ml_engineer': {
                'current_median': 150000,
                'trend': 'increasing',
                'growth_rate': 0.12,
                'percentile_25': 120000,
                'percentile_75': 180000
            },
            'data_scientist': {
                'current_median': 140000,
                'trend': 'stable',
                'growth_rate': 0.08,
                'percentile_25': 110000,
                'percentile_75': 170000
            },
            'cloud_architect': {
                'current_median': 160000,
                'trend': 'increasing',
                'growth_rate': 0.15,
                'percentile_25': 130000,
                'percentile_75': 190000
            },
            'software_engineer': {
                'current_median': 130000,
                'trend': 'stable',
                'growth_rate': 0.06,
                'percentile_25': 100000,
                'percentile_75': 160000
            }
        }
        
        return salary_trends
    
    async def _get_industry_growth(self, industry: str) -> Dict[str, float]:
        """Get industry growth rates and projections."""
        # Implementation for industry growth analysis
        
        industry_growth = {
            'technology': 0.15,
            'healthcare': 0.08,
            'finance': 0.06,
            'renewable_energy': 0.12,
            'ecommerce': 0.10,
            'cybersecurity': 0.18,
            'artificial_intelligence': 0.25
        }
        
        return {industry: industry_growth.get(industry, 0.05)}
    
    async def _get_geographic_hotspots(self) -> List[str]:
        """Identify geographic hotspots for tech jobs."""
        # Implementation for geographic analysis
        
        return ['San Francisco', 'New York', 'Austin', 'Seattle', 'Boston', 'Denver', 'Atlanta']
    
    async def _get_salary_from_glassdoor(self, role: str, location: str, experience: int) -> Dict[str, Any]:
        """Get salary data from Glassdoor API."""
        # Implementation for Glassdoor API integration
        return {
            'source': 'glassdoor',
            'median_salary': 140000 + (experience * 5000),
            'salary_range': (120000, 160000),
            'sample_size': 1000
        }
    
    async def _get_salary_from_payscale(self, role: str, location: str, experience: int) -> Dict[str, Any]:
        """Get salary data from Payscale API."""
        # Implementation for Payscale API integration
        return {
            'source': 'payscale',
            'median_salary': 135000 + (experience * 4500),
            'salary_range': (115000, 155000),
            'sample_size': 800
        }
    
    async def _get_salary_from_linkedin(self, role: str, location: str, experience: int) -> Dict[str, Any]:
        """Get salary data from LinkedIn."""
        # Implementation for LinkedIn salary insights
        return {
            'source': 'linkedin',
            'median_salary': 145000 + (experience * 5500),
            'salary_range': (125000, 165000),
            'sample_size': 1200
        }
    
    def _aggregate_salary_data(self, salary_sources: List[Dict[str, Any]], 
                             role: str, experience: int) -> Dict[str, Any]:
        """Aggregate salary data from multiple sources."""
        valid_sources = [s for s in salary_sources if not isinstance(s, Exception) and 'median_salary' in s]
        
        if not valid_sources:
            return {
                'role': role,
                'experience': experience,
                'median_salary': 0,
                'salary_range': (0, 0),
                'trend': 'unknown',
                'sources': []
            }
        
        # Calculate weighted average based on sample sizes
        total_samples = sum(s.get('sample_size', 1) for s in valid_sources)
        weighted_salary = sum(s['median_salary'] * s.get('sample_size', 1) for s in valid_sources) / total_samples
        
        # Aggregate salary range
        min_salaries = [s['salary_range'][0] for s in valid_sources]
        max_salaries = [s['salary_range'][1] for s in valid_sources]
        
        return {
            'role': role,
            'experience': experience,
            'median_salary': weighted_salary,
            'salary_range': (min(min_salaries), max(max_salaries)),
            'trend': 'increasing',  # Would be calculated from historical data
            'sources': [s['source'] for s in valid_sources],
            'confidence': min(1.0, total_samples / 2000)  # Confidence based on sample size
        }
    
    async def _analyze_single_skill(self, skill: str) -> Dict[str, Any]:
        """Analyze marketability of a single skill."""
        # Get skill demand data
        skill_demand = await self._get_skill_demand("global", "technology")
        demand_score = skill_demand.get(skill, 0.3)
        
        # Get related salary impact
        salary_impact = await self._calculate_salary_impact(skill)
        
        # Calculate skill value score
        value_score = (demand_score * 0.6) + (salary_impact * 0.4)
        
        return {
            'demand_score': demand_score,
            'salary_impact': salary_impact,
            'value_score': value_score,
            'market_trend': 'growing' if demand_score > 0.7 else 'stable' if demand_score > 0.4 else 'declining',
            'recommendation': self._get_skill_recommendation(demand_score, value_score)
        }
    
    async def _calculate_salary_impact(self, skill: str) -> float:
        """Calculate the salary impact of having a specific skill."""
        # Implementation for salary impact calculation
        # This would analyze how much each skill contributes to salary
        
        salary_impacts = {
            'machine_learning': 0.15,
            'deep_learning': 0.18,
            'aws': 0.12,
            'python': 0.10,
            'javascript': 0.08,
            'react': 0.09,
            'docker': 0.11,
            'kubernetes': 0.13
        }
        
        return salary_impacts.get(skill, 0.05)
    
    def _generate_strategic_recommendations(self, skill_demand: Dict[str, float],
                                          salary_trends: Dict[str, Any],
                                          industry_growth: Dict[str, float],
                                          geographic_hotspots: List[str]) -> List[str]:
        """Generate strategic recommendations based on market analysis."""
        recommendations = []
        
        # Skill-based recommendations
        high_demand_skills = [skill for skill, demand in skill_demand.items() if demand > 0.7]
        if high_demand_skills:
            recommendations.append(
                f"Focus on developing high-demand skills: {', '.join(high_demand_skills[:3])}"
            )
        
        # Industry-based recommendations
        growing_industries = [industry for industry, growth in industry_growth.items() if growth > 0.1]
        if growing_industries:
            recommendations.append(
                f"Target growing industries: {', '.join(growing_industries[:2])}"
            )
        
        # Location-based recommendations
        if geographic_hotspots:
            recommendations.append(
                f"Consider opportunities in these tech hubs: {', '.join(geographic_hotspots[:3])}"
            )
        
        # Salary-based recommendations
        high_growth_roles = [
            role for role, data in salary_trends.items() 
            if data.get('growth_rate', 0) > 0.1
        ]
        if high_growth_roles:
            recommendations.append(
                f"High salary growth potential in: {', '.join(high_growth_roles[:2])}"
            )
        
        return recommendations
    
    def _generate_skill_recommendations(self, skill_analysis: Dict[str, Any]) -> List[str]:
        """Generate skill development recommendations."""
        recommendations = []
        
        high_value_skills = [
            skill for skill, analysis in skill_analysis.items() 
            if skill != 'overall_recommendations' and analysis.get('value_score', 0) > 0.7
        ]
        
        if high_value_skills:
            recommendations.append(
                f"Your high-value skills: {', '.join(high_value_skills)}"
            )
        
        improvement_opportunities = [
            skill for skill, analysis in skill_analysis.items()
            if skill != 'overall_recommendations' and analysis.get('demand_score', 0) > 0.6 and analysis.get('value_score', 0) < 0.5
        ]
        
        if improvement_opportunities:
            recommendations.append(
                f"Skills to improve for higher value: {', '.join(improvement_opportunities)}"
            )
        
        return recommendations
    
    def _get_skill_recommendation(self, demand_score: float, value_score: float) -> str:
        """Get recommendation for a specific skill."""
        if demand_score > 0.8 and value_score > 0.7:
            return "High priority - critical skill with excellent market value"
        elif demand_score > 0.6 and value_score > 0.5:
            return "Medium priority - valuable skill with good demand"
        elif demand_score > 0.4:
            return "Low priority - consider if aligns with career goals"
        else:
            return "Not recommended - low market demand"
    
    def _calculate_market_health(self, skill_demand: Dict[str, float],
                               salary_trends: Dict[str, Any],
                               industry_growth: Dict[str, float]) -> float:
        """Calculate overall market health score."""
        # Average skill demand
        avg_demand = np.mean(list(skill_demand.values())) if skill_demand else 0
        
        # Average salary growth
        salary_growth_rates = [data.get('growth_rate', 0) for data in salary_trends.values()]
        avg_salary_growth = np.mean(salary_growth_rates) if salary_growth_rates else 0
        
        # Industry growth
        avg_industry_growth = np.mean(list(industry_growth.values())) if industry_growth else 0
        
        # Combined health score (weighted average)
        health_score = (avg_demand * 0.4) + (avg_salary_growth * 0.4) + (avg_industry_growth * 0.2)
        
        return min(1.0, health_score)
    
    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get result from cache if valid."""
        if key in self.cache:
            cached_time, result = self.cache[key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return result
            else:
                del self.cache[key]
        return None
    
    def _cache_result(self, key: str, result: Any):
        """Cache a result with timestamp."""
        self.cache[key] = (datetime.now(), result)

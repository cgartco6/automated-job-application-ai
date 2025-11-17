import asyncio
from typing import Dict, List, Any, Optional
import aiohttp
import logging
from dataclasses import dataclass

@dataclass
class CompetitorProfile:
    company: str
    open_positions: int
    hiring_trend: str
    key_technologies: List[str]
    salary_benchmarks: Dict[str, float]

class CompetitorAnalyzer:
    """
    Analyzes competitor companies and their hiring strategies.
    Provides insights for competitive job applications.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.session = None
        self.logger = logging.getLogger(__name__)
    
    async def analyze_competitor_jobs(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze competitor job postings for strategic insights.
        
        Args:
            user_profile: User's skills and preferences
            
        Returns:
            Competitor analysis results
        """
        try:
            target_companies = user_profile.get('preferences', {}).get('target_companies', [])
            if not target_companies:
                # Get companies from user's industry
                industry = user_profile.get('experience', {}).get('industry', 'technology')
                target_companies = await self._get_companies_by_industry(industry)
            
            # Analyze each competitor
            competitor_analyses = await asyncio.gather(*[
                self._analyze_single_competitor(company, user_profile)
                for company in target_companies[:10]  # Limit to top 10
            ], return_exceptions=True)
            
            # Aggregate insights
            aggregated_insights = self._aggregate_competitor_insights(competitor_analyses)
            
            return {
                'competitor_analyses': competitor_analyses,
                'aggregated_insights': aggregated_insights,
                'strategic_recommendations': self._generate_competitive_strategies(aggregated_insights, user_profile)
            }
            
        except Exception as e:
            self.logger.error(f"Competitor analysis failed: {e}")
            return {
                'error': str(e),
                'competitor_analyses': [],
                'aggregated_insights': {},
                'strategic_recommendations': []
            }
    
    async def get_competitive_landscape(self, role: str, location: str) -> Dict[str, Any]:
        """
        Get competitive landscape for a specific role and location.
        
        Args:
            role: Job role
            location: Geographic location
            
        Returns:
            Competitive landscape analysis
        """
        try:
            # Get companies hiring for this role
            hiring_companies = await self._get_companies_hiring_for_role(role, location)
            
            # Analyze each company's offerings
            company_analyses = await asyncio.gather(*[
                self._analyze_company_offerings(company, role, location)
                for company in hiring_companies[:15]  # Limit to top 15
            ], return_exceptions=True)
            
            return {
                'role': role,
                'location': location,
                'total_companies': len(hiring_companies),
                'company_analyses': company_analyses,
                'competitive_intensity': self._calculate_competitive_intensity(company_analyses),
                'market_opportunities': self._identify_market_opportunities(company_analyses)
            }
            
        except Exception as e:
            self.logger.error(f"Competitive landscape analysis failed: {e}")
            return {
                'role': role,
                'location': location,
                'error': str(e)
            }
    
    async def _analyze_single_competitor(self, company: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single competitor company."""
        try:
            # Get company job postings
            job_postings = await self._get_company_job_postings(company)
            
            # Analyze job requirements
            common_requirements = self._analyze_job_requirements(job_postings)
            
            # Analyze salary data if available
            salary_data = await self._get_company_salary_data(company, user_profile)
            
            # Analyze hiring trends
            hiring_trends = self._analyze_hiring_trends(job_postings)
            
            return {
                'company': company,
                'total_openings': len(job_postings),
                'common_requirements': common_requirements,
                'salary_benchmarks': salary_data,
                'hiring_trends': hiring_trends,
                'key_technologies': self._extract_key_technologies(job_postings),
                'competitive_advantage': self._assess_competitive_advantage(company, user_profile)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze competitor {company}: {e}")
            return {
                'company': company,
                'error': str(e)
            }
    
    async def _get_companies_by_industry(self, industry: str) -> List[str]:
        """Get companies in a specific industry."""
        # Implementation would integrate with company databases
        industry_companies = {
            'technology': ['Google', 'Microsoft', 'Apple', 'Amazon', 'Meta', 'Netflix', 'Twitter', 'Uber'],
            'finance': ['JPMorgan', 'Goldman Sachs', 'Morgan Stanley', 'Bank of America', 'Citigroup'],
            'healthcare': ['Johnson & Johnson', 'Pfizer', 'Merck', 'UnitedHealth', 'CVS Health']
        }
        
        return industry_companies.get(industry, ['Google', 'Microsoft', 'Amazon'])
    
    async def _get_company_job_postings(self, company: str) -> List[Dict[str, Any]]:
        """Get job postings for a company."""
        # Implementation would integrate with job APIs
        # Mock data for demonstration
        return [
            {
                'title': 'Senior Software Engineer',
                'description': 'Looking for experienced software engineer...',
                'requirements': ['Python', 'AWS', 'Docker', '5+ years experience'],
                'location': 'San Francisco'
            }
            # More job postings...
        ]
    
    def _analyze_job_requirements(self, job_postings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze common requirements across job postings."""
        requirement_frequency = {}
        
        for job in job_postings:
            requirements = job.get('requirements', [])
            for req in requirements:
                requirement_frequency[req] = requirement_frequency.get(req, 0) + 1
        
        # Return top requirements
        sorted_requirements = sorted(requirement_frequency.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_requirements[:10])
    
    async def _get_company_salary_data(self, company: str, user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Get salary data for a company."""
        # Implementation would integrate with salary APIs
        role = user_profile.get('experience', {}).get('title', 'Software Engineer')
        
        # Mock salary data
        salary_benchmarks = {
            'entry_level': 120000,
            'mid_level': 150000,
            'senior_level': 180000,
            'leadership': 220000
        }
        
        return salary_benchmarks
    
    def _analyze_hiring_trends(self, job_postings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze hiring trends from job postings."""
        # Simplified implementation
        total_openings = len(job_postings)
        
        return {
            'trend': 'growing' if total_openings > 50 else 'stable',
            'growth_rate': 0.15,  # Would be calculated from historical data
            'hot_roles': ['Machine Learning Engineer', 'Data Scientist', 'Cloud Architect']
        }
    
    def _extract_key_technologies(self, job_postings: List[Dict[str, Any]]) -> List[str]:
        """Extract key technologies from job postings."""
        technologies = set()
        
        for job in job_postings:
            description = job.get('description', '').lower()
            requirements = job.get('requirements', [])
            
            # Check for common technologies
            tech_keywords = ['python', 'java', 'javascript', 'aws', 'docker', 'kubernetes', 
                           'react', 'node.js', 'tensorflow', 'pytorch', 'machine learning']
            
            for tech in tech_keywords:
                if tech in description or any(tech in req.lower() for req in requirements):
                    technologies.add(tech)
        
        return list(technologies)
    
    def _assess_competitive_advantage(self, company: str, user_profile: Dict[str, Any]) -> str:
        """Assess competitive advantage for applying to this company."""
        user_skills = set(user_profile.get('skills', []))
        
        # This would be more sophisticated in practice
        if len(user_skills) > 10:
            return "strong"
        elif len(user_skills) > 5:
            return "moderate"
        else:
            return "weak"
    
    def _aggregate_competitor_insights(self, competitor_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate insights from all competitor analyses."""
        valid_analyses = [a for a in competitor_analyses if not isinstance(a, Exception) and 'error' not in a]
        
        if not valid_analyses:
            return {}
        
        # Aggregate common requirements
        all_requirements = {}
        for analysis in valid_analyses:
            requirements = analysis.get('common_requirements', {})
            for req, count in requirements.items():
                all_requirements[req] = all_requirements.get(req, 0) + count
        
        # Aggregate technologies
        all_technologies = set()
        for analysis in valid_analyses:
            technologies = analysis.get('key_technologies', [])
            all_technologies.update(technologies)
        
        # Calculate average salaries
        salaries = [analysis.get('salary_benchmarks', {}) for analysis in valid_analyses]
        avg_salaries = self._calculate_average_salaries(salaries)
        
        return {
            'total_companies_analyzed': len(valid_analyses),
            'most_common_requirements': dict(sorted(all_requirements.items(), key=lambda x: x[1], reverse=True)[:10]),
            'in_demand_technologies': list(all_technologies),
            'average_salaries': avg_salaries,
            'hiring_intensity': sum(analysis.get('total_openings', 0) for analysis in valid_analyses) / len(valid_analyses)
        }
    
    def _calculate_average_salaries(self, salary_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average salaries from multiple sources."""
        salary_levels = ['entry_level', 'mid_level', 'senior_level', 'leadership']
        avg_salaries = {}
        
        for level in salary_levels:
            level_salaries = [s.get(level, 0) for s in salary_data if s.get(level, 0) > 0]
            if level_salaries:
                avg_salaries[level] = sum(level_salaries) / len(level_salaries)
        
        return avg_salaries
    
    def _generate_competitive_strategies(self, insights: Dict[str, Any], user_profile: Dict[str, Any]) -> List[str]:
        """Generate competitive application strategies."""
        strategies = []
        
        user_skills = set(user_profile.get('skills', []))
        common_requirements = set(insights.get('most_common_requirements', {}).keys())
        
        # Skill gap analysis
        missing_skills = common_requirements - user_skills
        if missing_skills:
            strategies.append(f"Develop these in-demand skills: {', '.join(list(missing_skills)[:3])}")
        
        # Salary negotiation strategy
        avg_salaries = insights.get('average_salaries', {})
        if avg_salaries:
            target_level = self._determine_experience_level(user_profile)
            target_salary = avg_salaries.get(target_level, 0)
            if target_salary > 0:
                strategies.append(f"Target salary range: ${target_salary:,.0f} based on market data")
        
        # Application timing strategy
        hiring_intensity = insights.get('hiring_intensity', 0)
        if hiring_intensity > 20:
            strategies.append("High hiring intensity - apply now for best opportunities")
        else:
            strategies.append("Moderate hiring - focus on quality applications")
        
        return strategies
    
    def _determine_experience_level(self, user_profile: Dict[str, Any]) -> str:
        """Determine experience level based on user profile."""
        experience_years = user_profile.get('experience', {}).get('years', 0)
        
        if experience_years >= 8:
            return 'leadership'
        elif experience_years >= 5:
            return 'senior_level'
        elif experience_years >= 2:
            return 'mid_level'
        else:
            return 'entry_level'
    
    async def _get_companies_hiring_for_role(self, role: str, location: str) -> List[str]:
        """Get companies hiring for a specific role."""
        # Implementation would integrate with job APIs
        return ['Google', 'Microsoft', 'Amazon', 'Meta', 'Apple', 'Netflix']
    
    async def _analyze_company_offerings(self, company: str, role: str, location: str) -> Dict[str, Any]:
        """Analyze a company's offerings for a specific role."""
        # Implementation would gather comprehensive company data
        return {
            'company': company,
            'role': role,
            'estimated_salary': 150000,
            'benefits_rating': 4.5,
            'company_rating': 4.2,
            'interview_difficulty': 'high',
            'hiring_process_length': '4-6 weeks'
        }
    
    def _calculate_competitive_intensity(self, company_analyses: List[Dict[str, Any]]) -> str:
        """Calculate competitive intensity for the role."""
        valid_analyses = [a for a in company_analyses if not isinstance(a, Exception)]
        
        if not valid_analyses:
            return 'unknown'
        
        avg_company_rating = np.mean([a.get('company_rating', 0) for a in valid_analyses])
        
        if avg_company_rating > 4.0:
            return 'high'
        elif avg_company_rating > 3.5:
            return 'medium'
        else:
            return 'low'
    
    def _identify_market_opportunities(self, company_analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify market opportunities from company analyses."""
        opportunities = []
        valid_analyses = [a for a in company_analyses if not isinstance(a, Exception)]
        
        # Find companies with high ratings but reasonable interview difficulty
        for analysis in valid_analyses:
            if (analysis.get('company_rating', 0) > 4.0 and 
                analysis.get('interview_difficulty') != 'very high'):
                opportunities.append(analysis['company'])
        
        return opportunities[:3]  # Return top 3 opportunities

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import json
import logging
from .base_agent import BaseAIAgent, AgentResult

@dataclass
class JobListing:
    id: str
    title: str
    company: str
    location: str
    description: str
    url: str
    salary_range: Optional[str] = None
    requirements: List[str] = None
    posted_date: Optional[str] = None
    application_deadline: Optional[str] = None
    job_type: Optional[str] = None
    experience_level: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "description": self.description,
            "url": self.url,
            "salary_range": self.salary_range,
            "requirements": self.requirements or [],
            "posted_date": self.posted_date,
            "application_deadline": self.application_deadline,
            "job_type": self.job_type,
            "experience_level": self.experience_level
        }

class JobSearchAgent(BaseAIAgent):
    """
    AI agent for intelligent job search and discovery across multiple platforms.
    Uses strategic intelligence to find the most relevant job opportunities.
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__("job_search_agent", model_config)
        self.search_platforms = [
            "linkedin",
            "indeed", 
            "glassdoor",
            "monster",
            "careerbuilder"
        ]
        self.session = None
        self.search_strategies = {
            "keyword_optimization": self._optimize_keywords,
            "semantic_matching": self._semantic_match,
            "trend_analysis": self._analyze_trends,
            "competitor_analysis": self._analyze_competitors
        }
    
    async def process(self, user_profile: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Search for jobs based on user profile and preferences.
        
        Args:
            user_profile: Dictionary containing user skills, experience, preferences
            **kwargs: Additional search parameters
            
        Returns:
            AgentResult containing list of JobListing objects
        """
        import time
        start_time = time.time()
        
        try:
            if not self.validate_input(user_profile):
                return AgentResult(
                    success=False,
                    data=None,
                    error="Invalid user profile format"
                )
            
            self.status = self.AgentStatus.PROCESSING
            self.logger.info(f"Starting job search for user: {user_profile.get('user_id', 'unknown')}")
            
            # Apply search strategies
            optimized_query = await self._apply_search_strategies(user_profile)
            
            # Search across multiple platforms
            search_tasks = []
            for platform in self.search_platforms:
                task = self._search_platform(platform, optimized_query, user_profile)
                search_tasks.append(task)
            
            # Execute all searches concurrently
            platform_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and deduplicate results
            all_jobs = await self._combine_results(platform_results)
            
            # Rank jobs by relevance
            ranked_jobs = await self._rank_jobs(all_jobs, user_profile)
            
            # Filter based on user preferences
            filtered_jobs = self._filter_jobs(ranked_jobs, user_profile)
            
            processing_time = time.time() - start_time
            self.update_metrics(True, processing_time)
            
            return AgentResult(
                success=True,
                data=filtered_jobs,
                metadata={
                    "total_found": len(all_jobs),
                    "total_returned": len(filtered_jobs),
                    "search_strategies_used": list(self.search_strategies.keys()),
                    "processing_time": processing_time
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(False, processing_time)
            self.logger.error(f"Job search failed: {e}")
            
            return AgentResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate user profile for job search."""
        required_fields = ['skills', 'experience', 'preferences']
        if not all(field in input_data for field in required_fields):
            return False
        
        # Validate skills
        skills = input_data.get('skills', [])
        if not isinstance(skills, list) or len(skills) == 0:
            return False
        
        # Validate experience
        experience = input_data.get('experience', {})
        if not isinstance(experience, dict):
            return False
        
        return True
    
    async def _apply_search_strategies(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Apply various search strategies to optimize job search."""
        optimized_query = {
            "keywords": [],
            "filters": {},
            "boosted_terms": [],
            "excluded_terms": []
        }
        
        # Apply each search strategy
        for strategy_name, strategy_func in self.search_strategies.items():
            try:
                strategy_result = await strategy_func(user_profile)
                optimized_query = self._merge_strategy_results(optimized_query, strategy_result)
            except Exception as e:
                self.logger.warning(f"Strategy {strategy_name} failed: {e}")
        
        return optimized_query
    
    async def _optimize_keywords(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize search keywords based on user skills and market trends."""
        skills = user_profile.get('skills', [])
        experience = user_profile.get('experience', {})
        
        # Expand skills with synonyms and related terms
        expanded_keywords = []
        for skill in skills:
            synonyms = await self._get_skill_synonyms(skill)
            expanded_keywords.extend(synonyms)
        
        # Add experience-level specific terms
        experience_level = experience.get('years', 0)
        if experience_level > 5:
            expanded_keywords.extend(["senior", "lead", "principal", "architect"])
        elif experience_level > 2:
            expanded_keywords.extend(["mid-level", "experienced"])
        else:
            expanded_keywords.extend(["junior", "entry-level", "associate"])
        
        return {
            "keywords": list(set(expanded_keywords)),
            "boosted_terms": skills  # Boost original skills
        }
    
    async def _semantic_match(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Use semantic matching to find conceptually similar jobs."""
        # This would integrate with NLP models for semantic understanding
        from ..deep_agents.nlp_models.semantic_matcher import SemanticMatcher
        
        matcher = SemanticMatcher()
        semantic_terms = await matcher.find_similar_roles(user_profile)
        
        return {
            "keywords": semantic_terms,
            "filters": {
                "semantic_expansion": True
            }
        }
    
    async def _analyze_trends(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trends to identify emerging opportunities."""
        from ..strategic_intelligence.trend_prediction import TrendAnalyzer
        
        analyzer = TrendAnalyzer()
        trends = await analyzer.get_emerging_trends(user_profile)
        
        return {
            "keywords": trends.get("emerging_skills", []),
            "boosted_terms": trends.get("hot_technologies", [])
        }
    
    async def _analyze_competitors(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitor job postings for strategic insights."""
        from ..strategic_intelligence.competitor_analysis import CompetitorAnalyzer
        
        analyzer = CompetitorAnalyzer()
        competitor_insights = await analyzer.analyze_competitor_jobs(user_profile)
        
        return {
            "keywords": competitor_insights.get("common_requirements", []),
            "excluded_terms": competitor_insights.get("oversaturated_skills", [])
        }
    
    async def _search_platform(self, platform: str, query: Dict[str, Any], 
                             user_profile: Dict[str, Any]) -> List[JobListing]:
        """Search a specific job platform."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Platform-specific search implementation
            if platform == "linkedin":
                return await self._search_linkedin(query, user_profile)
            elif platform == "indeed":
                return await self._search_indeed(query, user_profile)
            elif platform == "glassdoor":
                return await self._search_glassdoor(query, user_profile)
            else:
                return await self._search_generic(platform, query, user_profile)
                
        except Exception as e:
            self.logger.error(f"Platform {platform} search failed: {e}")
            return []
    
    async def _search_linkedin(self, query: Dict[str, Any], user_profile: Dict[str, Any]) -> List[JobListing]:
        """Search LinkedIn jobs."""
        # Implementation for LinkedIn API integration
        headers = {
            "Authorization": f"Bearer {self.model_config.get('linkedin_api_key')}",
            "Content-Type": "application/json"
        }
        
        search_params = {
            "keywords": " ".join(query["keywords"]),
            "location": user_profile.get('preferences', {}).get('location', ''),
            "experience": self._map_experience_level(user_profile),
            "limit": 50
        }
        
        async with self.session.get(
            "https://api.linkedin.com/v2/jobSearch",
            headers=headers,
            params=search_params
        ) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_linkedin_jobs(data)
            else:
                self.logger.warning(f"LinkedIn API returned {response.status}")
                return []
    
    async def _search_indeed(self, query: Dict[str, Any], user_profile: Dict[str, Any]) -> List[JobListing]:
        """Search Indeed jobs."""
        # Implementation for Indeed API
        return []  # Placeholder
    
    async def _search_glassdoor(self, query: Dict[str, Any], user_profile: Dict[str, Any]) -> List[JobListing]:
        """Search Glassdoor jobs."""
        # Implementation for Glassdoor API
        return []  # Placeholder
    
    async def _search_generic(self, platform: str, query: Dict[str, Any], 
                            user_profile: Dict[str, Any]) -> List[JobListing]:
        """Generic web scraping fallback for job platforms."""
        # Implementation for web scraping
        return []  # Placeholder
    
    async def _combine_results(self, platform_results: List[List[JobListing]]) -> List[JobListing]:
        """Combine and deduplicate results from multiple platforms."""
        all_jobs = []
        seen_ids = set()
        
        for platform_jobs in platform_results:
            if isinstance(platform_jobs, Exception):
                continue
                
            for job in platform_jobs:
                # Create unique ID based on title, company, and location
                job_id = f"{job.title}_{job.company}_{job.location}".lower().replace(" ", "_")
                if job_id not in seen_ids:
                    seen_ids.add(job_id)
                    all_jobs.append(job)
        
        return all_jobs
    
    async def _rank_jobs(self, jobs: List[JobListing], user_profile: Dict[str, Any]) -> List[JobListing]:
        """Rank jobs by relevance to user profile."""
        from ..deep_agents.nlp_models.ranking_model import JobRanker
        
        ranker = JobRanker()
        ranked_jobs = await ranker.rank_jobs(jobs, user_profile)
        
        return ranked_jobs
    
    def _filter_jobs(self, jobs: List[JobListing], user_profile: Dict[str, Any]) -> List[JobListing]:
        """Filter jobs based on user preferences."""
        preferences = user_profile.get('preferences', {})
        filtered_jobs = []
        
        for job in jobs:
            if self._matches_preferences(job, preferences):
                filtered_jobs.append(job)
        
        return filtered_jobs
    
    def _matches_preferences(self, job: JobListing, preferences: Dict[str, Any]) -> bool:
        """Check if job matches user preferences."""
        # Location filter
        preferred_locations = preferences.get('locations', [])
        if preferred_locations and job.location not in preferred_locations:
            return False
        
        # Salary filter
        min_salary = preferences.get('min_salary')
        if min_salary and job.salary_range:
            # Parse salary range and compare
            salary_low = self._parse_salary(job.salary_range)[0]
            if salary_low and salary_low < min_salary:
                return False
        
        # Job type filter
        preferred_types = preferences.get('job_types', [])
        if preferred_types and job.job_type not in preferred_types:
            return False
        
        # Experience level filter
        preferred_levels = preferences.get('experience_levels', [])
        if preferred_levels and job.experience_level not in preferred_levels:
            return False
        
        return True
    
    def _parse_salary(self, salary_range: str) -> tuple:
        """Parse salary range string into numeric values."""
        # Implementation for salary parsing
        return (None, None)
    
    def _map_experience_level(self, user_profile: Dict[str, Any]) -> str:
        """Map user experience to platform-specific levels."""
        experience_years = user_profile.get('experience', {}).get('years', 0)
        
        if experience_years >= 7:
            return "7"
        elif experience_years >= 5:
            return "5"
        elif experience_years >= 3:
            return "3"
        elif experience_years >= 1:
            return "2"
        else:
            return "1"
    
    async def _get_skill_synonyms(self, skill: str) -> List[str]:
        """Get synonyms and related terms for a skill."""
        # This could integrate with external APIs or internal knowledge base
        synonym_map = {
            "python": ["python", "python3", "python programming", "python development"],
            "machine learning": ["ml", "ai", "artificial intelligence", "deep learning"],
            "aws": ["amazon web services", "cloud", "cloud computing"],
            # Add more skill mappings
        }
        
        return synonym_map.get(skill.lower(), [skill])
    
    def _merge_strategy_results(self, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from different search strategies."""
        merged = base.copy()
        
        # Merge keywords
        merged["keywords"].extend(new.get("keywords", []))
        merged["keywords"] = list(set(merged["keywords"]))
        
        # Merge boosted terms
        merged["boosted_terms"].extend(new.get("boosted_terms", []))
        merged["boosted_terms"] = list(set(merged["boosted_terms"]))
        
        # Merge excluded terms
        merged["excluded_terms"].extend(new.get("excluded_terms", []))
        merged["excluded_terms"] = list(set(merged["excluded_terms"]))
        
        # Merge filters
        merged["filters"].update(new.get("filters", {}))
        
        return merged
    
    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        if self.session:
            await self.session.close()
            self.session = None

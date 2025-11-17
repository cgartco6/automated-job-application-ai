import asyncio
from typing import Dict, List, Any, Optional
import json
import logging
from .base_agent import BaseAIAgent, AgentResult

class ResumeAgent(BaseAIAgent):
    """
    AI agent for resume analysis, optimization, and tailoring.
    Uses deep learning models to analyze and improve resumes.
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__("resume_agent", model_config)
        self.optimization_strategies = [
            "keyword_optimization",
            "achievement_quantification", 
            "skill_highlighting",
            "ats_optimization",
            "readability_improvement"
        ]
    
    async def process(self, resume_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Analyze and optimize a resume.
        
        Args:
            resume_data: Dictionary containing resume content and metadata
            **kwargs: Additional optimization parameters
            
        Returns:
            AgentResult with optimized resume and analysis
        """
        import time
        start_time = time.time()
        
        try:
            if not self.validate_input(resume_data):
                return AgentResult(
                    success=False,
                    data=None,
                    error="Invalid resume data format"
                )
            
            self.status = self.AgentStatus.PROCESSING
            
            # Analyze current resume
            analysis = await self._analyze_resume(resume_data)
            
            # Optimize resume based on analysis
            optimized_resume = await self._optimize_resume(resume_data, analysis, kwargs)
            
            # Generate improvement suggestions
            suggestions = await self._generate_suggestions(analysis, optimized_resume)
            
            processing_time = time.time() - start_time
            self.update_metrics(True, processing_time)
            
            return AgentResult(
                success=True,
                data={
                    "original_resume": resume_data,
                    "optimized_resume": optimized_resume,
                    "analysis": analysis,
                    "suggestions": suggestions
                },
                metadata={
                    "optimization_strategies_applied": self.optimization_strategies,
                    "processing_time": processing_time,
                    "improvement_score": analysis.get("improvement_score", 0)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(False, processing_time)
            self.logger.error(f"Resume processing failed: {e}")
            
            return AgentResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate resume data structure."""
        required_fields = ['content', 'format', 'sections']
        if not all(field in input_data for field in required_fields):
            return False
        
        # Validate content structure
        content = input_data['content']
        if not isinstance(content, dict) or 'raw_text' not in content:
            return False
        
        return True
    
    async def tailor_resume(self, base_resume: Dict[str, Any], 
                          job_description: str) -> Dict[str, Any]:
        """
        Tailor a resume to a specific job description.
        
        Args:
            base_resume: The original resume data
            job_description: The target job description
            
        Returns:
            Tailored resume data
        """
        try:
            # Extract key requirements from job description
            job_requirements = await self._extract_requirements(job_description)
            
            # Analyze resume against requirements
            gap_analysis = await self._analyze_requirements_gap(base_resume, job_requirements)
            
            # Tailor resume content
            tailored_resume = await self._tailor_content(base_resume, job_requirements, gap_analysis)
            
            return tailored_resume
            
        except Exception as e:
            self.logger.error(f"Resume tailoring failed: {e}")
            return base_resume  # Return original resume if tailoring fails
    
    async def _analyze_resume(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive resume analysis using multiple models."""
        from ..deep_agents.nlp_models.resume_analyzer import DeepResumeAnalyzer
        
        analyzer = DeepResumeAnalyzer()
        analysis = await analyzer.analyze(resume_data)
        
        # Add strategic analysis
        strategic_analysis = await self._strategic_analysis(resume_data)
        analysis.update(strategic_analysis)
        
        return analysis
    
    async def _optimize_resume(self, resume_data: Dict[str, Any], 
                             analysis: Dict[str, Any], 
                             options: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resume based on analysis results."""
        optimized = resume_data.copy()
        
        # Apply various optimization strategies
        for strategy in self.optimization_strategies:
            try:
                if strategy == "keyword_optimization":
                    optimized = await self._optimize_keywords(optimized, analysis)
                elif strategy == "achievement_quantification":
                    optimized = await self._quantify_achievements(optimized)
                elif strategy == "skill_highlighting":
                    optimized = await self._highlight_skills(optimized, analysis)
                elif strategy == "ats_optimization":
                    optimized = await self._optimize_for_ats(optimized)
                elif strategy == "readability_improvement":
                    optimized = await self._improve_readability(optimized)
            except Exception as e:
                self.logger.warning(f"Optimization strategy {strategy} failed: {e}")
        
        return optimized
    
    async def _generate_suggestions(self, analysis: Dict[str, Any], 
                                  optimized_resume: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        # Skill-related suggestions
        if analysis.get("missing_skills"):
            suggestions.append(
                f"Consider acquiring these in-demand skills: {', '.join(analysis['missing_skills'][:3])}"
            )
        
        # Experience-related suggestions
        if analysis.get("experience_gaps"):
            suggestions.append(
                "Highlight transferable skills to address experience gaps"
            )
        
        # Format-related suggestions
        if analysis.get("readability_score", 0) < 0.7:
            suggestions.append(
                "Improve readability by using bullet points and concise language"
            )
        
        # Achievement-related suggestions
        if analysis.get("achievement_density", 0) < 0.5:
            suggestions.append(
                "Add more quantifiable achievements with metrics and results"
            )
        
        return suggestions
    
    async def _extract_requirements(self, job_description: str) -> Dict[str, Any]:
        """Extract key requirements from job description."""
        from ..deep_agents.nlp_models.requirement_extractor import RequirementExtractor
        
        extractor = RequirementExtractor()
        requirements = await extractor.extract(job_description)
        
        return requirements
    
    async def _analyze_requirements_gap(self, resume: Dict[str, Any], 
                                      requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze gap between resume and job requirements."""
        gap_analysis = {
            "missing_skills": [],
            "matching_skills": [],
            "experience_gaps": [],
            "strengths": [],
            "weaknesses": []
        }
        
        # Analyze skills gap
        required_skills = requirements.get("skills", [])
        resume_skills = resume.get("content", {}).get("skills", [])
        
        for skill in required_skills:
            if skill in resume_skills:
                gap_analysis["matching_skills"].append(skill)
            else:
                gap_analysis["missing_skills"].append(skill)
        
        # Analyze experience requirements
        required_experience = requirements.get("experience", {})
        resume_experience = resume.get("content", {}).get("experience", {})
        
        # Calculate experience gaps
        for role, years in required_experience.items():
            resume_years = resume_experience.get(role, 0)
            if resume_years < years:
                gap_analysis["experience_gaps"].append({
                    "role": role,
                    "required": years,
                    "current": resume_years
                })
        
        return gap_analysis
    
    async def _tailor_content(self, resume: Dict[str, Any], 
                            requirements: Dict[str, Any],
                            gap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Tailor resume content to job requirements."""
        tailored = resume.copy()
        
        # Highlight matching skills
        tailored["content"]["skills"] = self._reorder_skills(
            resume["content"]["skills"],
            requirements["skills"]
        )
        
        # Tailor professional summary
        tailored["content"]["summary"] = await self._tailor_summary(
            resume["content"]["summary"],
            requirements
        )
        
        # Emphasize relevant experience
        tailored["content"]["experience"] = await self._tailor_experience(
            resume["content"]["experience"],
            requirements
        )
        
        return tailored
    
    def _reorder_skills(self, skills: List[str], required_skills: List[str]) -> List[str]:
        """Reorder skills to highlight relevant ones first."""
        relevant_skills = [s for s in skills if s in required_skills]
        other_skills = [s for s in skills if s not in required_skills]
        
        return relevant_skills + other_skills
    
    async def _tailor_summary(self, original_summary: str, 
                            requirements: Dict[str, Any]) -> str:
        """Tailor professional summary to job requirements."""
        from ..synthetic_intelligence.response_generation import SummaryGenerator
        
        generator = SummaryGenerator()
        tailored_summary = await generator.generate_tailored_summary(
            original_summary=original_summary,
            job_requirements=requirements
        )
        
        return tailored_summary
    
    async def _tailor_experience(self, experience: List[Dict[str, Any]], 
                               requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Tailor experience section to highlight relevant achievements."""
        tailored_experience = []
        
        for job in experience:
            tailored_job = job.copy()
            
            # Filter and reorder bullet points based on relevance
            if "achievements" in job:
                tailored_job["achievements"] = await self._tailor_achievements(
                    job["achievements"],
                    requirements
                )
            
            tailored_experience.append(tailored_job)
        
        return tailored_experience
    
    async def _tailor_achievements(self, achievements: List[str], 
                                 requirements: Dict[str, Any]) -> List[str]:
        """Tailor achievements to highlight relevant ones."""
        from ..deep_agents.nlp_models.relevance_scorer import RelevanceScorer
        
        scorer = RelevanceScorer()
        scored_achievements = []
        
        for achievement in achievements:
            score = await scorer.score_relevance(achievement, requirements)
            scored_achievements.append((achievement, score))
        
        # Sort by relevance score
        scored_achievements.sort(key=lambda x: x[1], reverse=True)
        
        return [achievement for achievement, score in scored_achievements]
    
    async def _strategic_analysis(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform strategic analysis of resume content."""
        from ..strategic_intelligence.market_analysis import MarketIntelligence
        
        market_intel = MarketIntelligence()
        market_trends = await market_intel.analyze_market_trends()
        
        analysis = {}
        
        # Analyze skill market value
        skills = resume_data.get("content", {}).get("skills", [])
        skill_values = []
        
        for skill in skills:
            value = market_trends.get("skill_demand", {}).get(skill, 0)
            skill_values.append((skill, value))
        
        analysis["skill_market_value"] = skill_values
        analysis["average_skill_value"] = (
            sum(value for _, value in skill_values) / len(skill_values) if skill_values else 0
        )
        
        return analysis
    
    async def _optimize_keywords(self, resume: Dict[str, Any], 
                               analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resume keywords for ATS systems."""
        from ..deep_agents.nlp_models.keyword_optimizer import KeywordOptimizer
        
        optimizer = KeywordOptimizer()
        optimized = await optimizer.optimize_keywords(resume)
        
        return optimized
    
    async def _quantify_achievements(self, resume: Dict[str, Any]) -> Dict[str, Any]:
        """Quantify achievements with metrics and numbers."""
        from ..synthetic_intelligence.data_generation import AchievementQuantifier
        
        quantifier = AchievementQuantifier()
        quantified = await quantifier.quantify_achievements(resume)
        
        return quantified
    
    async def _highlight_skills(self, resume: Dict[str, Any], 
                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Highlight relevant skills throughout the resume."""
        highlighted = resume.copy()
        
        # Get high-value skills from analysis
        high_value_skills = [
            skill for skill, value in analysis.get("skill_market_value", [])
            if value > 0.7
        ]
        
        # Ensure these skills are prominently featured
        if "content" in highlighted and "skills" in highlighted["content"]:
            # Reorder to put high-value skills first
            current_skills = highlighted["content"]["skills"]
            prioritized_skills = [
                skill for skill in high_value_skills if skill in current_skills
            ]
            other_skills = [
                skill for skill in current_skills if skill not in high_value_skills
            ]
            highlighted["content"]["skills"] = prioritized_skills + other_skills
        
        return highlighted
    
    async def _optimize_for_ats(self, resume: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resume for Applicant Tracking Systems."""
        optimized = resume.copy()
        
        # Ensure standard section headings
        standard_sections = {
            "professional experience": "experience",
            "work history": "experience", 
            "education history": "education",
            "academic background": "education",
            "skills & abilities": "skills",
            "technical skills": "skills"
        }
        
        if "sections" in optimized:
            for old_name, new_name in standard_sections.items():
                if old_name in optimized["sections"]:
                    optimized["sections"][new_name] = optimized["sections"].pop(old_name)
        
        # Remove complex formatting
        if "format" in optimized:
            optimized["format"] = {
                "type": "standard",
                "fonts": ["Arial", "Calibri", "Times New Roman"],
                "font_size": 11,
                "margins": "1 inch"
            }
        
        return optimized
    
    async def _improve_readability(self, resume: Dict[str, Any]) -> Dict[str, Any]:
        """Improve resume readability and scannability."""
        from ..deep_agents.nlp_models.readability_analyzer import ReadabilityAnalyzer
        
        analyzer = ReadabilityAnalyzer()
        improved = await analyzer.improve_readability(resume)
        
        return improved

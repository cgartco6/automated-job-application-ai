import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import logging
from datetime import datetime, timedelta
import base64
import io
from .base_agent import BaseAIAgent, AgentResult
from ..content_creator import ContentCreator
from ..social_media_manager import SocialMediaManager
from ..marketing_analyzer import MarketingAnalyzer

@dataclass
class InterviewQuestion:
    question: str
    category: str
    difficulty: str
    expected_answer: str
    scoring_criteria: List[str]
    marketable_content: bool = False

@dataclass
class InterviewResponse:
    question: str
    user_answer: str
    ai_feedback: str
    score: float
    improvements: List[str]
    content_opportunities: List[str]

@dataclass
class MarketingContent:
    content_type: str  # post, reel, short, story, carousel
    platform: str
    content: str
    media_url: Optional[str] = None
    hashtags: List[str] = None
    scheduled_time: Optional[datetime] = None
    target_audience: Dict[str, Any] = None

class InterviewAgent(BaseAIAgent):
    """
    Enhanced AI agent for interview preparation with integrated marketing content creation
    and social media automation capabilities.
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__("interview_agent", model_config)
        self.interview_types = [
            "technical",
            "behavioral", 
            "cultural",
            "case_study",
            "system_design",
            "leadership",
            "sales",
            "executive"
        ]
        self.difficulty_levels = ["junior", "mid", "senior", "expert", "executive"]
        
        # Initialize marketing components
        self.content_creator = ContentCreator()
        self.social_media_manager = SocialMediaManager()
        self.marketing_analyzer = MarketingAnalyzer()
        
        # Marketing configuration
        self.marketing_config = {
            "auto_generate_content": True,
            "auto_post_to_socials": True,
            "content_scheduling": True,
            "performance_tracking": True,
            "platforms": ["tiktok", "instagram", "facebook", "x", "linkedin", "youtube_shorts"]
        }
    
    async def process(self, interview_request: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Enhanced interview processing with marketing content generation.
        
        Args:
            interview_request: Dictionary containing interview parameters
            **kwargs: Additional interview and marketing options
            
        Returns:
            AgentResult with interview results, marketing content, and analytics
        """
        import time
        start_time = time.time()
        
        try:
            if not self.validate_input(interview_request):
                return AgentResult(
                    success=False,
                    data=None,
                    error="Invalid interview request format"
                )
            
            self.status = self.AgentStatus.PROCESSING
            
            interview_type = interview_request.get("type", "technical")
            difficulty = interview_request.get("difficulty", "mid")
            user_profile = interview_request.get("user_profile", {})
            marketing_enabled = interview_request.get("enable_marketing", True)
            
            # Generate interview questions
            questions = await self._generate_questions(interview_type, difficulty, user_profile)
            
            # If user provided answers, evaluate them
            user_answers = interview_request.get("answers", {})
            evaluation_results = []
            marketing_contents = []
            
            if user_answers:
                evaluation_results = await self._evaluate_answers(questions, user_answers, user_profile)
                
                # Generate marketing content from insights
                if marketing_enabled:
                    marketing_contents = await self._generate_marketing_content(
                        evaluation_results, user_profile, interview_type
                    )
            
            # Generate overall feedback
            overall_feedback = await self._generate_overall_feedback(evaluation_results, user_profile)
            
            # Post to social media if enabled
            social_media_results = []
            if marketing_enabled and marketing_contents and self.marketing_config["auto_post_to_socials"]:
                social_media_results = await self._post_to_social_media(marketing_contents)
            
            processing_time = time.time() - start_time
            self.update_metrics(True, processing_time)
            
            return AgentResult(
                success=True,
                data={
                    "questions": questions,
                    "evaluations": evaluation_results,
                    "overall_feedback": overall_feedback,
                    "marketing_content": marketing_contents,
                    "social_media_results": social_media_results,
                    "interview_type": interview_type,
                    "difficulty": difficulty,
                    "analytics": await self._generate_marketing_analytics(marketing_contents, social_media_results)
                },
                metadata={
                    "total_questions": len(questions),
                    "questions_answered": len(user_answers),
                    "marketing_content_generated": len(marketing_contents),
                    "social_media_posts": len(social_media_results),
                    "processing_time": processing_time
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(False, processing_time)
            self.logger.error(f"Enhanced interview processing failed: {e}")
            
            return AgentResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    async def conduct_live_interview(self, user_profile: Dict[str, Any], 
                                   interview_params: Dict[str, Any]) -> AgentResult:
        """
        Conduct a live, interactive interview session with real-time content generation.
        
        Args:
            user_profile: User's professional background
            interview_params: Interview configuration
            
        Returns:
            AgentResult with live interview session and marketing insights
        """
        try:
            session_id = self._generate_session_id()
            interview_type = interview_params.get("type", "technical")
            difficulty = interview_params.get("difficulty", "mid")
            enable_recording = interview_params.get("enable_recording", True)
            
            # Initialize enhanced interview session
            session = {
                "session_id": session_id,
                "interview_type": interview_type,
                "difficulty": difficulty,
                "questions_asked": [],
                "user_responses": [],
                "current_question_index": 0,
                "session_active": True,
                "recording_enabled": enable_recording,
                "marketing_insights": [],
                "content_opportunities": []
            }
            
            # Generate initial question set with marketing potential
            questions = await self._generate_questions(interview_type, difficulty, user_profile)
            session["questions"] = questions
            
            # Pre-generate content templates
            if enable_recording:
                session["content_templates"] = await self._generate_content_templates(questions, user_profile)
            
            return AgentResult(
                success=True,
                data=session,
                metadata={
                    "session_id": session_id,
                    "marketing_ready": True,
                    "content_templates_created": len(session.get("content_templates", []))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Live interview setup failed: {e}")
            return AgentResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    async def generate_interview_content_package(self, user_profile: Dict[str, Any],
                                               interview_results: Dict[str, Any]) -> AgentResult:
        """
        Generate comprehensive marketing content package from interview results.
        
        Args:
            user_profile: User's professional background
            interview_results: Results from completed interview
            
        Returns:
            AgentResult with complete content package
        """
        try:
            content_package = {
                "user_profile": user_profile,
                "interview_results": interview_results,
                "content_assets": {},
                "social_media_calendar": {},
                "performance_metrics": {}
            }
            
            # Generate different content types
            content_tasks = [
                self._generate_video_content(interview_results, user_profile),
                self._generate_image_content(interview_results, user_profile),
                self._generate_text_content(interview_results, user_profile),
                self._generate_audio_content(interview_results, user_profile),
                self._generate_carousel_content(interview_results, user_profile)
            ]
            
            content_results = await asyncio.gather(*content_tasks, return_exceptions=True)
            
            # Organize content assets
            content_package["content_assets"] = {
                "videos": content_results[0] if not isinstance(content_results[0], Exception) else [],
                "images": content_results[1] if not isinstance(content_results[1], Exception) else [],
                "text_posts": content_results[2] if not isinstance(content_results[2], Exception) else [],
                "audio_clips": content_results[3] if not isinstance(content_results[3], Exception) else [],
                "carousels": content_results[4] if not isinstance(content_results[4], Exception) else []
            }
            
            # Create social media calendar
            content_package["social_media_calendar"] = await self._create_content_calendar(
                content_package["content_assets"]
            )
            
            # Generate performance predictions
            content_package["performance_metrics"] = await self._predict_content_performance(
                content_package["content_assets"]
            )
            
            return AgentResult(
                success=True,
                data=content_package,
                metadata={
                    "total_assets": sum(len(assets) for assets in content_package["content_assets"].values()),
                    "calendar_entries": len(content_package["social_media_calendar"]),
                    "platform_coverage": len(self.marketing_config["platforms"])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Content package generation failed: {e}")
            return AgentResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    async def _generate_questions(self, interview_type: str, difficulty: str,
                                user_profile: Dict[str, Any]) -> List[InterviewQuestion]:
        """Generate interview questions with marketing content potential."""
        from ..deep_agents.nlp_models.question_generator import QuestionGenerator
        
        generator = QuestionGenerator()
        questions_data = await generator.generate_questions(
            interview_type=interview_type,
            difficulty=difficulty,
            user_profile=user_profile
        )
        
        questions = []
        for q_data in questions_data:
            # Assess marketing potential
            marketable = await self._assess_marketing_potential(q_data["question"], interview_type)
            
            question = InterviewQuestion(
                question=q_data["question"],
                category=q_data["category"],
                difficulty=q_data["difficulty"],
                expected_answer=q_data.get("expected_answer", ""),
                scoring_criteria=q_data.get("scoring_criteria", []),
                marketable_content=marketable
            )
            questions.append(question)
        
        return questions
    
    async def _evaluate_answers(self, questions: List[InterviewQuestion],
                              user_answers: Dict[str, str],
                              user_profile: Dict[str, Any]) -> List[InterviewResponse]:
        """Evaluate user answers and identify content opportunities."""
        evaluations = []
        
        for question in questions:
            if question.question in user_answers:
                user_answer = user_answers[question.question]
                evaluation = await self._evaluate_single_answer(question, user_answer, user_profile)
                
                # Identify content creation opportunities
                if question.marketable_content:
                    content_ops = await self._identify_content_opportunities(question, evaluation, user_profile)
                    evaluation.content_opportunities = content_ops
                
                evaluations.append(evaluation)
        
        return evaluations
    
    async def _evaluate_single_answer(self, question: InterviewQuestion,
                                    user_answer: str, 
                                    user_profile: Dict[str, Any]) -> InterviewResponse:
        """Evaluate a single answer using multiple criteria."""
        from ..deep_agents.nlp_models.answer_evaluator import AnswerEvaluator
        
        evaluator = AnswerEvaluator()
        evaluation = await evaluator.evaluate_answer(
            question=question.question,
            user_answer=user_answer,
            expected_answer=question.expected_answer,
            scoring_criteria=question.scoring_criteria,
            user_profile=user_profile
        )
        
        response = InterviewResponse(
            question=question.question,
            user_answer=user_answer,
            ai_feedback=evaluation["feedback"],
            score=evaluation["score"],
            improvements=evaluation["improvements"],
            content_opportunities=[]
        )
        
        return response
    
    async def _generate_marketing_content(self, evaluations: List[InterviewResponse],
                                        user_profile: Dict[str, Any],
                                        interview_type: str) -> List[MarketingContent]:
        """Generate marketing content from interview insights."""
        marketing_contents = []
        
        for evaluation in evaluations:
            if evaluation.content_opportunities:
                for opportunity in evaluation.content_opportunities:
                    try:
                        content = await self._create_specific_content(
                            opportunity, evaluation, user_profile, interview_type
                        )
                        if content:
                            marketing_contents.extend(content)
                    except Exception as e:
                        self.logger.warning(f"Content creation failed for opportunity {opportunity}: {e}")
                        continue
        
        # Generate summary content
        summary_content = await self._create_summary_content(evaluations, user_profile, interview_type)
        marketing_contents.extend(summary_content)
        
        return marketing_contents
    
    async def _create_specific_content(self, opportunity: str, evaluation: InterviewResponse,
                                     user_profile: Dict[str, Any], interview_type: str) -> List[MarketingContent]:
        """Create specific marketing content based on opportunity."""
        contents = []
        
        if "video" in opportunity.lower():
            # Create video content (Reels/Shorts/TikToks)
            video_content = await self.content_creator.create_video_content(
                topic=evaluation.question,
                insights=evaluation.ai_feedback,
                user_profile=user_profile,
                style="educational",
                duration=60  # seconds
            )
            contents.extend(video_content)
        
        if "post" in opportunity.lower() or "text" in opportunity.lower():
            # Create text content for social media
            text_content = await self.content_creator.create_text_content(
                topic=evaluation.question,
                insights=evaluation.ai_feedback,
                platform="all",
                tone="professional"
            )
            contents.extend(text_content)
        
        if "carousel" in opportunity.lower():
            # Create carousel content for Instagram/LinkedIn
            carousel_content = await self.content_creator.create_carousel_content(
                topic=evaluation.question,
                key_points=evaluation.improvements,
                user_profile=user_profile
            )
            contents.extend(carousel_content)
        
        return contents
    
    async def _create_summary_content(self, evaluations: List[InterviewResponse],
                                   user_profile: Dict[str, Any], interview_type: str) -> List[MarketingContent]:
        """Create summary content from all evaluations."""
        summary_contents = []
        
        # Calculate overall performance
        overall_score = np.mean([e.score for e in evaluations]) if evaluations else 0
        strengths = [e for e in evaluations if e.score >= 0.8]
        improvements = [e for e in evaluations if e.score < 0.6]
        
        # Create performance summary content
        summary_data = {
            "overall_score": overall_score,
            "total_questions": len(evaluations),
            "strengths_count": len(strengths),
            "improvements_count": len(improvements),
            "interview_type": interview_type
        }
        
        # Generate summary video
        summary_video = await self.content_creator.create_summary_video(
            summary_data=summary_data,
            user_profile=user_profile,
            interview_type=interview_type
        )
        if summary_video:
            summary_contents.append(summary_video)
        
        # Generate infographic
        infographic = await self.content_creator.create_infographic(
            summary_data=summary_data,
            user_profile=user_profile
        )
        if infographic:
            summary_contents.append(infographic)
        
        return summary_contents
    
    async def _post_to_social_media(self, marketing_contents: List[MarketingContent]) -> List[Dict[str, Any]]:
        """Post marketing content to social media platforms."""
        posting_results = []
        
        for content in marketing_contents:
            try:
                result = await self.social_media_manager.post_content(
                    content=content,
                    platforms=self.marketing_config["platforms"],
                    schedule=content.scheduled_time
                )
                posting_results.append(result)
            except Exception as e:
                self.logger.error(f"Social media posting failed for {content.content_type}: {e}")
                posting_results.append({
                    "success": False,
                    "platform": content.platform,
                    "error": str(e),
                    "content_type": content.content_type
                })
        
        return posting_results
    
    async def _generate_marketing_analytics(self, marketing_contents: List[MarketingContent],
                                          social_media_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate marketing analytics and performance insights."""
        return await self.marketing_analyzer.analyze_campaign_performance(
            contents=marketing_contents,
            posting_results=social_media_results
        )
    
    async def _identify_content_opportunities(self, question: InterviewQuestion,
                                            evaluation: InterviewResponse,
                                            user_profile: Dict[str, Any]) -> List[str]:
        """Identify content creation opportunities from questions and answers."""
        opportunities = []
        
        # Content opportunity detection logic
        if evaluation.score >= 0.8:
            opportunities.extend([
                "success_story_video",
                "tip_carousel",
                "expert_insight_post"
            ])
        elif evaluation.score <= 0.5:
            opportunities.extend([
                "learning_moment_video",
                "common_mistake_post",
                "improvement_tips_reel"
            ])
        
        # Additional opportunities based on question category
        if question.category == "technical":
            opportunities.extend(["technical_tutorial", "code_review_short"])
        elif question.category == "behavioral":
            opportunities.extend(["storytelling_post", "soft_skills_video"])
        
        return opportunities
    
    async def _assess_marketing_potential(self, question: str, interview_type: str) -> bool:
        """Assess if a question has marketing content potential."""
        # Keywords that indicate high marketing potential
        high_potential_keywords = [
            "how would you", "tell me about", "describe a time",
            "what is your approach", "explain", "demonstrate"
        ]
        
        question_lower = question.lower()
        has_potential = any(keyword in question_lower for keyword in high_potential_keywords)
        
        # Certain interview types have higher marketing potential
        high_potential_types = ["technical", "behavioral", "case_study"]
        type_multiplier = 2 if interview_type in high_potential_types else 1
        
        return has_potential and type_multiplier > 0
    
    async def _generate_content_templates(self, questions: List[InterviewQuestion],
                                        user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate content templates for potential marketing content."""
        templates = []
        
        for question in questions:
            if question.marketable_content:
                template = {
                    "question": question.question,
                    "category": question.category,
                    "content_ideas": await self._generate_content_ideas(question, user_profile),
                    "suggested_platforms": self._suggest_platforms(question.category),
                    "estimated_engagement": await self._estimate_engagement(question, user_profile)
                }
                templates.append(template)
        
        return templates
    
    async def _generate_content_ideas(self, question: InterviewQuestion,
                                    user_profile: Dict[str, Any]) -> List[str]:
        """Generate specific content ideas for a question."""
        content_ideas = []
        
        base_ideas = {
            "technical": [
                f"Step-by-step solution for: {question.question}",
                f"Common mistakes in answering: {question.question}",
                f"Expert approach to: {question.question}"
            ],
            "behavioral": [
                f"Storytelling framework for: {question.question}",
                f"Real-world example answering: {question.question}",
                f"What interviewers really want to hear about: {question.question}"
            ],
            "case_study": [
                f"Framework breakdown for: {question.question}",
                f"Case study walkthrough: {question.question}",
                f"Structured approach to: {question.question}"
            ]
        }
        
        category_ideas = base_ideas.get(question.category, [])
        content_ideas.extend(category_ideas)
        
        return content_ideas
    
    def _suggest_platforms(self, category: str) -> List[str]:
        """Suggest social media platforms based on content category."""
        platform_suggestions = {
            "technical": ["youtube_shorts", "linkedin", "x", "tiktok"],
            "behavioral": ["instagram", "facebook", "linkedin", "tiktok"],
            "case_study": ["linkedin", "x", "instagram_carousel"],
            "cultural": ["instagram", "facebook", "tiktok"]
        }
        
        return platform_suggestions.get(category, ["linkedin", "x"])
    
    async def _estimate_engagement(self, question: InterviewQuestion,
                                 user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Estimate potential engagement for content."""
        # This would integrate with historical performance data
        return {
            "estimated_views": 1000 + len(question.question) * 10,
            "estimated_likes": 100 + len(question.question) * 5,
            "estimated_shares": 20 + len(question.question) * 2,
            "engagement_rate": 0.05  # 5% engagement rate
        }
    
    async def _generate_video_content(self, interview_results: Dict[str, Any],
                                   user_profile: Dict[str, Any]) -> List[MarketingContent]:
        """Generate video content from interview results."""
        return await self.content_creator.create_video_package(interview_results, user_profile)
    
    async def _generate_image_content(self, interview_results: Dict[str, Any],
                                    user_profile: Dict[str, Any]) -> List[MarketingContent]:
        """Generate image content from interview results."""
        return await self.content_creator.create_image_package(interview_results, user_profile)
    
    async def _generate_text_content(self, interview_results: Dict[str, Any],
                                  user_profile: Dict[str, Any]) -> List[MarketingContent]:
        """Generate text content from interview results."""
        return await self.content_creator.create_text_package(interview_results, user_profile)
    
    async def _generate_audio_content(self, interview_results: Dict[str, Any],
                                   user_profile: Dict[str, Any]) -> List[MarketingContent]:
        """Generate audio content from interview results."""
        return await self.content_creator.create_audio_package(interview_results, user_profile)
    
    async def _generate_carousel_content(self, interview_results: Dict[str, Any],
                                       user_profile: Dict[str, Any]) -> List[MarketingContent]:
        """Generate carousel content from interview results."""
        return await self.content_creator.create_carousel_package(interview_results, user_profile)
    
    async def _create_content_calendar(self, content_assets: Dict[str, List[MarketingContent]]) -> Dict[str, Any]:
        """Create social media content calendar."""
        return await self.content_creator.generate_content_calendar(content_assets)
    
    async def _predict_content_performance(self, content_assets: Dict[str, List[MarketingContent]]) -> Dict[str, Any]:
        """Predict content performance across platforms."""
        return await self.marketing_analyzer.predict_performance(content_assets)
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())
    
    async def _generate_overall_feedback(self, evaluations: List[InterviewResponse],
                                       user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall feedback with marketing insights."""
        if not evaluations:
            return {
                "overall_score": 0,
                "strengths": [],
                "weaknesses": [],
                "recommendations": ["No answers provided for evaluation"],
                "content_opportunities": []
            }
        
        # Calculate overall score
        total_score = sum(eval_obj.score for eval_obj in evaluations)
        overall_score = total_score / len(evaluations)
        
        # Enhanced feedback with marketing insights
        feedback = {
            "overall_score": overall_score,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "content_opportunities": [],
            "marketing_potential": await self._calculate_marketing_potential(evaluations),
            "estimated_audience_reach": await self._estimate_audience_reach(evaluations, user_profile)
        }
        
        # Identify strengths and weaknesses
        for eval_obj in evaluations:
            if eval_obj.score >= 0.8:
                category = self._categorize_question(eval_obj.question)
                if category not in feedback["strengths"]:
                    feedback["strengths"].append(category)
            elif eval_obj.score <= 0.5:
                category = self._categorize_question(eval_obj.question)
                if category not in feedback["weaknesses"]:
                    feedback["weaknesses"].append(category)
        
        # Generate recommendations with content focus
        feedback["recommendations"] = await self._generate_content_focused_recommendations(evaluations, user_profile)
        
        # Aggregate content opportunities
        all_opportunities = []
        for eval_obj in evaluations:
            all_opportunities.extend(eval_obj.content_opportunities)
        feedback["content_opportunities"] = list(set(all_opportunities))
        
        feedback["performance_level"] = self._get_performance_level(overall_score)
        
        return feedback
    
    async def _calculate_marketing_potential(self, evaluations: List[InterviewResponse]) -> float:
        """Calculate overall marketing potential from evaluations."""
        if not evaluations:
            return 0.0
        
        content_opportunities = sum(len(eval_obj.content_opportunities) for eval_obj in evaluations)
        max_possible_opportunities = len(evaluations) * 3  # Assume max 3 opportunities per evaluation
        
        return min(1.0, content_opportunities / max_possible_opportunities)
    
    async def _estimate_audience_reach(self, evaluations: List[InterviewResponse],
                                     user_profile: Dict[str, Any]) -> Dict[str, int]:
        """Estimate potential audience reach for generated content."""
        # This would integrate with audience analytics
        return {
            "total_potential_reach": 10000,
            "linkedin_reach": 3000,
            "instagram_reach": 4000,
            "tiktok_reach": 5000,
            "youtube_reach": 2000,
            "estimated_impressions": 50000
        }
    
    async def _generate_content_focused_recommendations(self, evaluations: List[InterviewResponse],
                                                      user_profile: Dict[str, Any]) -> List[str]:
        """Generate recommendations focused on content creation."""
        recommendations = []
        
        # Analyze common issues for content ideas
        low_scoring_answers = [e for e in evaluations if e.score < 0.6]
        
        if low_scoring_answers:
            common_issues = await self._analyze_common_issues(low_scoring_answers)
            
            for issue in common_issues:
                if issue == "clarity":
                    recommendations.append("Create content on structuring clear, concise answers")
                elif issue == "depth":
                    recommendations.append("Develop video content demonstrating detailed explanations")
                elif issue == "relevance":
                    recommendations.append("Produce content on tailoring answers to specific roles")
        
        # Add content-specific recommendations
        high_scoring_answers = [e for e in evaluations if e.score >= 0.8]
        if high_scoring_answers:
            recommendations.append("Leverage your strong answers for expert content creation")
        
        # Platform-specific recommendations
        recommendations.extend([
            "Create TikTok/Reels for quick tips from your interview experience",
            "Develop LinkedIn carousels for in-depth answer breakdowns",
            "Produce YouTube Shorts for common interview questions"
        ])
        
        return recommendations
    
    async def _analyze_common_issues(self, low_scoring_answers: List[InterviewResponse]) -> List[str]:
        """Analyze common issues in low-scoring answers."""
        from ..deep_agents.nlp_models.issue_detector import IssueDetector
        
        detector = IssueDetector()
        issues = await detector.detect_common_issues(
            [eval_obj.user_answer for eval_obj in low_scoring_answers]
        )
        
        return issues
    
    def _categorize_question(self, question: str) -> str:
        """Categorize a question into a general category."""
        question_lower = question.lower()
        
        if any(term in question_lower for term in ["algorithm", "data structure", "complexity"]):
            return "algorithms"
        elif any(term in question_lower for term in ["system", "architecture", "design"]):
            return "system_design"
        elif any(term in question_lower for term in ["experience", "project", "worked"]):
            return "experience"
        elif any(term in question_lower for term in ["behavior", "situation", "conflict"]):
            return "behavioral"
        elif any(term in question_lower for term in ["culture", "team", "collaboration"]):
            return "cultural"
        elif any(term in question_lower for term in ["leadership", "manage", "team"]):
            return "leadership"
        else:
            return "general"
    
    def _get_performance_level(self, score: float) -> str:
        """Convert numerical score to performance level."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "satisfactory"
        elif score >= 0.6:
            return "needs_improvement"
        else:
            return "poor"

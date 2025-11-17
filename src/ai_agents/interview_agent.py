import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import logging
from .base_agent import BaseAIAgent, AgentResult

@dataclass
class InterviewQuestion:
    question: str
    category: str
    difficulty: str
    expected_answer: str
    scoring_criteria: List[str]

@dataclass
class InterviewResponse:
    question: str
    user_answer: str
    ai_feedback: str
    score: float
    improvements: List[str]

class InterviewAgent(BaseAIAgent):
    """
    AI agent for interview preparation and simulation.
    Provides realistic interview practice with AI-powered feedback.
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__("interview_agent", model_config)
        self.interview_types = [
            "technical",
            "behavioral", 
            "cultural",
            "case_study",
            "system_design"
        ]
        self.difficulty_levels = ["junior", "mid", "senior", "expert"]
    
    async def process(self, interview_request: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Conduct an interview simulation or provide interview preparation.
        
        Args:
            interview_request: Dictionary containing interview parameters
            **kwargs: Additional interview options
            
        Returns:
            AgentResult with interview results and feedback
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
            
            # Generate interview questions
            questions = await self._generate_questions(interview_type, difficulty, user_profile)
            
            # If user provided answers, evaluate them
            user_answers = interview_request.get("answers", {})
            evaluation_results = []
            
            if user_answers:
                evaluation_results = await self._evaluate_answers(questions, user_answers, user_profile)
            
            # Generate overall feedback
            overall_feedback = await self._generate_overall_feedback(evaluation_results, user_profile)
            
            processing_time = time.time() - start_time
            self.update_metrics(True, processing_time)
            
            return AgentResult(
                success=True,
                data={
                    "questions": questions,
                    "evaluations": evaluation_results,
                    "overall_feedback": overall_feedback,
                    "interview_type": interview_type,
                    "difficulty": difficulty
                },
                metadata={
                    "total_questions": len(questions),
                    "questions_answered": len(user_answers),
                    "processing_time": processing_time
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(False, processing_time)
            self.logger.error(f"Interview processing failed: {e}")
            
            return AgentResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate interview request data."""
        if "type" not in input_data:
            return False
        
        if input_data["type"] not in self.interview_types:
            return False
        
        if "difficulty" in input_data and input_data["difficulty"] not in self.difficulty_levels:
            return False
        
        return True
    
    async def conduct_live_interview(self, user_profile: Dict[str, Any], 
                                   interview_params: Dict[str, Any]) -> AgentResult:
        """
        Conduct a live, interactive interview session.
        
        Args:
            user_profile: User's professional background
            interview_params: Interview configuration
            
        Returns:
            AgentResult with live interview session
        """
        try:
            session_id = self._generate_session_id()
            interview_type = interview_params.get("type", "technical")
            difficulty = interview_params.get("difficulty", "mid")
            
            # Initialize interview session
            session = {
                "session_id": session_id,
                "interview_type": interview_type,
                "difficulty": difficulty,
                "questions_asked": [],
                "user_responses": [],
                "current_question_index": 0,
                "session_active": True
            }
            
            # Generate initial question set
            questions = await self._generate_questions(interview_type, difficulty, user_profile)
            session["questions"] = questions
            
            return AgentResult(
                success=True,
                data=session,
                metadata={"session_id": session_id}
            )
            
        except Exception as e:
            self.logger.error(f"Live interview setup failed: {e}")
            return AgentResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    async def _generate_questions(self, interview_type: str, difficulty: str,
                                user_profile: Dict[str, Any]) -> List[InterviewQuestion]:
        """Generate interview questions based on type and difficulty."""
        from ..deep_agents.nlp_models.question_generator import QuestionGenerator
        
        generator = QuestionGenerator()
        questions_data = await generator.generate_questions(
            interview_type=interview_type,
            difficulty=difficulty,
            user_profile=user_profile
        )
        
        questions = []
        for q_data in questions_data:
            question = InterviewQuestion(
                question=q_data["question"],
                category=q_data["category"],
                difficulty=q_data["difficulty"],
                expected_answer=q_data.get("expected_answer", ""),
                scoring_criteria=q_data.get("scoring_criteria", [])
            )
            questions.append(question)
        
        return questions
    
    async def _evaluate_answers(self, questions: List[InterviewQuestion],
                              user_answers: Dict[str, str],
                              user_profile: Dict[str, Any]) -> List[InterviewResponse]:
        """Evaluate user answers against expected responses."""
        evaluations = []
        
        for question in questions:
            if question.question in user_answers:
                user_answer = user_answers[question.question]
                evaluation = await self._evaluate_single_answer(question, user_answer, user_profile)
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
            improvements=evaluation["improvements"]
        )
        
        return response
    
    async def _generate_overall_feedback(self, evaluations: List[InterviewResponse],
                                       user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall feedback based on all evaluations."""
        if not evaluations:
            return {
                "overall_score": 0,
                "strengths": [],
                "weaknesses": [],
                "recommendations": ["No answers provided for evaluation"]
            }
        
        # Calculate overall score
        total_score = sum(eval_obj.score for eval_obj in evaluations)
        overall_score = total_score / len(evaluations)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for eval_obj in evaluations:
            if eval_obj.score >= 0.8:
                category = self._categorize_question(eval_obj.question)
                if category not in strengths:
                    strengths.append(category)
            elif eval_obj.score <= 0.5:
                category = self._categorize_question(eval_obj.question)
                if category not in weaknesses:
                    weaknesses.append(category)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(evaluations, user_profile)
        
        return {
            "overall_score": overall_score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "performance_level": self._get_performance_level(overall_score)
        }
    
    async def _generate_recommendations(self, evaluations: List[InterviewResponse],
                                      user_profile: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations for improvement."""
        recommendations = []
        
        # Analyze common issues
        low_scoring_answers = [e for e in evaluations if e.score < 0.6]
        
        if low_scoring_answers:
            # Identify patterns in low-scoring answers
            common_issues = await self._analyze_common_issues(low_scoring_answers)
            
            for issue in common_issues:
                if issue == "clarity":
                    recommendations.append("Work on providing clearer, more structured answers")
                elif issue == "depth":
                    recommendations.append("Provide more detailed explanations with examples")
                elif issue == "relevance":
                    recommendations.append("Focus on making answers more relevant to the question")
        
        # Add strategic recommendations
        experience_level = user_profile.get("experience", {}).get("years", 0)
        if experience_level < 3:
            recommendations.append("Practice explaining technical concepts in simple terms")
        else:
            recommendations.append("Focus on leadership and architectural decision examples")
        
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
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())

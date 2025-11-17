import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import logging
from datetime import datetime, timedelta
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ..base_agent import BaseAIAgent, AgentResult

@dataclass
class VideoContent:
    title: str
    description: str
    script: str
    duration: int
    style: str
    platform: str
    hashtags: List[str]
    thumbnail_url: Optional[str] = None
    audio_track: Optional[str] = None
    visual_elements: List[str] = None

@dataclass
class ImageContent:
    title: str
    caption: str
    style: str
    platform: str
    dimensions: tuple
    elements: List[str]
    branding: Dict[str, Any] = None

@dataclass
class AudioContent:
    title: str
    script: str
    duration: int
    voice_style: str
    background_music: Optional[str] = None
    platform: str = "all"

class ContentCreator(BaseAIAgent):
    """
    AI-powered content creator for generating marketing content across multiple formats
    and platforms including videos, images, audio, and text.
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__("content_creator", model_config)
        self.content_templates = self._load_content_templates()
        self.platform_specs = self._load_platform_specifications()
        self.brand_guidelines = self._load_brand_guidelines()
        
    async def create_video_content(self, topic: str, insights: str, 
                                 user_profile: Dict[str, Any], style: str = "educational",
                                 duration: int = 60) -> List[VideoContent]:
        """
        Create video content for platforms like TikTok, Reels, Shorts.
        
        Args:
            topic: Main topic for the video
            insights: Key insights to include
            user_profile: User profile for personalization
            style: Content style (educational, entertaining, inspirational)
            duration: Video duration in seconds
            
        Returns:
            List of VideoContent objects for different platforms
        """
        try:
            video_contents = []
            
            # Generate content for each platform
            platforms = ["tiktok", "instagram_reels", "youtube_shorts"]
            
            for platform in platforms:
                # Generate platform-specific content
                video_content = await self._generate_platform_video(
                    platform=platform,
                    topic=topic,
                    insights=insights,
                    user_profile=user_profile,
                    style=style,
                    duration=duration
                )
                
                if video_content:
                    video_contents.append(video_content)
            
            return video_contents
            
        except Exception as e:
            self.logger.error(f"Video content creation failed: {e}")
            return []
    
    async def create_text_content(self, topic: str, insights: str, 
                                platform: str = "all", tone: str = "professional") -> List[Dict[str, Any]]:
        """
        Create text content for social media posts.
        
        Args:
            topic: Main topic
            insights: Key insights
            platform: Target platform
            tone: Content tone
            
        Returns:
            List of text content objects
        """
        try:
            text_contents = []
            
            platforms = self._get_text_platforms(platform)
            
            for platform_name in platforms:
                content = await self._generate_platform_text(
                    platform=platform_name,
                    topic=topic,
                    insights=insights,
                    tone=tone
                )
                
                if content:
                    text_contents.append(content)
            
            return text_contents
            
        except Exception as e:
            self.logger.error(f"Text content creation failed: {e}")
            return []
    
    async def create_carousel_content(self, topic: str, key_points: List[str],
                                    user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create carousel content for Instagram and LinkedIn.
        
        Args:
            topic: Main topic
            key_points: Key points to include
            user_profile: User profile for personalization
            
        Returns:
            List of carousel content objects
        """
        try:
            carousels = []
            
            # Generate carousel for each supported platform
            platforms = ["instagram", "linkedin"]
            
            for platform in platforms:
                carousel = await self._generate_platform_carousel(
                    platform=platform,
                    topic=topic,
                    key_points=key_points,
                    user_profile=user_profile
                )
                
                if carousel:
                    carousels.append(carousel)
            
            return carousels
            
        except Exception as e:
            self.logger.error(f"Carousel content creation failed: {e}")
            return []
    
    async def create_audio_content(self, topic: str, script: str,
                                 duration: int, voice_style: str = "professional") -> List[AudioContent]:
        """
        Create audio content for podcasts and social media.
        
        Args:
            topic: Main topic
            script: Audio script
            duration: Duration in seconds
            voice_style: Voice style for narration
            
        Returns:
            List of AudioContent objects
        """
        try:
            audio_contents = []
            
            # Generate different audio formats
            formats = ["podcast_snippet", "voiceover", "social_audio"]
            
            for audio_format in formats:
                audio_content = await self._generate_audio_format(
                    format_type=audio_format,
                    topic=topic,
                    script=script,
                    duration=duration,
                    voice_style=voice_style
                )
                
                if audio_content:
                    audio_contents.append(audio_content)
            
            return audio_contents
            
        except Exception as e:
            self.logger.error(f"Audio content creation failed: {e}")
            return []
    
    async def create_summary_video(self, summary_data: Dict[str, Any],
                                 user_profile: Dict[str, Any], interview_type: str) -> Optional[VideoContent]:
        """
        Create a summary video from interview results.
        
        Args:
            summary_data: Interview summary data
            user_profile: User profile
            interview_type: Type of interview
            
        Returns:
            VideoContent object or None
        """
        try:
            # Generate video script
            script = await self._generate_summary_script(summary_data, user_profile, interview_type)
            
            # Create video content
            video_content = VideoContent(
                title=f"Interview Performance Summary - {interview_type.title()}",
                description=script[:200] + "...",  # First 200 characters
                script=script,
                duration=90,  # 1.5 minutes
                style="summary",
                platform="all",
                hashtags=self._generate_summary_hashtags(interview_type, summary_data),
                visual_elements=["performance_charts", "key_metrics", "improvement_tips"]
            )
            
            return video_content
            
        except Exception as e:
            self.logger.error(f"Summary video creation failed: {e}")
            return None
    
    async def create_infographic(self, summary_data: Dict[str, Any],
                              user_profile: Dict[str, Any]) -> Optional[ImageContent]:
        """
        Create an infographic from interview results.
        
        Args:
            summary_data: Interview summary data
            user_profile: User profile
            
        Returns:
            ImageContent object or None
        """
        try:
            infographic = ImageContent(
                title="Interview Performance Infographic",
                caption=await self._generate_infographic_caption(summary_data),
                style="infographic",
                platform="all",
                dimensions=(1080, 1920),  # Instagram story size
                elements=[
                    "performance_score",
                    "strengths_weaknesses",
                    "improvement_areas",
                    "key_metrics"
                ],
                branding={
                    "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
                    "font": "Montserrat",
                    "logo_position": "top_right"
                }
            )
            
            return infographic
            
        except Exception as e:
            self.logger.error(f"Infographic creation failed: {e}")
            return None
    
    async def create_video_package(self, interview_results: Dict[str, Any],
                                 user_profile: Dict[str, Any]) -> List[VideoContent]:
        """Create comprehensive video package from interview results."""
        # Implementation for video package creation
        return []
    
    async def create_image_package(self, interview_results: Dict[str, Any],
                                 user_profile: Dict[str, Any]) -> List[ImageContent]:
        """Create comprehensive image package from interview results."""
        # Implementation for image package creation
        return []
    
    async def create_text_package(self, interview_results: Dict[str, Any],
                                user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create comprehensive text package from interview results."""
        # Implementation for text package creation
        return []
    
    async def create_audio_package(self, interview_results: Dict[str, Any],
                                 user_profile: Dict[str, Any]) -> List[AudioContent]:
        """Create comprehensive audio package from interview results."""
        # Implementation for audio package creation
        return []
    
    async def create_carousel_package(self, interview_results: Dict[str, Any],
                                   user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create comprehensive carousel package from interview results."""
        # Implementation for carousel package creation
        return []
    
    async def generate_content_calendar(self, content_assets: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Generate social media content calendar."""
        calendar = {
            "weekly_schedule": {},
            "content_mix": {},
            "optimization_suggestions": [],
            "performance_predictions": {}
        }
        
        # Generate weekly schedule
        calendar["weekly_schedule"] = await self._generate_weekly_schedule(content_assets)
        
        # Analyze content mix
        calendar["content_mix"] = self._analyze_content_mix(content_assets)
        
        # Generate optimization suggestions
        calendar["optimization_suggestions"] = await self._generate_optimization_suggestions(content_assets)
        
        return calendar
    
    async def _generate_platform_video(self, platform: str, topic: str, insights: str,
                                    user_profile: Dict[str, Any], style: str, duration: int) -> Optional[VideoContent]:
        """Generate platform-specific video content."""
        try:
            # Get platform specifications
            platform_spec = self.platform_specs.get(platform, {})
            
            # Generate video script
            script = await self._generate_video_script(topic, insights, user_profile, style, duration)
            
            # Generate title and description
            title = await self._generate_video_title(topic, platform, style)
            description = await self._generate_video_description(script, platform)
            
            # Generate hashtags
            hashtags = await self._generate_video_hashtags(topic, platform, style)
            
            video_content = VideoContent(
                title=title,
                description=description,
                script=script,
                duration=duration,
                style=style,
                platform=platform,
                hashtags=hashtags,
                visual_elements=platform_spec.get("visual_elements", [])
            )
            
            return video_content
            
        except Exception as e:
            self.logger.error(f"Platform video generation failed for {platform}: {e}")
            return None
    
    async def _generate_platform_text(self, platform: str, topic: str, 
                                   insights: str, tone: str) -> Optional[Dict[str, Any]]:
        """Generate platform-specific text content."""
        try:
            # Generate text based on platform and tone
            content = await self._generate_platform_specific_text(platform, topic, insights, tone)
            
            # Generate hashtags and mentions
            hashtags = await self._generate_text_hashtags(topic, platform)
            
            return {
                "platform": platform,
                "content": content,
                "hashtags": hashtags,
                "character_count": len(content),
                "optimized": len(content) <= self.platform_specs.get(platform, {}).get("max_length", 280)
            }
            
        except Exception as e:
            self.logger.error(f"Platform text generation failed for {platform}: {e}")
            return None
    
    async def _generate_platform_carousel(self, platform: str, topic: str,
                                       key_points: List[str], user_profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate platform-specific carousel content."""
        try:
            carousel = {
                "platform": platform,
                "topic": topic,
                "slides": [],
                "total_slides": len(key_points) + 2,  # +2 for title and conclusion
                "style": "educational",
                "call_to_action": await self._generate_carousel_cta(topic, platform)
            }
            
            # Generate slides
            slides = await self._generate_carousel_slides(topic, key_points, user_profile, platform)
            carousel["slides"] = slides
            
            return carousel
            
        except Exception as e:
            self.logger.error(f"Carousel generation failed for {platform}: {e}")
            return None
    
    async def _generate_audio_format(self, format_type: str, topic: str, script: str,
                                  duration: int, voice_style: str) -> Optional[AudioContent]:
        """Generate format-specific audio content."""
        try:
            audio_content = AudioContent(
                title=await self._generate_audio_title(topic, format_type),
                script=await self._adapt_script_for_audio(script, format_type),
                duration=duration,
                voice_style=voice_style,
                platform=format_type
            )
            
            return audio_content
            
        except Exception as e:
            self.logger.error(f"Audio format generation failed for {format_type}: {e}")
            return None
    
    async def _generate_video_script(self, topic: str, insights: str, 
                                   user_profile: Dict[str, Any], style: str, duration: int) -> str:
        """Generate video script using AI."""
        # This would integrate with AI text generation
        script_template = """
        Topic: {topic}
        Style: {style}
        Duration: {duration}s
        
        Opening Hook (5-10 seconds):
        {hook}
        
        Main Content ({main_duration} seconds):
        {main_content}
        
        Key Insights:
        {insights}
        
        Closing (5-10 seconds):
        {closing}
        
        Call to Action:
        {cta}
        """
        
        # Calculate timing
        main_duration = duration - 20  # Subtract hook and closing
        
        return script_template.format(
            topic=topic,
            style=style,
            duration=duration,
            hook=await self._generate_video_hook(topic, style),
            main_duration=main_duration,
            main_content=await self._generate_main_content(topic, insights, main_duration),
            insights=insights,
            closing=await self._generate_video_closing(topic, style),
            cta=await self._generate_video_cta(topic, user_profile)
        )
    
    async def _generate_video_title(self, topic: str, platform: str, style: str) -> str:
        """Generate engaging video title."""
        title_templates = {
            "educational": [
                "How to Ace: {topic}",
                "Master {topic} in Minutes",
                "Expert Tips for {topic}"
            ],
            "entertaining": [
                "This {topic} Trick Will Blow Your Mind",
                "You Won't Believe This {topic} Secret",
                "{topic} Made Easy and Fun"
            ],
            "inspirational": [
                "Transform Your Approach to {topic}",
                "The {topic} Breakthrough You Need",
                "Unlock Your Potential with {topic}"
            ]
        }
        
        templates = title_templates.get(style, title_templates["educational"])
        selected_template = np.random.choice(templates)
        
        return selected_template.format(topic=topic)
    
    async def _generate_video_description(self, script: str, platform: str) -> str:
        """Generate video description."""
        # Extract key points from script for description
        lines = script.split('\n')
        key_lines = [line for line in lines if line.strip() and not line.startswith(('Topic:', 'Style:', 'Duration:'))]
        
        description = " | ".join(key_lines[:3])  # First 3 key lines
        
        # Add platform-specific formatting
        if platform == "youtube_shorts":
            description += "\n\n#Shorts #CareerTips #InterviewPrep"
        
        return description[:500]  # Limit length
    
    async def _generate_video_hashtags(self, topic: str, platform: str, style: str) -> List[str]:
        """Generate relevant hashtags for video content."""
        base_hashtags = ["#career", "#interview", "#jobtips", "#careeradvice"]
        
        # Topic-specific hashtags
        topic_hashtags = [
            f"#{topic.replace(' ', '')}",
            f"#{topic.lower().replace(' ', '')}tips",
            "#careergrowth"
        ]
        
        # Platform-specific hashtags
        platform_hashtags = {
            "tiktok": ["#tiktokcareer", "#learnontiktok", "#careertok"],
            "instagram_reels": ["#reels", "#instagramreels", "#careerreels"],
            "youtube_shorts": ["#shorts", "#youtubeshorts", "#careershorts"]
        }
        
        # Style-specific hashtags
        style_hashtags = {
            "educational": ["#edutok", "#learnwithme", "#education"],
            "entertaining": ["#funlearning", "#entertaining", "#viral"],
            "inspirational": ["#motivation", "#inspiration", "#success"]
        }
        
        all_hashtags = base_hashtags + topic_hashtags
        all_hashtags.extend(platform_hashtags.get(platform, []))
        all_hashtags.extend(style_hashtags.get(style, []))
        
        return list(set(all_hashtags))[:20]  # Limit to 20 hashtags
    
    def _load_content_templates(self) -> Dict[str, Any]:
        """Load content creation templates."""
        return {
            "video_scripts": {
                "educational": "template_educational_video",
                "entertaining": "template_entertaining_video", 
                "inspirational": "template_inspirational_video"
            },
            "text_posts": {
                "linkedin": "template_linkedin_post",
                "twitter": "template_twitter_post",
                "instagram": "template_instagram_post"
            },
            "carousels": {
                "instagram": "template_instagram_carousel",
                "linkedin": "template_linkedin_carousel"
            }
        }
    
    def _load_platform_specifications(self) -> Dict[str, Any]:
        """Load platform-specific content specifications."""
        return {
            "tiktok": {
                "max_duration": 180,
                "aspect_ratio": "9:16",
                "max_hashtags": 20,
                "optimal_duration": 60
            },
            "instagram_reels": {
                "max_duration": 90,
                "aspect_ratio": "9:16", 
                "max_hashtags": 30,
                "optimal_duration": 45
            },
            "youtube_shorts": {
                "max_duration": 60,
                "aspect_ratio": "9:16",
                "max_hashtags": 10,
                "optimal_duration": 30
            },
            "linkedin": {
                "max_length": 3000,
                "optimal_length": 1500,
                "hashtags_recommended": 5
            },
            "x": {
                "max_length": 280,
                "optimal_length": 240,
                "hashtags_recommended": 3
            }
        }
    
    def _load_brand_guidelines(self) -> Dict[str, Any]:
        """Load brand guidelines for content creation."""
        return {
            "primary_colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
            "secondary_colors": ["#96CEB4", "#FFEAA7", "#DDA0DD"],
            "fonts": ["Montserrat", "Open Sans", "Roboto"],
            "logo_usage": {
                "size": "appropriate",
                "position": "consistent",
                "spacing": "adequate"
            },
            "tone_of_voice": {
                "professional": 0.7,
                "friendly": 0.8,
                "authoritative": 0.6,
                "inspirational": 0.9
            }
        }
    
    def _get_text_platforms(self, platform: str) -> List[str]:
        """Get list of text platforms to target."""
        if platform == "all":
            return ["linkedin", "x", "instagram", "facebook"]
        elif platform == "professional":
            return ["linkedin", "x"]
        elif platform == "social":
            return ["instagram", "facebook", "x"]
        else:
            return [platform]
    
    # Additional helper methods would be implemented here...
    async def _generate_video_hook(self, topic: str, style: str) -> str:
        """Generate engaging video hook."""
        hooks = {
            "educational": f"Struggling with {topic}? Here's the secret most people don't know...",
            "entertaining": f"You won't believe what happened when I tried {topic} this way!",
            "inspirational": f"This one change to your {topic} approach will transform your results!"
        }
        return hooks.get(style, hooks["educational"])
    
    async def _generate_main_content(self, topic: str, insights: str, duration: int) -> str:
        """Generate main video content."""
        return f"In this {duration} second segment, we'll break down {topic} with these key insights: {insights}"
    
    async def _generate_video_closing(self, topic: str, style: str) -> str:
        """Generate video closing."""
        closings = {
            "educational": f"Now you have the tools to master {topic}. Practice these techniques!",
            "entertaining": f"And that's how you make {topic} actually enjoyable!",
            "inspirational": f"Remember, mastering {topic} is a journey. You've got this!"
        }
        return closings.get(style, closings["educational"])
    
    async def _generate_video_cta(self, topic: str, user_profile: Dict[str, Any]) -> str:
        """Generate call to action."""
        return f"Ready to ace your next interview? Download our app for more {topic} tips!"
    
    async def _generate_platform_specific_text(self, platform: str, topic: str, insights: str, tone: str) -> str:
        """Generate platform-specific text content."""
        templates = {
            "linkedin": f"Professional insight on {topic}: {insights}. #CareerGrowth #ProfessionalDevelopment",
            "x": f"Quick tip for {topic}: {insights[:100]}... #CareerTips #JobSearch",
            "instagram": f"🌟 Game-changing insight for {topic}! 👇\n\n{insights}\n\nDouble tap if this helps! ❤️",
            "facebook": f"Interesting perspective on {topic}:\n\n{insights}\n\nWhat are your thoughts? Share below! 👇"
        }
        return templates.get(platform, f"Content about {topic}: {insights}")
    
    async def _generate_text_hashtags(self, topic: str, platform: str) -> List[str]:
        """Generate hashtags for text content."""
        base_hashtags = ["#career", "#interview", "#success"]
        platform_hashtags = {
            "linkedin": ["#linkedin", "#professional", "#business"],
            "x": ["#twitter", "#tips", "#advice"],
            "instagram": ["#instagram", "#instadaily", "#motivation"],
            "facebook": ["#facebook", "#community", "#discussion"]
        }
        return base_hashtags + platform_hashtags.get(platform, [])
    
    async def _generate_carousel_cta(self, topic: str, platform: str) -> str:
        """Generate carousel call to action."""
        ctas = {
            "instagram": "Swipe through to learn more! 👉",
            "linkedin": "Click through for the complete breakdown 📊"
        }
        return ctas.get(platform, "Learn more in the slides above!")
    
    async def _generate_carousel_slides(self, topic: str, key_points: List[str], 
                                      user_profile: Dict[str, Any], platform: str) -> List[Dict[str, str]]:
        """Generate carousel slides."""
        slides = []
        
        # Title slide
        slides.append({
            "title": f"Mastering {topic}",
            "content": "A comprehensive guide based on expert insights",
            "image_suggestion": "title_background"
        })
        
        # Key points slides
        for i, point in enumerate(key_points, 1):
            slides.append({
                "title": f"Key Insight #{i}",
                "content": point,
                "image_suggestion": f"insight_{i}"
            })
        
        # Conclusion slide
        slides.append({
            "title": "Ready to Implement?",
            "content": "Apply these insights in your next interview!",
            "image_suggestion": "conclusion"
        })
        
        return slides
    
    async def _generate_audio_title(self, topic: str, format_type: str) -> str:
        """Generate audio content title."""
        titles = {
            "podcast_snippet": f"Podcast Clip: Expert Tips on {topic}",
            "voiceover": f"Audio Guide: Mastering {topic}",
            "social_audio": f"Quick Audio Tip: {topic} Secrets"
        }
        return titles.get(format_type, f"Audio Content: {topic}")
    
    async def _adapt_script_for_audio(self, script: str, format_type: str) -> str:
        """Adapt script for audio format."""
        # Remove visual references and make it audio-friendly
        audio_script = script.replace("visual elements", "key points")
        audio_script = audio_script.replace("on screen", "in this audio")
        
        if format_type == "social_audio":
            # Make it more concise
            lines = audio_script.split('\n')
            key_lines = [line for line in lines if any(keyword in line.lower() for keyword in 
                                                     ['hook', 'main', 'insight', 'closing'])]
            audio_script = '\n'.join(key_lines)
        
        return audio_script
    
    async def _generate_summary_script(self, summary_data: Dict[str, Any], 
                                     user_profile: Dict[str, Any], interview_type: str) -> str:
        """Generate summary video script."""
        return f"""
        Interview Performance Summary for {interview_type.title()} Role
        
        Overall Score: {summary_data.get('overall_score', 0):.1%}
        Questions Answered: {summary_data.get('total_questions', 0)}
        
        Key Strengths: {len(summary_data.get('strengths', []))} areas
        Improvement Opportunities: {len(summary_data.get('weaknesses', []))} areas
        
        Based on your performance, we've identified specific areas for growth and content creation opportunities.
        """
    
    def _generate_summary_hashtags(self, interview_type: str, summary_data: Dict[str, Any]) -> List[str]:
        """Generate hashtags for summary content."""
        base_hashtags = [
            "#InterviewSummary", 
            "#CareerProgress", 
            "#PerformanceReview",
            "#JobSearchTips"
        ]
        
        type_hashtags = {
            "technical": ["#TechnicalInterview", "#Coding", "#TechCareers"],
            "behavioral": ["#BehavioralInterview", "#SoftSkills", "#CareerGrowth"],
            "leadership": ["#Leadership", "#Management", "#ExecutivePresence"]
        }
        
        return base_hashtags + type_hashtags.get(interview_type, [])
    
    async def _generate_infographic_caption(self, summary_data: Dict[str, Any]) -> str:
        """Generate infographic caption."""
        return f"""
        📊 Interview Performance Breakdown
        
        Overall Score: {summary_data.get('overall_score', 0):.1%}
        Strengths: {len(summary_data.get('strengths', []))}
        Areas for Improvement: {len(summary_data.get('weaknesses', []))}
        
        Visualizing your interview journey and growth opportunities!
        """
    
    async def _generate_weekly_schedule(self, content_assets: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Generate weekly content schedule."""
        schedule = {}
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        
        for day in days:
            schedule[day] = {
                "platforms": {},
                "total_posts": np.random.randint(2, 5),
                "optimal_times": ["09:00", "12:00", "17:00", "20:00"]
            }
            
            for platform in self.platform_specs.keys():
                schedule[day]["platforms"][platform] = {
                    "posts": np.random.randint(1, 3),
                    "content_types": np.random.choice(["video", "image", "text"], 2).tolist()
                }
        
        return schedule
    
    def _analyze_content_mix(self, content_assets: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze content mix across platforms and formats."""
        total_assets = sum(len(assets) for assets in content_assets.values())
        
        if total_assets == 0:
            return {"error": "No content assets available"}
        
        mix = {
            "total_assets": total_assets,
            "by_platform": {},
            "by_format": {
                "video": len(content_assets.get("videos", [])),
                "image": len(content_assets.get("images", [])),
                "text": len(content_assets.get("text_posts", [])),
                "audio": len(content_assets.get("audio_clips", [])),
                "carousel": len(content_assets.get("carousels", []))
            },
            "balance_score": 0.0
        }
        
        # Calculate platform distribution
        for platform in self.platform_specs.keys():
            platform_count = sum(
                1 for asset_list in content_assets.values() 
                for asset in asset_list 
                if getattr(asset, 'platform', None) == platform
            )
            mix["by_platform"][platform] = platform_count
        
        # Calculate balance score (0-1, higher is more balanced)
        format_counts = list(mix["by_format"].values())
        if format_counts:
            mix["balance_score"] = 1 - (np.std(format_counts) / np.mean(format_counts) if np.mean(format_counts) > 0 else 1)
        
        return mix
    
    async def _generate_optimization_suggestions(self, content_assets: Dict[str, List[Any]]) -> List[str]:
        """Generate content optimization suggestions."""
        suggestions = []
        
        mix = self._analyze_content_mix(content_assets)
        
        # Check content balance
        if mix["balance_score"] < 0.7:
            suggestions.append("Diversify your content formats for better audience engagement")
        
        # Platform-specific suggestions
        for platform, count in mix["by_platform"].items():
            if count == 0:
                suggestions.append(f"Consider creating content for {platform} to reach a wider audience")
        
        # Timing suggestions
        suggestions.extend([
            "Post video content in the evening for higher engagement",
            "Share educational content on LinkedIn during work hours",
            "Use Instagram Stories for behind-the-scenes content"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions

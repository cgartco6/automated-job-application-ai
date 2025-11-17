import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from typing import Dict, List, Any, Tuple
import numpy as np
import logging

class ResumeAnalyzer(nn.Module):
    """
    Deep learning model for comprehensive resume analysis and optimization.
    Uses BERT-based architecture with multi-task learning.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_skills: int = 500):
        super(ResumeAnalyzer, self).__init__()
        
        # BERT backbone for text understanding
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Feature dimensions
        self.hidden_size = self.bert.config.hidden_size
        self.num_skills = num_skills
        
        # Multi-head attention for different resume sections
        self.section_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Task-specific heads
        self.skill_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_skills),
            nn.Sigmoid()
        )
        
        self.experience_analyzer = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5),  # 5 experience levels
            nn.Softmax(dim=1)
        )
        
        self.achievement_detector = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.optimization_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Section embeddings for different resume parts
        self.section_embeddings = nn.Embedding(10, self.hidden_size)  # 10 sections
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                section_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            section_ids: Section identifiers for different resume parts
            
        Returns:
            Dictionary of model outputs
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Add section embeddings if provided
        if section_ids is not None:
            section_embeds = self.section_embeddings(section_ids)
            sequence_output = sequence_output + section_embeds.unsqueeze(1)
        
        # Apply layer normalization
        sequence_output = self.layer_norm(sequence_output)
        
        # Apply section-aware attention
        attended_output, attention_weights = self.section_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Global average pooling
        if attention_mask is not None:
            # Masked mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(attended_output.size()).float()
            sum_embeddings = torch.sum(attended_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = attended_output.mean(dim=1)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Generate predictions for different tasks
        skills_pred = self.skill_classifier(pooled_output)
        experience_pred = self.experience_analyzer(pooled_output)
        achievement_score = self.achievement_detector(pooled_output)
        optimization_score = self.optimization_scorer(pooled_output)
        
        return {
            'skills': skills_pred,
            'experience_level': experience_pred,
            'achievement_score': achievement_score,
            'optimization_score': optimization_score,
            'attention_weights': attention_weights
        }
    
    async def analyze_resume(self, resume_text: str, sections: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Analyze a resume and provide comprehensive feedback.
        
        Args:
            resume_text: The resume content
            sections: Dictionary of resume sections
            
        Returns:
            Analysis results with scores and suggestions
        """
        try:
            # Preprocess and tokenize
            inputs = self._preprocess_resume(resume_text, sections)
            
            with torch.no_grad():
                outputs = self.forward(**inputs)
            
            # Process outputs
            analysis = self._process_outputs(outputs, resume_text)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Resume analysis failed: {e}")
            raise
    
    def _preprocess_resume(self, resume_text: str, sections: Dict[str, str] = None) -> Dict[str, torch.Tensor]:
        """Preprocess resume text for model input."""
        # Tokenize the text
        encoding = self.tokenizer(
            resume_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        inputs = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
        
        # Add section information if available
        if sections:
            section_ids = self._create_section_ids(resume_text, sections)
            inputs['section_ids'] = section_ids
        
        return inputs
    
    def _create_section_ids(self, resume_text: str, sections: Dict[str, str]) -> torch.Tensor:
        """Create section ID tensor for different resume sections."""
        # This is a simplified implementation
        # In practice, you'd map text positions to section IDs
        section_mapping = {
            'summary': 0,
            'experience': 1,
            'education': 2,
            'skills': 3,
            'projects': 4,
            'certifications': 5
        }
        
        # Create a tensor with section IDs (simplified)
        # Actual implementation would require more sophisticated text segmentation
        num_tokens = len(self.tokenizer.tokenize(resume_text))
        section_ids = torch.zeros(num_tokens, dtype=torch.long)
        
        return section_ids.unsqueeze(0)  # Add batch dimension
    
    def _process_outputs(self, outputs: Dict[str, torch.Tensor], resume_text: str) -> Dict[str, Any]:
        """Process model outputs into actionable insights."""
        # Extract skills
        skills_tensor = outputs['skills'].squeeze()
        top_skill_indices = torch.topk(skills_tensor, 15).indices.tolist()
        detected_skills = [self.skill_vocab[i] for i in top_skill_indices if i < len(self.skill_vocab)]
        
        # Experience level
        experience_probs = outputs['experience_level'].squeeze()
        experience_level = torch.argmax(experience_probs).item()
        
        # Scores
        achievement_score = outputs['achievement_score'].item()
        optimization_score = outputs['optimization_score'].item()
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            detected_skills, experience_level, achievement_score, optimization_score
        )
        
        return {
            'detected_skills': detected_skills,
            'experience_level': experience_level,
            'achievement_score': achievement_score,
            'optimization_score': optimization_score,
            'suggestions': suggestions,
            'skill_confidence_scores': {
                skill: float(score) for skill, score in zip(detected_skills, skills_tensor[top_skill_indices])
            }
        }
    
    def _generate_suggestions(self, skills: List[str], exp_level: int, 
                            achievement_score: float, optimization_score: float) -> List[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        # Skill-based suggestions
        if len(skills) < 10:
            suggestions.append("Consider adding more technical skills to your resume")
        
        # Achievement-based suggestions
        if achievement_score < 0.6:
            suggestions.append("Include more quantifiable achievements with metrics")
        
        # Optimization suggestions
        if optimization_score < 0.7:
            suggestions.append("Optimize resume structure for better ATS compatibility")
        
        # Experience-level suggestions
        if exp_level < 3:
            suggestions.append("Highlight learning agility and rapid skill acquisition")
        else:
            suggestions.append("Emphasize leadership and strategic impact")
        
        # Keyword optimization
        suggestions.append("Ensure resume includes industry-specific keywords")
        
        return suggestions
    
    def load_skill_vocab(self, vocab_path: str):
        """Load skill vocabulary from file."""
        import json
        with open(vocab_path, 'r') as f:
            self.skill_vocab = json.load(f)
    
    def save_model(self, save_path: str):
        """Save model weights and configuration."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'skill_vocab': getattr(self, 'skill_vocab', []),
            'model_config': {
                'hidden_size': self.hidden_size,
                'num_skills': self.num_skills
            }
        }, save_path)
    
    def load_model(self, load_path: str):
        """Load model weights and configuration."""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.skill_vocab = checkpoint.get('skill_vocab', [])

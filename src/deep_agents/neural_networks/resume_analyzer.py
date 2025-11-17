import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Any

class ResumeAnalyzer(nn.Module):
    """Deep learning model for resume analysis and optimization"""
    
    def __init__(self, bert_model_name: str = 'bert-base-uncased', hidden_dim: int = 768):
        super(ResumeAnalyzer, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Multi-head attention for different resume sections
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Classification heads for different aspects
        self.skill_extractor = nn.Linear(hidden_dim, 100)  # 100 common skills
        self.experience_analyzer = nn.Linear(hidden_dim, 5)  # Experience levels
        self.optimization_scorer = nn.Linear(hidden_dim, 1)  # Optimization score
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply attention
        attended_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
        
        # Global average pooling
        pooled_output = attended_output.mean(dim=1)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Generate predictions for different aspects
        skills_logits = self.skill_extractor(pooled_output)
        experience_logits = self.experience_analyzer(pooled_output)
        optimization_score = self.optimization_scorer(pooled_output)
        
        return {
            'skills': skills_logits,
            'experience_level': experience_logits,
            'optimization_score': optimization_score
        }
    
    def analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """Analyze resume and provide optimization suggestions"""
        inputs = self.tokenizer(
            resume_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.forward(inputs['input_ids'], inputs['attention_mask'])
        
        # Process outputs
        skills = torch.sigmoid(outputs['skills']).squeeze()
        top_skills = self._get_top_skills(skills)
        
        experience_level = torch.argmax(outputs['experience_level']).item()
        optimization_score = torch.sigmoid(outputs['optimization_score']).item()
        
        return {
            'detected_skills': top_skills,
            'experience_level': experience_level,
            'optimization_score': optimization_score,
            'suggestions': self._generate_suggestions(top_skills, experience_level, optimization_score)
        }
    
    def _get_top_skills(self, skills_tensor: torch.Tensor, top_k: int = 10) -> List[str]:
        """Extract top predicted skills"""
        skill_names = [
            'Python', 'JavaScript', 'Java', 'C++', 'SQL', 'AWS', 'Docker', 'Kubernetes',
            'Machine Learning', 'Deep Learning', 'Data Analysis', 'Project Management',
            # ... more skills
        ]
        
        top_indices = torch.topk(skills_tensor, top_k).indices.tolist()
        return [skill_names[i] for i in top_indices if i < len(skill_names)]
    
    def _generate_suggestions(self, skills: List[str], exp_level: int, score: float) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        if score < 0.7:
            suggestions.append("Consider adding more quantifiable achievements")
        
        if len(skills) < 8:
            suggestions.append("Include more relevant technical skills")
        
        if exp_level < 3:
            suggestions.append("Highlight leadership and project experience")
        
        return suggestions

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Any, Tuple
import numpy as np
import logging

class JobMatcher(nn.Module):
    """
    Deep learning model for matching resumes to job descriptions.
    Uses siamese network architecture with contrastive learning.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", embedding_dim: int = 768):
        super(JobMatcher, self).__init__()
        
        # Shared BERT encoder for both resumes and job descriptions
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Projection layers
        self.resume_projection = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        self.job_projection = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Similarity head
        self.similarity_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Feature cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, resume_inputs: Dict[str, torch.Tensor], 
                job_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for resume-job matching.
        
        Args:
            resume_inputs: Tokenized resume inputs
            job_inputs: Tokenized job description inputs
            
        Returns:
            Similarity scores and feature representations
        """
        # Encode resume
        resume_embeddings = self._encode_text(
            resume_inputs['input_ids'],
            resume_inputs['attention_mask']
        )
        resume_features = self.resume_projection(resume_embeddings)
        
        # Encode job description
        job_embeddings = self._encode_text(
            job_inputs['input_ids'],
            job_inputs['attention_mask']
        )
        job_features = self.job_projection(job_embeddings)
        
        # Apply cross-attention
        attended_resume, _ = self.cross_attention(
            resume_features.unsqueeze(1),
            job_features.unsqueeze(1),
            job_features.unsqueeze(1)
        )
        
        attended_job, _ = self.cross_attention(
            job_features.unsqueeze(1),
            resume_features.unsqueeze(1),
            resume_features.unsqueeze(1)
        )
        
        # Concatenate features for similarity calculation
        combined_features = torch.cat([
            attended_resume.squeeze(1),
            attended_job.squeeze(1)
        ], dim=1)
        
        # Calculate similarity score
        similarity_score = self.similarity_head(combined_features)
        
        return {
            'similarity_score': similarity_score,
            'resume_features': resume_features,
            'job_features': job_features
        }
    
    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text using BERT and return pooled output."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use mean pooling of all token embeddings
        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        return pooled_output
    
    async def calculate_match_score(self, resume_text: str, job_description: str) -> float:
        """Calculate match score between resume and job description."""
        try:
            # Tokenize inputs
            resume_inputs = self.tokenizer(
                resume_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            job_inputs = self.tokenizer(
                job_description,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.forward(resume_inputs, job_inputs)
            
            return outputs['similarity_score'].item()
            
        except Exception as e:
            self.logger.error(f"Match score calculation failed: {e}")
            return 0.0
    
    async def find_best_matches(self, resume_text: str, job_descriptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find best matching jobs for a given resume."""
        matches = []
        
        for job in job_descriptions:
            try:
                score = await self.calculate_match_score(resume_text, job['description'])
                
                matches.append({
                    'job_id': job['id'],
                    'title': job['title'],
                    'company': job['company'],
                    'match_score': score,
                    'explanation': await self._generate_match_explanation(resume_text, job['description'], score)
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to match job {job.get('id', 'unknown')}: {e}")
                continue
        
        # Sort by match score descending
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return matches
    
    async def _generate_match_explanation(self, resume_text: str, job_description: str, score: float) -> str:
        """Generate explanation for the match score."""
        if score >= 0.8:
            return "Excellent match: Strong alignment between your skills and job requirements"
        elif score >= 0.6:
            return "Good match: Significant overlap in required and demonstrated skills"
        elif score >= 0.4:
            return "Moderate match: Some relevant skills but may need additional qualifications"
        else:
            return "Weak match: Limited alignment between your profile and job requirements"

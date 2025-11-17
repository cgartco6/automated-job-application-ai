import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import logging

class SemanticMatcher:
    """
    Semantic matching model for understanding text similarity and relationships.
    Uses sentence transformers for semantic understanding.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)
    
    async def find_similar_roles(self, user_profile: Dict[str, Any]) -> List[str]:
        """
        Find similar job roles based on semantic understanding of user profile.
        
        Args:
            user_profile: User's skills, experience, and preferences
            
        Returns:
            List of similar job roles
        """
        skills = user_profile.get('skills', [])
        experience = user_profile.get('experience', {})
        
        # Create semantic query
        query_text = self._create_semantic_query(skills, experience)
        
        # Get query embedding
        query_embedding = self.model.encode([query_text])
        
        # Compare with role embeddings
        similar_roles = []
        for role, role_data in self.role_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, role_data['embedding'])
            
            if similarity > 0.7:  # Threshold for similarity
                similar_roles.append({
                    'role': role,
                    'similarity': similarity,
                    'description': role_data['description']
                })
        
        # Sort by similarity
        similar_roles.sort(key=lambda x: x['similarity'], reverse=True)
        
        return [role['role'] for role in similar_roles[:5]]
    
    async def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self.model.encode([text1, text2])
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    async def extract_key_concepts(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key concepts from text using semantic analysis.
        
        Args:
            text: Input text
            top_k: Number of concepts to extract
            
        Returns:
            List of key concepts with scores
        """
        # Implementation for concept extraction
        # This would use techniques like keyphrase extraction and semantic clustering
        
        concepts = []
        sentences = text.split('.')
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                embedding = self.model.encode([sentence])
                # Simple implementation - in practice, use more sophisticated clustering
                concepts.append({
                    'concept': sentence.strip(),
                    'importance': len(sentence) / 100.0,  # Simplified importance
                    'embedding': embedding
                })
        
        # Sort by importance and return top_k
        concepts.sort(key=lambda x: x['importance'], reverse=True)
        return concepts[:top_k]
    
    def _create_semantic_query(self, skills: List[str], experience: Dict[str, Any]) -> str:
        """Create semantic query from user profile."""
        query_parts = []
        
        # Add skills
        query_parts.extend(skills)
        
        # Add experience information
        if 'title' in experience:
            query_parts.append(experience['title'])
        if 'industry' in experience:
            query_parts.append(experience['industry'])
        
        return " ".join(query_parts)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def load_role_embeddings(self, embeddings_path: str):
        """Load pre-computed role embeddings."""
        import pickle
        with open(embeddings_path, 'rb') as f:
            self.role_embeddings = pickle.load(f)

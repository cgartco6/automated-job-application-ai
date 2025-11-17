import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from collections import deque
import random

class AgentOptimizer:
    """
    Reinforcement Learning-based optimizer for AI agents.
    Uses Proximal Policy Optimization (PPO) to improve agent performance.
    """
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Policy network
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # PPO parameters
        self.ppo_epochs = 10
        self.clip_param = 0.2
        self.entropy_coef = 0.01
        
        self.logger = logging.getLogger(__name__)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """
        Select action based on current policy.
        
        Args:
            state: Current state representation
            training: Whether in training mode
            
        Returns:
            Selected action and log probability
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            
            if training:
                action = action_dist.sample()
            else:
                action = torch.argmax(action_probs)
            
            log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def update_policy(self, states: List[np.ndarray], actions: List[int], 
                     log_probs: List[float], rewards: List[float],
                     next_states: List[np.ndarray], dones: List[bool]):
        """
        Update policy using PPO.
        
        Args:
            states: List of states
            actions: List of actions taken
            log_probs: List of log probabilities
            rewards: List of rewards received
            next_states: List of next states
            dones: List of done flags
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.value_net(states)
            next_values = self.value_net(next_states)
            targets = rewards + (1 - dones.float()) * 0.99 * next_values
            advantages = targets - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Calculate new action probabilities
            action_probs = self.policy_net(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # Value loss
            value_loss = nn.MSELoss()(self.value_net(states), targets)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
    
    def optimize_agent_strategy(self, agent_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize agent strategy based on performance metrics.
        
        Args:
            agent_performance: Performance metrics of the agent
            
        Returns:
            Optimization results and new strategy
        """
        # Extract features for RL state
        state = self._extract_state_features(agent_performance)
        
        # Select optimization action
        action, log_prob = self.select_action(state)
        
        # Apply optimization
        optimization_result = self._apply_optimization(action, agent_performance)
        
        # Store experience
        reward = self._calculate_reward(optimization_result)
        next_state = self._extract_state_features(optimization_result)
        
        self.memory.append((state, action, log_prob, reward, next_state, False))
        
        # Update policy if enough experiences
        if len(self.memory) >= self.batch_size:
            self._update_from_memory()
        
        return optimization_result
    
    def _extract_state_features(self, performance: Dict[str, Any]) -> np.ndarray:
        """Extract state features from performance metrics."""
        features = [
            performance.get('success_rate', 0),
            performance.get('average_processing_time', 0),
            performance.get('resource_usage', 0),
            performance.get('user_satisfaction', 0),
            performance.get('error_rate', 0)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _apply_optimization(self, action: int, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization action to agent strategy."""
        optimizations = {
            0: self._optimize_processing_pipeline,
            1: self._optimize_resource_allocation,
            2: self._optimize_decision_thresholds,
            3: self._optimize_feature_selection,
            4: self._optimize_ensemble_weights
        }
        
        optimization_func = optimizations.get(action, self._optimize_processing_pipeline)
        return optimization_func(performance)
    
    def _calculate_reward(self, optimization_result: Dict[str, Any]) -> float:
        """Calculate reward for optimization action."""
        reward = 0.0
        
        # Positive rewards for improvements
        if optimization_result.get('success_rate_improvement', 0) > 0:
            reward += 1.0
        
        if optimization_result.get('processing_time_reduction', 0) > 0:
            reward += 0.5
        
        if optimization_result.get('resource_usage_reduction', 0) > 0:
            reward += 0.3
        
        # Negative rewards for degradations
        if optimization_result.get('success_rate_improvement', 0) < 0:
            reward -= 1.0
        
        if optimization_result.get('error_rate_increase', 0) > 0:
            reward -= 0.5
        
        return reward
    
    def _update_from_memory(self):
        """Update policy from experience replay memory."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, log_probs, rewards, next_states, dones = zip(*batch)
        
        self.update_policy(
            list(states), list(actions), list(log_probs),
            list(rewards), list(next_states), list(dones)
        )
    
    def _optimize_processing_pipeline(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize agent processing pipeline."""
        # Implementation for pipeline optimization
        return {
            'optimization_type': 'pipeline',
            'success_rate_improvement': 0.02,
            'processing_time_reduction': 0.1,
            'changes_applied': ['parallel_processing', 'caching_strategy']
        }
    
    def _optimize_resource_allocation(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation strategy."""
        # Implementation for resource optimization
        return {
            'optimization_type': 'resource_allocation',
            'resource_usage_reduction': 0.15,
            'success_rate_improvement': 0.01,
            'changes_applied': ['dynamic_batching', 'memory_optimization']
        }
    
    def _optimize_decision_thresholds(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize decision thresholds and confidence levels."""
        # Implementation for threshold optimization
        return {
            'optimization_type': 'decision_thresholds',
            'success_rate_improvement': 0.03,
            'error_rate_reduction': 0.05,
            'changes_applied': ['adaptive_thresholds', 'confidence_calibration']
        }
    
    def _optimize_feature_selection(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize feature selection for decision making."""
        # Implementation for feature optimization
        return {
            'optimization_type': 'feature_selection',
            'processing_time_reduction': 0.2,
            'success_rate_improvement': 0.01,
            'changes_applied': ['feature_importance', 'dimensionality_reduction']
        }
    
    def _optimize_ensemble_weights(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize ensemble model weights."""
        # Implementation for ensemble optimization
        return {
            'optimization_type': 'ensemble_weights',
            'success_rate_improvement': 0.04,
            'error_rate_reduction': 0.03,
            'changes_applied': ['weight_optimization', 'model_fusion']
        }

class PolicyNetwork(nn.Module):
    """Policy network for reinforcement learning."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ValueNetwork(nn.Module):
    """Value network for reinforcement learning."""
    
    def __init__(self, state_dim: int):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

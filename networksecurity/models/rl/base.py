"""
RL代理基类
定义强化学习代理的抽象接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Type
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """RL配置"""
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 10


@dataclass
class TrainingStats:
    """训练统计"""
    episode: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    epsilon: float = 1.0
    loss: float = 0.0
    steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode': self.episode,
            'total_reward': self.total_reward,
            'avg_reward': self.avg_reward,
            'epsilon': self.epsilon,
            'loss': self.loss,
            'steps': self.steps
        }


class RLAgentBase(ABC):
    """RL代理基类"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[RLConfig] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or RLConfig()
        self.is_trained = False
        self.training_history: List[TrainingStats] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """选择动作"""
        pass
    
    @abstractmethod
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> float:
        """学习更新"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> bool:
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        pass
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.config.epsilon > self.config.epsilon_min:
            self.config.epsilon *= self.config.epsilon_decay


class RLAgentRegistry:
    """RL代理注册表"""
    
    _agents: Dict[str, Type[RLAgentBase]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(agent_class: Type[RLAgentBase]):
            cls._agents[name] = agent_class
            return agent_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[RLAgentBase]]:
        return cls._agents.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[RLAgentBase]:
        agent_class = cls.get(name)
        return agent_class(**kwargs) if agent_class else None
    
    @classmethod
    def list_agents(cls) -> List[str]:
        return list(cls._agents.keys())

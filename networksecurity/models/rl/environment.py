"""
安全环境
定义强化学习的网络安全环境
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import IntEnum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """可执行的动作"""
    ALLOW = 0       # 放行
    BLOCK = 1       # 阻断
    CHALLENGE = 2   # 人机验证
    LOG = 3         # 仅记录
    ALERT = 4       # 告警


@dataclass
class State:
    """环境状态"""
    features: np.ndarray
    risk_score: float = 0.0
    threat_type: int = 0
    source_reputation: float = 0.5
    
    def to_array(self) -> np.ndarray:
        base = self.features.flatten() if isinstance(self.features, np.ndarray) else np.array(self.features)
        extra = np.array([self.risk_score, self.threat_type, self.source_reputation])
        return np.concatenate([base, extra])


class SecurityEnvironment:
    """网络安全强化学习环境"""
    
    def __init__(self, feature_dim: int = 30):
        self.feature_dim = feature_dim
        self.state_dim = feature_dim + 3  # features + risk_score + threat_type + reputation
        self.action_dim = len(Action)
        self.current_state: Optional[State] = None
        self.episode_reward = 0.0
        self.steps = 0
        
        # 奖励配置
        self.rewards = {
            'correct_block': 10.0,      # 正确阻断威胁
            'correct_allow': 5.0,       # 正确放行正常流量
            'false_positive': -8.0,     # 误报（阻断正常流量）
            'false_negative': -15.0,    # 漏报（放行威胁）
            'challenge_threat': 3.0,    # 对威胁进行验证
            'challenge_normal': -2.0,   # 对正常流量验证（用户体验差）
        }
    
    def reset(self, features: Optional[np.ndarray] = None, 
              is_threat: bool = False) -> np.ndarray:
        """重置环境"""
        if features is None:
            features = np.random.randn(self.feature_dim)
        
        risk_score = np.random.uniform(0.7, 1.0) if is_threat else np.random.uniform(0, 0.3)
        threat_type = np.random.randint(1, 5) if is_threat else 0
        reputation = np.random.uniform(0, 0.3) if is_threat else np.random.uniform(0.7, 1.0)
        
        self.current_state = State(
            features=features,
            risk_score=risk_score,
            threat_type=threat_type,
            source_reputation=reputation
        )
        self.episode_reward = 0.0
        self.steps = 0
        return self.current_state.to_array()
    
    def step(self, action: int, true_label: int = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        if self.current_state is None:
            raise ValueError("请先调用reset()")
        
        self.steps += 1
        is_threat = true_label == 1 if true_label is not None else self.current_state.risk_score > 0.5
        action_enum = Action(action)
        
        # 计算奖励
        reward = self._calculate_reward(action_enum, is_threat)
        self.episode_reward += reward
        
        # 生成下一状态
        done = True  # 单步决策环境
        next_state = self.current_state.to_array()
        
        info = {
            'action': action_enum.name,
            'is_threat': is_threat,
            'reward': reward,
            'episode_reward': self.episode_reward
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action: Action, is_threat: bool) -> float:
        """计算奖励"""
        if action == Action.BLOCK:
            return self.rewards['correct_block'] if is_threat else self.rewards['false_positive']
        elif action == Action.ALLOW:
            return self.rewards['correct_allow'] if not is_threat else self.rewards['false_negative']
        elif action == Action.CHALLENGE:
            return self.rewards['challenge_threat'] if is_threat else self.rewards['challenge_normal']
        elif action == Action.ALERT:
            return self.rewards['correct_block'] * 0.5 if is_threat else self.rewards['false_positive'] * 0.3
        else:  # LOG
            return 0.0
    
    def get_optimal_action(self, risk_score: float) -> Action:
        """获取最优动作（用于评估）"""
        if risk_score > 0.8:
            return Action.BLOCK
        elif risk_score > 0.5:
            return Action.CHALLENGE
        elif risk_score > 0.3:
            return Action.ALERT
        else:
            return Action.ALLOW

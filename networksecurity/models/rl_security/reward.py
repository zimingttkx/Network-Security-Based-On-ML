"""
奖励计算器
为网络安全RL环境提供灵活的奖励计算
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """奖励配置"""
    # 正确决策奖励
    correct_block: float = 10.0
    correct_allow: float = 5.0
    
    # 错误决策惩罚
    false_positive: float = -8.0
    false_negative: float = -15.0
    
    # 中间动作奖励
    alert_threat: float = 3.0
    alert_normal: float = -1.0
    challenge_threat: float = 4.0
    challenge_normal: float = -2.0
    rate_limit_threat: float = 5.0
    rate_limit_normal: float = -3.0
    log_action: float = 0.0
    
    # 额外奖励/惩罚
    consecutive_correct: float = 1.0
    consecutive_wrong: float = -2.0
    
    # 威胁严重性权重
    severity_multiplier: float = 1.5


class RewardCalculator:
    """
    奖励计算器
    
    根据动作、真实标签和上下文计算奖励
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.total_rewards = 0.0
        self.episode_count = 0
    
    def calculate(self, action: int, is_attack: bool, 
                  threat_score: float = 0.5,
                  attack_type: int = 0) -> float:
        """
        计算奖励
        
        Args:
            action: 执行的动作
            is_attack: 是否为攻击
            threat_score: 威胁分数
            attack_type: 攻击类型
        """
        base_reward = self._get_base_reward(action, is_attack)
        
        # 威胁严重性调整
        if is_attack and threat_score > 0.7:
            base_reward *= self.config.severity_multiplier
        
        # 连续正确/错误奖励
        is_correct = self._is_correct_action(action, is_attack)
        if is_correct:
            self.consecutive_correct += 1
            self.consecutive_wrong = 0
            if self.consecutive_correct > 3:
                base_reward += self.config.consecutive_correct
        else:
            self.consecutive_wrong += 1
            self.consecutive_correct = 0
            if self.consecutive_wrong > 3:
                base_reward += self.config.consecutive_wrong
        
        self.total_rewards += base_reward
        self.episode_count += 1
        
        return base_reward
    
    def _get_base_reward(self, action: int, is_attack: bool) -> float:
        """获取基础奖励"""
        # 动作映射: 0=ALLOW, 1=BLOCK, 2=ALERT, 3=LOG, 4=CHALLENGE, 5=RATE_LIMIT, 6=QUARANTINE
        if action == 1 or action == 6:  # BLOCK or QUARANTINE
            return self.config.correct_block if is_attack else self.config.false_positive
        elif action == 0:  # ALLOW
            return self.config.correct_allow if not is_attack else self.config.false_negative
        elif action == 2:  # ALERT
            return self.config.alert_threat if is_attack else self.config.alert_normal
        elif action == 4:  # CHALLENGE
            return self.config.challenge_threat if is_attack else self.config.challenge_normal
        elif action == 5:  # RATE_LIMIT
            return self.config.rate_limit_threat if is_attack else self.config.rate_limit_normal
        else:  # LOG
            return self.config.log_action
    
    def _is_correct_action(self, action: int, is_attack: bool) -> bool:
        """判断动作是否正确"""
        blocking_actions = {1, 6}  # BLOCK, QUARANTINE
        allowing_actions = {0}  # ALLOW
        
        if is_attack:
            return action in blocking_actions or action in {2, 4, 5}  # 阻断或警告
        else:
            return action in allowing_actions or action == 3  # 放行或记录
    
    def reset(self):
        """重置状态"""
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.total_rewards = 0.0
        self.episode_count = 0
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'total_rewards': self.total_rewards,
            'episode_count': self.episode_count,
            'avg_reward': self.total_rewards / max(1, self.episode_count),
            'consecutive_correct': self.consecutive_correct,
            'consecutive_wrong': self.consecutive_wrong
        }

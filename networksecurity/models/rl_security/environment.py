"""
网络安全强化学习环境
基于 gym-optimal-intrusion-response 和 ReinforcementWall 实现

提供Gymnasium兼容的网络安全防御环境
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class SecurityAction(IntEnum):
    """安全动作空间"""
    ALLOW = 0       # 放行流量
    BLOCK = 1       # 阻断流量
    ALERT = 2       # 发出告警
    LOG = 3         # 仅记录
    CHALLENGE = 4   # 人机验证
    RATE_LIMIT = 5  # 限速
    QUARANTINE = 6  # 隔离


@dataclass
class SecurityState:
    """安全状态"""
    # 流量特征 (15维)
    packet_rate: float = 0.0          # 包速率
    byte_rate: float = 0.0            # 字节速率
    unique_src_ips: int = 0           # 唯一源IP数
    unique_dst_ports: int = 0         # 唯一目标端口数
    syn_ratio: float = 0.0            # SYN包比例
    avg_packet_size: float = 0.0      # 平均包大小
    flow_duration: float = 0.0        # 流持续时间
    
    # 威胁指标
    threat_score: float = 0.0         # 威胁分数
    anomaly_score: float = 0.0        # 异常分数
    reputation_score: float = 0.5     # IP信誉分数
    
    # 历史信息
    recent_blocks: int = 0            # 最近阻断数
    recent_alerts: int = 0            # 最近告警数
    false_positives: int = 0          # 误报数
    true_positives: int = 0           # 正确检测数
    attack_type: int = 0              # 攻击类型 (0=正常)
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([
            self.packet_rate / 10000,
            self.byte_rate / 1e8,
            min(self.unique_src_ips / 1000, 1.0),
            min(self.unique_dst_ports / 100, 1.0),
            self.syn_ratio,
            self.avg_packet_size / 1500,
            min(self.flow_duration / 60, 1.0),
            self.threat_score,
            self.anomaly_score,
            self.reputation_score,
            min(self.recent_blocks / 100, 1.0),
            min(self.recent_alerts / 100, 1.0),
            min(self.false_positives / 10, 1.0),
            min(self.true_positives / 100, 1.0),
            self.attack_type / 10
        ], dtype=np.float32)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'SecurityState':
        """从数组创建状态"""
        return cls(
            packet_rate=arr[0] * 10000,
            byte_rate=arr[1] * 1e8,
            unique_src_ips=int(arr[2] * 1000),
            unique_dst_ports=int(arr[3] * 100),
            syn_ratio=arr[4],
            avg_packet_size=arr[5] * 1500,
            flow_duration=arr[6] * 60,
            threat_score=arr[7],
            anomaly_score=arr[8],
            reputation_score=arr[9],
            recent_blocks=int(arr[10] * 100),
            recent_alerts=int(arr[11] * 100),
            false_positives=int(arr[12] * 10),
            true_positives=int(arr[13] * 100),
            attack_type=int(arr[14] * 10)
        )


class NetworkSecurityEnv:
    """
    网络安全强化学习环境
    
    Gymnasium兼容的环境，用于训练网络安全防御代理。
    
    状态空间: 15维连续特征
    动作空间: 7个离散动作
    """
    
    # 攻击类型
    ATTACK_TYPES = {
        0: 'normal',
        1: 'dos',
        2: 'ddos',
        3: 'port_scan',
        4: 'brute_force',
        5: 'sql_injection',
        6: 'xss',
        7: 'botnet',
        8: 'malware',
        9: 'c2'
    }
    
    def __init__(self, episode_length: int = 1000):
        """
        Args:
            episode_length: 每个episode的步数
        """
        self.episode_length = episode_length
        self.state_dim = 15
        self.action_dim = len(SecurityAction)
        
        # 状态
        self.current_state: Optional[SecurityState] = None
        self.step_count = 0
        self.episode_reward = 0.0
        
        # 奖励配置
        self.rewards = {
            'correct_block': 10.0,
            'correct_allow': 5.0,
            'false_positive': -8.0,
            'false_negative': -15.0,
            'alert_threat': 3.0,
            'alert_normal': -1.0,
            'rate_limit_threat': 5.0,
            'rate_limit_normal': -3.0
        }
        
        # 统计
        self.stats = {
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_state = self._generate_state(is_attack=False)
        self.step_count = 0
        self.episode_reward = 0.0
        
        return self.current_state.to_array(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        action_enum = SecurityAction(action)
        
        # 确定真实标签
        is_attack = self.current_state.attack_type > 0
        
        # 计算奖励
        reward = self._calculate_reward(action_enum, is_attack)
        self.episode_reward += reward
        
        # 更新统计
        self._update_stats(action_enum, is_attack)
        
        # 生成下一状态
        attack_prob = 0.3 if self.step_count < self.episode_length // 2 else 0.5
        next_is_attack = np.random.random() < attack_prob
        self.current_state = self._generate_state(next_is_attack)
        
        # 检查终止
        terminated = False
        truncated = self.step_count >= self.episode_length
        
        info = {
            'action': action_enum.name,
            'is_attack': is_attack,
            'reward': reward,
            'stats': self.stats.copy()
        }
        
        return self.current_state.to_array(), reward, terminated, truncated, info
    
    def _generate_state(self, is_attack: bool) -> SecurityState:
        """生成状态"""
        if is_attack:
            attack_type = np.random.randint(1, 10)
            return SecurityState(
                packet_rate=np.random.uniform(5000, 50000),
                byte_rate=np.random.uniform(1e7, 1e9),
                unique_src_ips=np.random.randint(1, 100) if attack_type != 2 else np.random.randint(100, 1000),
                unique_dst_ports=np.random.randint(1, 50) if attack_type != 3 else np.random.randint(50, 500),
                syn_ratio=np.random.uniform(0.5, 0.95) if attack_type in [1, 2] else np.random.uniform(0.1, 0.3),
                avg_packet_size=np.random.uniform(40, 200) if attack_type in [1, 2] else np.random.uniform(200, 1000),
                flow_duration=np.random.uniform(0.1, 10),
                threat_score=np.random.uniform(0.6, 1.0),
                anomaly_score=np.random.uniform(0.5, 1.0),
                reputation_score=np.random.uniform(0.0, 0.4),
                attack_type=attack_type
            )
        else:
            return SecurityState(
                packet_rate=np.random.uniform(100, 2000),
                byte_rate=np.random.uniform(1e5, 1e7),
                unique_src_ips=np.random.randint(1, 20),
                unique_dst_ports=np.random.randint(1, 10),
                syn_ratio=np.random.uniform(0.05, 0.2),
                avg_packet_size=np.random.uniform(500, 1400),
                flow_duration=np.random.uniform(1, 60),
                threat_score=np.random.uniform(0.0, 0.3),
                anomaly_score=np.random.uniform(0.0, 0.3),
                reputation_score=np.random.uniform(0.6, 1.0),
                attack_type=0
            )
    
    def _calculate_reward(self, action: SecurityAction, is_attack: bool) -> float:
        """计算奖励"""
        if action == SecurityAction.BLOCK:
            return self.rewards['correct_block'] if is_attack else self.rewards['false_positive']
        elif action == SecurityAction.ALLOW:
            return self.rewards['correct_allow'] if not is_attack else self.rewards['false_negative']
        elif action == SecurityAction.ALERT:
            return self.rewards['alert_threat'] if is_attack else self.rewards['alert_normal']
        elif action == SecurityAction.RATE_LIMIT:
            return self.rewards['rate_limit_threat'] if is_attack else self.rewards['rate_limit_normal']
        elif action == SecurityAction.QUARANTINE:
            return self.rewards['correct_block'] * 0.8 if is_attack else self.rewards['false_positive'] * 1.2
        elif action == SecurityAction.CHALLENGE:
            return 2.0 if is_attack else -0.5
        else:  # LOG
            return 0.0
    
    def _update_stats(self, action: SecurityAction, is_attack: bool):
        """更新统计"""
        is_blocking = action in [SecurityAction.BLOCK, SecurityAction.QUARANTINE]
        
        if is_attack and is_blocking:
            self.stats['true_positives'] += 1
        elif not is_attack and not is_blocking:
            self.stats['true_negatives'] += 1
        elif not is_attack and is_blocking:
            self.stats['false_positives'] += 1
        else:
            self.stats['false_negatives'] += 1
    
    def get_metrics(self) -> Dict:
        """获取性能指标"""
        tp = self.stats['true_positives']
        tn = self.stats['true_negatives']
        fp = self.stats['false_positives']
        fn = self.stats['false_negatives']
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'episode_reward': self.episode_reward
        }

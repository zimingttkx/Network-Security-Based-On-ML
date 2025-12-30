"""
RL模块单元测试
"""

import pytest
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from networksecurity.models.rl.base import RLAgentBase, RLAgentRegistry, RLConfig, TrainingStats
from networksecurity.models.rl.environment import SecurityEnvironment, Action, State


class TestRLConfig:
    """RL配置测试"""
    
    def test_default_config(self):
        config = RLConfig()
        assert config.learning_rate == 0.001
        assert config.gamma == 0.99
        assert config.epsilon == 1.0
    
    def test_custom_config(self):
        config = RLConfig(learning_rate=0.01, gamma=0.95)
        assert config.learning_rate == 0.01
        assert config.gamma == 0.95


class TestTrainingStats:
    """训练统计测试"""
    
    def test_stats_creation(self):
        stats = TrainingStats(episode=10, total_reward=100.0)
        assert stats.episode == 10
        assert stats.total_reward == 100.0
    
    def test_stats_to_dict(self):
        stats = TrainingStats(episode=5, avg_reward=20.0)
        d = stats.to_dict()
        assert d['episode'] == 5
        assert d['avg_reward'] == 20.0


class TestAction:
    """动作枚举测试"""
    
    def test_action_values(self):
        assert Action.ALLOW == 0
        assert Action.BLOCK == 1
        assert Action.CHALLENGE == 2
        assert Action.LOG == 3
        assert Action.ALERT == 4
    
    def test_action_count(self):
        assert len(Action) == 5


class TestState:
    """状态测试"""
    
    def test_state_creation(self):
        features = np.random.randn(10)
        state = State(features=features, risk_score=0.8)
        assert state.risk_score == 0.8
    
    def test_state_to_array(self):
        features = np.random.randn(10)
        state = State(features=features, risk_score=0.5, threat_type=1, source_reputation=0.3)
        arr = state.to_array()
        assert len(arr) == 13  # 10 + 3


class TestSecurityEnvironment:
    """安全环境测试"""
    
    def test_env_creation(self):
        env = SecurityEnvironment(feature_dim=20)
        assert env.feature_dim == 20
        assert env.state_dim == 23
        assert env.action_dim == 5
    
    def test_reset(self):
        env = SecurityEnvironment()
        state = env.reset()
        assert len(state) == env.state_dim
        assert env.steps == 0
    
    def test_reset_with_threat(self):
        env = SecurityEnvironment()
        state = env.reset(is_threat=True)
        assert env.current_state.risk_score >= 0.7
    
    def test_step_block_threat(self):
        env = SecurityEnvironment()
        env.reset(is_threat=True)
        _, reward, done, info = env.step(Action.BLOCK, true_label=1)
        assert reward > 0  # 正确阻断
        assert done
    
    def test_step_allow_normal(self):
        env = SecurityEnvironment()
        env.reset(is_threat=False)
        _, reward, done, info = env.step(Action.ALLOW, true_label=0)
        assert reward > 0  # 正确放行
    
    def test_step_false_positive(self):
        env = SecurityEnvironment()
        env.reset(is_threat=False)
        _, reward, _, _ = env.step(Action.BLOCK, true_label=0)
        assert reward < 0  # 误报
    
    def test_step_false_negative(self):
        env = SecurityEnvironment()
        env.reset(is_threat=True)
        _, reward, _, _ = env.step(Action.ALLOW, true_label=1)
        assert reward < 0  # 漏报
    
    def test_optimal_action(self):
        env = SecurityEnvironment()
        assert env.get_optimal_action(0.9) == Action.BLOCK
        assert env.get_optimal_action(0.6) == Action.CHALLENGE
        assert env.get_optimal_action(0.1) == Action.ALLOW


class TestRLAgentRegistry:
    """代理注册表测试"""
    
    def test_list_agents(self):
        agents = RLAgentRegistry.list_agents()
        assert 'dqn' in agents
        assert 'double_dqn' in agents
        assert 'ppo' in agents
    
    def test_create_agent(self):
        agent = RLAgentRegistry.create('dqn', state_dim=10, action_dim=5)
        assert agent is not None
        assert agent.name == "DQN"


class TestDQNAgent:
    """DQN代理测试"""
    
    def test_dqn_creation(self):
        from networksecurity.models.rl.agents import DQNAgent
        agent = DQNAgent(state_dim=10, action_dim=5)
        assert agent.state_dim == 10
        assert agent.action_dim == 5
    
    def test_dqn_select_action(self):
        from networksecurity.models.rl.agents import DQNAgent
        agent = DQNAgent(state_dim=10, action_dim=5)
        state = np.random.randn(10)
        action = agent.select_action(state, training=False)
        assert 0 <= action < 5
    
    def test_dqn_learn(self):
        from networksecurity.models.rl.agents import DQNAgent
        agent = DQNAgent(state_dim=10, action_dim=5)
        state = np.random.randn(10)
        next_state = np.random.randn(10)
        loss = agent.learn(state, 0, 1.0, next_state, False)
        assert isinstance(loss, float)


class TestDoubleDQNAgent:
    """Double DQN代理测试"""
    
    def test_double_dqn_creation(self):
        from networksecurity.models.rl.agents import DoubleDQNAgent
        agent = DoubleDQNAgent(state_dim=10, action_dim=5)
        assert agent.name == "Double DQN"


class TestPPOAgent:
    """PPO代理测试"""
    
    def test_ppo_creation(self):
        from networksecurity.models.rl.agents import PPOAgent
        agent = PPOAgent(state_dim=10, action_dim=5)
        assert agent.name == "PPO"
    
    def test_ppo_select_action(self):
        from networksecurity.models.rl.agents import PPOAgent
        agent = PPOAgent(state_dim=10, action_dim=5)
        state = np.random.randn(10)
        action = agent.select_action(state, training=False)
        assert 0 <= action < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

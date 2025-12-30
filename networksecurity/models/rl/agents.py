"""
RL代理实现
包含DQN、Double DQN、PPO等强化学习代理
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import random
import logging

from networksecurity.models.rl.base import RLAgentBase, RLAgentRegistry, RLConfig, TrainingStats

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


@RLAgentRegistry.register("dqn")
class DQNAgent(RLAgentBase):
    """DQN代理"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[RLConfig] = None):
        super().__init__(state_dim, action_dim, config)
        self.memory = ReplayBuffer(self.config.memory_size)
        self.model = None
        self.target_model = None
        self.update_counter = 0
    
    @property
    def name(self) -> str:
        return "DQN"
    
    def _build_model(self):
        try:
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.state_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(self.action_dim, activation='linear')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate), loss='mse')
            return model
        except ImportError:
            raise ImportError("请安装tensorflow")
    
    def _ensure_models(self):
        if self.model is None:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.target_model.set_weights(self.model.get_weights())
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        self._ensure_models()
        if training and np.random.random() < self.config.epsilon:
            return np.random.randint(self.action_dim)
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))
    
    def learn(self, state, action, reward, next_state, done) -> float:
        self._ensure_models()
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) < self.config.batch_size:
            return 0.0
        
        batch = self.memory.sample(self.config.batch_size)
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        targets = current_q.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.config.gamma * np.max(next_q[i])
        
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        self.update_counter += 1
        if self.update_counter % self.config.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        self.decay_epsilon()
        return loss
    
    def save(self, path: str) -> bool:
        try:
            if self.model:
                self.model.save(path)
            return True
        except Exception as e:
            logger.error(f"保存DQN失败: {e}")
            return False
    
    def load(self, path: str) -> bool:
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(path)
            self.target_model = tf.keras.models.load_model(path)
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"加载DQN失败: {e}")
            return False


@RLAgentRegistry.register("double_dqn")
class DoubleDQNAgent(DQNAgent):
    """Double DQN代理"""
    
    @property
    def name(self) -> str:
        return "Double DQN"
    
    def learn(self, state, action, reward, next_state, done) -> float:
        self._ensure_models()
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) < self.config.batch_size:
            return 0.0
        
        batch = self.memory.sample(self.config.batch_size)
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        current_q = self.model.predict(states, verbose=0)
        next_q_online = self.model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)
        
        targets = current_q.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                best_action = np.argmax(next_q_online[i])
                targets[i][actions[i]] = rewards[i] + self.config.gamma * next_q_target[i][best_action]
        
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        self.update_counter += 1
        if self.update_counter % self.config.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        self.decay_epsilon()
        return loss


@RLAgentRegistry.register("ppo")
class PPOAgent(RLAgentBase):
    """PPO代理（简化版）"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[RLConfig] = None):
        super().__init__(state_dim, action_dim, config)
        self.actor = None
        self.critic = None
        self.clip_ratio = 0.2
        self.states, self.actions, self.rewards, self.values, self.log_probs = [], [], [], [], []
    
    @property
    def name(self) -> str:
        return "PPO"
    
    def _build_models(self):
        try:
            import tensorflow as tf
            # Actor
            self.actor = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.state_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(self.action_dim, activation='softmax')
            ])
            self.actor.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate))
            
            # Critic
            self.critic = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.state_dim,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            self.critic.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate), loss='mse')
        except ImportError:
            raise ImportError("请安装tensorflow")
    
    def _ensure_models(self):
        if self.actor is None:
            self._build_models()
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        self._ensure_models()
        state = np.array(state).reshape(1, -1)
        probs = self.actor.predict(state, verbose=0)[0]
        
        if training:
            action = np.random.choice(self.action_dim, p=probs)
            self.log_probs.append(np.log(probs[action] + 1e-10))
            self.values.append(self.critic.predict(state, verbose=0)[0][0])
        else:
            action = np.argmax(probs)
        return int(action)
    
    def learn(self, state, action, reward, next_state, done) -> float:
        self._ensure_models()
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
        if not done:
            return 0.0
        
        # 计算优势
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        returns = np.array(returns)
        values = np.array(self.values)
        advantages = returns - values
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)
        
        # 更新
        loss = self._update(states, actions, advantages, returns, old_log_probs)
        
        # 清空缓存
        self.states, self.actions, self.rewards, self.values, self.log_probs = [], [], [], [], []
        return loss
    
    def _update(self, states, actions, advantages, returns, old_log_probs) -> float:
        import tensorflow as tf
        
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            indices = tf.stack([tf.range(len(actions)), actions], axis=1)
            new_probs = tf.gather_nd(probs, indices)
            new_log_probs = tf.math.log(new_probs + 1e-10)
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))
        
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        
        self.critic.fit(states, returns, epochs=1, verbose=0)
        return float(actor_loss.numpy())
    
    def save(self, path: str) -> bool:
        try:
            if self.actor:
                self.actor.save(f"{path}_actor")
                self.critic.save(f"{path}_critic")
            return True
        except:
            return False
    
    def load(self, path: str) -> bool:
        try:
            import tensorflow as tf
            self.actor = tf.keras.models.load_model(f"{path}_actor")
            self.critic = tf.keras.models.load_model(f"{path}_critic")
            self.is_trained = True
            return True
        except:
            return False

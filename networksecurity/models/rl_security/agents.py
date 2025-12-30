"""
强化学习代理
DQN, Double DQN, PPO 实现
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
import random

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN代理
    使用深度Q网络进行决策
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.memory = ReplayBuffer(buffer_size)
        self.model = None
        self._tf = None
    
    def _check_tensorflow(self):
        if self._tf is None:
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                self._tf = tf
            except ImportError:
                raise ImportError("需要TensorFlow")
        return self._tf
    
    def _build_model(self):
        tf = self._check_tensorflow()
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model
    
    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        if self.model is None:
            self.model = self._build_model()
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(np.argmax(q_values[0]))
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if self.model is None:
            self.model = self._build_model()
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        targets = self.model.predict(states, verbose=0)
        next_q = self.model.predict(next_states, verbose=0)
        
        for i in range(len(states)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def save(self, path: str):
        if self.model:
            self.model.save(path)
    
    def load(self, path: str):
        tf = self._check_tensorflow()
        self.model = tf.keras.models.load_model(path)


class DoubleDQNAgent(DQNAgent):
    """Double DQN代理"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_model = None
        self.update_target_freq = 100
        self.train_step = 0
    
    def _update_target(self):
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
    
    def train(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if self.model is None:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self._update_target()
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        targets = self.model.predict(states, verbose=0)
        next_q_main = self.model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)
        
        for i in range(len(states)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                best_action = np.argmax(next_q_main[i])
                targets[i][actions[i]] = rewards[i] + self.gamma * next_q_target[i][best_action]
        
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self._update_target()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]


class PPOAgent:
    """PPO代理"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 clip_ratio: float = 0.2,
                 epochs: int = 10):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        
        self.actor = None
        self.critic = None
        self._tf = None
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
    
    def _check_tensorflow(self):
        if self._tf is None:
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                self._tf = tf
            except ImportError:
                raise ImportError("需要TensorFlow")
        return self._tf
    
    def _build_actor(self):
        tf = self._check_tensorflow()
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def _build_critic(self):
        tf = self._check_tensorflow()
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model
    
    def select_action(self, state: np.ndarray) -> int:
        if self.actor is None:
            self.actor = self._build_actor()
            self.critic = self._build_critic()
        
        probs = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        action = np.random.choice(self.action_dim, p=probs)
        
        value = self.critic.predict(state.reshape(1, -1), verbose=0)[0][0]
        log_prob = np.log(probs[action] + 1e-10)
        
        self.values.append(value)
        self.log_probs.append(log_prob)
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train(self) -> float:
        if len(self.states) < 32:
            return 0.0
        
        returns = self._compute_returns()
        advantages = returns - np.array(self.values[:len(returns)])
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        states = np.array(self.states[:len(returns)])
        actions = np.array(self.actions[:len(returns)])
        old_log_probs = np.array(self.log_probs[:len(returns)])
        
        total_loss = 0.0
        for _ in range(self.epochs):
            loss = self._train_step(states, actions, old_log_probs, returns, advantages)
            total_loss += loss
        
        self._clear_memory()
        return total_loss / self.epochs
    
    def _compute_returns(self) -> np.ndarray:
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return np.array(returns)
    
    def _train_step(self, states, actions, old_log_probs, returns, advantages) -> float:
        tf = self._tf
        
        with tf.GradientTape() as tape:
            probs = self.actor(states)
            indices = tf.stack([tf.range(len(actions)), actions], axis=1)
            action_probs = tf.gather_nd(probs, indices)
            new_log_probs = tf.math.log(action_probs + 1e-10)
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        self.critic.fit(states, returns, epochs=1, verbose=0)
        
        return float(actor_loss)
    
    def _clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()

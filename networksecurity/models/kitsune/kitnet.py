"""
KitNET - 自编码器集成异常检测
基于 ymirsky/Kitsune-py 实现

KitNET使用多个小型自编码器组成的集成来检测网络异常。
每个自编码器负责学习特征子集的正常模式。
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class AutoEncoder:
    """
    轻量级自编码器
    使用单隐藏层实现快速训练和推理
    """
    
    def __init__(self, input_dim: int, hidden_ratio: float = 0.75, 
                 learning_rate: float = 0.1):
        """
        Args:
            input_dim: 输入维度
            hidden_ratio: 隐藏层相对于输入层的比例
            learning_rate: 学习率
        """
        self.input_dim = input_dim
        self.hidden_dim = max(1, int(input_dim * hidden_ratio))
        self.learning_rate = learning_rate
        
        # 初始化权重 (Xavier初始化)
        limit = np.sqrt(6.0 / (input_dim + self.hidden_dim))
        self.W_encode = np.random.uniform(-limit, limit, (input_dim, self.hidden_dim))
        self.b_encode = np.zeros(self.hidden_dim)
        
        limit = np.sqrt(6.0 / (self.hidden_dim + input_dim))
        self.W_decode = np.random.uniform(-limit, limit, (self.hidden_dim, input_dim))
        self.b_decode = np.zeros(input_dim)
        
        # 归一化参数
        self.norm_mean = None
        self.norm_std = None
        self.is_fitted = False
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid导数"""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """归一化输入"""
        if self.norm_mean is None:
            return x
        return (x - self.norm_mean) / (self.norm_std + 1e-10)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """编码"""
        return self._sigmoid(np.dot(x, self.W_encode) + self.b_encode)
    
    def decode(self, h: np.ndarray) -> np.ndarray:
        """解码"""
        return self._sigmoid(np.dot(h, self.W_decode) + self.b_decode)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """前向传播，返回 (重构, 隐藏层, 归一化输入)"""
        x_norm = self._normalize(x)
        h = self.encode(x_norm)
        x_recon = self.decode(h)
        return x_recon, h, x_norm
    
    def train_step(self, x: np.ndarray) -> float:
        """单步训练，返回重构误差"""
        x_recon, h, x_norm = self.forward(x)
        
        # 计算误差
        error = x_norm - x_recon
        rmse = np.sqrt(np.mean(error ** 2))
        
        # 反向传播
        d_decode = error * self._sigmoid_derivative(
            np.dot(h, self.W_decode) + self.b_decode
        )
        d_encode = np.dot(d_decode, self.W_decode.T) * self._sigmoid_derivative(
            np.dot(x_norm, self.W_encode) + self.b_encode
        )
        
        # 更新权重
        self.W_decode += self.learning_rate * np.outer(h, d_decode)
        self.b_decode += self.learning_rate * d_decode
        self.W_encode += self.learning_rate * np.outer(x_norm, d_encode)
        self.b_encode += self.learning_rate * d_encode
        
        return rmse
    
    def compute_rmse(self, x: np.ndarray) -> float:
        """计算重构RMSE"""
        x_recon, _, x_norm = self.forward(x)
        return np.sqrt(np.mean((x_norm - x_recon) ** 2))
    
    def fit_normalization(self, X: np.ndarray):
        """拟合归一化参数"""
        self.norm_mean = np.mean(X, axis=0)
        self.norm_std = np.std(X, axis=0)
        self.norm_std[self.norm_std < 1e-10] = 1.0


class KitNET:
    """
    KitNET - 自编码器集成
    
    架构:
    1. 特征映射层: 将输入特征分组到多个小型自编码器
    2. 集成层: 多个自编码器并行处理各自的特征子集
    3. 输出层: 一个自编码器聚合所有集成层的RMSE输出
    """
    
    def __init__(self, input_dim: int, max_autoencoder_size: int = 10,
                 fm_grace_period: int = 5000, ad_grace_period: int = 50000,
                 learning_rate: float = 0.1, hidden_ratio: float = 0.75):
        """
        Args:
            input_dim: 输入特征维度
            max_autoencoder_size: 单个自编码器最大输入维度
            fm_grace_period: 特征映射学习期（数据包数）
            ad_grace_period: 异常检测训练期（数据包数）
            learning_rate: 学习率
            hidden_ratio: 隐藏层比例
        """
        self.input_dim = input_dim
        self.max_ae_size = max_autoencoder_size
        self.fm_grace = fm_grace_period
        self.ad_grace = ad_grace_period
        self.learning_rate = learning_rate
        self.hidden_ratio = hidden_ratio
        
        # 特征映射
        self.feature_map: List[List[int]] = []
        
        # 集成层自编码器
        self.ensemble: List[AutoEncoder] = []
        
        # 输出层自编码器
        self.output_ae: Optional[AutoEncoder] = None
        
        # 训练状态
        self.n_trained = 0
        self.fm_data: List[np.ndarray] = []
        self.is_fm_done = False
        self.is_ad_done = False
        
        # 异常阈值
        self.threshold = None
        self.rmse_history: List[float] = []
    
    def _build_feature_map(self, X: np.ndarray):
        """构建特征映射 - 使用相关性聚类"""
        n_features = X.shape[1]
        
        # 计算特征相关性矩阵
        corr_matrix = np.corrcoef(X.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # 贪婪聚类
        assigned = set()
        self.feature_map = []
        
        for i in range(n_features):
            if i in assigned:
                continue
            
            # 创建新组
            group = [i]
            assigned.add(i)
            
            # 找相关特征
            correlations = np.abs(corr_matrix[i])
            sorted_indices = np.argsort(correlations)[::-1]
            
            for j in sorted_indices:
                if j in assigned:
                    continue
                if len(group) >= self.max_ae_size:
                    break
                if correlations[j] > 0.3:  # 相关性阈值
                    group.append(j)
                    assigned.add(j)
            
            self.feature_map.append(group)
        
        logger.info(f"KitNET: 创建了 {len(self.feature_map)} 个特征组")
    
    def _build_ensemble(self):
        """构建自编码器集成"""
        self.ensemble = []
        for group in self.feature_map:
            ae = AutoEncoder(
                input_dim=len(group),
                hidden_ratio=self.hidden_ratio,
                learning_rate=self.learning_rate
            )
            self.ensemble.append(ae)
        
        # 输出层自编码器
        self.output_ae = AutoEncoder(
            input_dim=len(self.ensemble),
            hidden_ratio=self.hidden_ratio,
            learning_rate=self.learning_rate
        )
        
        logger.info(f"KitNET: 构建了 {len(self.ensemble)} 个集成自编码器")
    
    def process(self, x: np.ndarray) -> float:
        """
        处理单个样本
        
        在训练期间返回0，训练完成后返回异常分数(RMSE)
        """
        self.n_trained += 1
        
        # 特征映射学习期
        if self.n_trained <= self.fm_grace:
            self.fm_data.append(x.copy())
            return 0.0
        
        # 完成特征映射
        if not self.is_fm_done:
            fm_array = np.array(self.fm_data)
            self._build_feature_map(fm_array)
            self._build_ensemble()
            
            # 为每个自编码器拟合归一化参数
            for i, group in enumerate(self.feature_map):
                group_data = fm_array[:, group]
                self.ensemble[i].fit_normalization(group_data)
            
            self.fm_data = []  # 释放内存
            self.is_fm_done = True
        
        # 异常检测训练期
        if self.n_trained <= self.fm_grace + self.ad_grace:
            rmse = self._train_step(x)
            self.rmse_history.append(rmse)
            return 0.0
        
        # 完成训练，设置阈值
        if not self.is_ad_done:
            if self.rmse_history:
                self.threshold = np.percentile(self.rmse_history, 99)
            else:
                self.threshold = 1.0
            self.rmse_history = []
            self.is_ad_done = True
            logger.info(f"KitNET: 训练完成，阈值={self.threshold:.4f}")
        
        # 执行检测
        return self._execute(x)
    
    def _train_step(self, x: np.ndarray) -> float:
        """训练步骤"""
        ensemble_rmses = []
        
        for i, group in enumerate(self.feature_map):
            x_group = x[group]
            rmse = self.ensemble[i].train_step(x_group)
            ensemble_rmses.append(rmse)
        
        # 训练输出层
        ensemble_rmses = np.array(ensemble_rmses)
        output_rmse = self.output_ae.train_step(ensemble_rmses)
        
        return output_rmse
    
    def _execute(self, x: np.ndarray) -> float:
        """执行检测，返回异常分数"""
        ensemble_rmses = []
        
        for i, group in enumerate(self.feature_map):
            x_group = x[group]
            rmse = self.ensemble[i].compute_rmse(x_group)
            ensemble_rmses.append(rmse)
        
        ensemble_rmses = np.array(ensemble_rmses)
        output_rmse = self.output_ae.compute_rmse(ensemble_rmses)
        
        return output_rmse
    
    def is_anomaly(self, rmse: float) -> bool:
        """判断是否为异常"""
        if self.threshold is None:
            return False
        return rmse > self.threshold
    
    def get_state(self) -> Dict:
        """获取模型状态"""
        return {
            'n_trained': self.n_trained,
            'is_fm_done': self.is_fm_done,
            'is_ad_done': self.is_ad_done,
            'threshold': self.threshold,
            'n_ensembles': len(self.ensemble) if self.ensemble else 0
        }

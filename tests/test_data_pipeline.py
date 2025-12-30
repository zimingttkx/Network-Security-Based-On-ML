"""
数据管道模块单元测试
"""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np

from networksecurity.data.dataset_base import (
    DatasetBase, DatasetRegistry, DatasetInfo, DatasetType, AttackCategory
)
from networksecurity.data.datasets import PhishingDataset, CustomDataset
from networksecurity.data.balancer import DataBalancer, BalanceStrategy
from networksecurity.data.pipeline import DataPipeline, PipelineConfig


class TestDatasetBase:
    """数据集基类测试"""
    
    def test_phishing_dataset_info(self):
        """测试钓鱼数据集信息"""
        dataset = PhishingDataset()
        info = dataset.info
        assert info.name == "Phishing Website Dataset"
        assert info.type == DatasetType.PHISHING
        assert info.num_features == 30
        assert info.num_classes == 2
    
    def test_phishing_feature_names(self):
        """测试特征名列表"""
        dataset = PhishingDataset()
        features = dataset.get_feature_names()
        assert len(features) == 30
        assert 'having_IP_Address' in features
        assert 'URL_Length' in features
    
    def test_phishing_target_column(self):
        """测试目标列名"""
        dataset = PhishingDataset()
        assert dataset.get_target_column() == 'Result'


class TestDatasetRegistry:
    """数据集注册表测试"""
    
    def test_list_datasets(self):
        """测试列出数据集"""
        datasets = DatasetRegistry.list_datasets()
        assert 'phishing' in datasets
        assert 'custom' in datasets
    
    def test_get_dataset(self):
        """测试获取数据集类"""
        dataset_class = DatasetRegistry.get('phishing')
        assert dataset_class == PhishingDataset
    
    def test_create_dataset(self):
        """测试创建数据集实例"""
        dataset = DatasetRegistry.create('phishing')
        assert isinstance(dataset, PhishingDataset)
    
    def test_get_nonexistent(self):
        """测试获取不存在的数据集"""
        dataset = DatasetRegistry.get('nonexistent')
        assert dataset is None


class TestCustomDataset:
    """自定义数据集测试"""
    
    def test_custom_dataset_creation(self):
        """测试创建自定义数据集"""
        dataset = CustomDataset(target_column='label', feature_columns=['f1', 'f2'])
        assert dataset.get_target_column() == 'label'
        assert dataset.get_feature_names() == ['f1', 'f2']
    
    def test_custom_dataset_preprocess(self):
        """测试自定义数据集预处理"""
        dataset = CustomDataset(target_column='label', feature_columns=['f1', 'f2'])
        df = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6], 'label': [0, 1, 0]})
        X, y = dataset.preprocess(df)
        assert len(X) == 3
        assert len(y) == 3


class TestDataBalancer:
    """数据平衡器测试"""
    
    @pytest.fixture
    def imbalanced_data(self):
        """创建不平衡数据"""
        np.random.seed(42)
        X = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100)
        })
        y = pd.Series([0] * 90 + [1] * 10)
        return X, y
    
    def test_balancer_creation(self):
        """测试创建平衡器"""
        balancer = DataBalancer()
        assert balancer is not None
    
    def test_get_class_distribution(self, imbalanced_data):
        """测试获取类别分布"""
        X, y = imbalanced_data
        balancer = DataBalancer()
        dist = balancer.get_class_distribution(y)
        assert dist[0] == 90
        assert dist[1] == 10
    
    def test_get_imbalance_ratio(self, imbalanced_data):
        """测试计算不平衡比率"""
        X, y = imbalanced_data
        balancer = DataBalancer()
        ratio = balancer.get_imbalance_ratio(y)
        assert ratio == 9.0
    
    def test_balance_none(self, imbalanced_data):
        """测试无平衡策略"""
        X, y = imbalanced_data
        balancer = DataBalancer()
        X_bal, y_bal, report = balancer.balance(X, y, strategy='none')
        assert len(X_bal) == len(X)
        assert report.success
    
    def test_available_strategies(self):
        """测试获取可用策略"""
        balancer = DataBalancer()
        strategies = balancer.get_available_strategies()
        assert 'none' in strategies


class TestPipelineConfig:
    """管道配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PipelineConfig()
        assert config.dataset_name == 'phishing'
        assert config.test_size == 0.2
        assert config.balance_strategy == 'none'
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = PipelineConfig(
            dataset_name='custom',
            test_size=0.3,
            balance_strategy='smote'
        )
        assert config.dataset_name == 'custom'
        assert config.test_size == 0.3
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        config = PipelineConfig()
        d = config.to_dict()
        assert 'dataset_name' in d
        assert 'test_size' in d


class TestDataPipeline:
    """数据管道测试"""
    
    @pytest.fixture
    def sample_df(self):
        """创建示例数据"""
        np.random.seed(42)
        n = 200
        data = {f'f{i}': np.random.randn(n) for i in range(5)}
        data['label'] = np.random.randint(0, 2, n)
        return pd.DataFrame(data)
    
    def test_pipeline_creation(self):
        """测试创建管道"""
        pipeline = DataPipeline()
        assert pipeline is not None
    
    def test_pipeline_with_config(self):
        """测试带配置的管道"""
        config = PipelineConfig(dataset_name='custom', test_size=0.3)
        pipeline = DataPipeline(config)
        assert pipeline.config.test_size == 0.3
    
    def test_pipeline_run_with_df(self, sample_df):
        """测试使用DataFrame运行管道"""
        config = PipelineConfig(
            dataset_name='custom',
            scale_features=False,
            balance_strategy='none'
        )
        pipeline = DataPipeline(config)
        result = pipeline.run(sample_df)
        
        assert result.success
        assert result.X_train is not None
        assert result.X_test is not None
        assert len(result.X_train) > 0
    
    def test_pipeline_with_scaling(self, sample_df):
        """测试带缩放的管道"""
        config = PipelineConfig(
            dataset_name='custom',
            scale_features=True
        )
        pipeline = DataPipeline(config)
        result = pipeline.run(sample_df)
        
        assert result.success
        assert result.scaler is not None
    
    def test_list_datasets(self):
        """测试列出数据集"""
        datasets = DataPipeline.list_datasets()
        assert len(datasets) > 0
    
    def test_list_balance_strategies(self):
        """测试列出平衡策略"""
        strategies = DataPipeline.list_balance_strategies()
        assert 'none' in strategies
        assert 'smote' in strategies


class TestDatasetInfo:
    """数据集信息测试"""
    
    def test_info_creation(self):
        """测试创建数据集信息"""
        info = DatasetInfo(
            name="Test Dataset",
            type=DatasetType.CUSTOM,
            description="测试数据集",
            num_features=10,
            num_classes=2,
            attack_categories=[AttackCategory.BENIGN]
        )
        assert info.name == "Test Dataset"
        assert info.num_features == 10
    
    def test_info_to_dict(self):
        """测试信息转字典"""
        info = DatasetInfo(
            name="Test",
            type=DatasetType.PHISHING,
            description="Test",
            num_features=5,
            num_classes=2,
            attack_categories=[AttackCategory.PHISHING]
        )
        d = info.to_dict()
        assert d['name'] == "Test"
        assert d['type'] == 'phishing'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

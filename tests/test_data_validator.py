"""
测试数据验证和特征补全功能
"""

import pytest
import pandas as pd
import numpy as np
from networksecurity.utils.ml_utils.data_validator import DataValidator


class TestDataValidator:
    """数据验证器测试类"""
    
    @pytest.fixture
    def validator(self):
        """创建验证器实例"""
        return DataValidator()
    
    @pytest.fixture
    def complete_data(self):
        """创建完整的测试数据"""
        data = {}
        for feature in DataValidator.REQUIRED_FEATURES:
            data[feature] = np.random.choice([-1, 0, 1], size=100)
        return pd.DataFrame(data)
    
    @pytest.fixture
    def incomplete_data(self):
        """创建不完整的测试数据"""
        data = {}
        # 只包含前10个特征
        for feature in DataValidator.REQUIRED_FEATURES[:10]:
            data[feature] = np.random.choice([-1, 0, 1], size=100)
        return pd.DataFrame(data)
    
    @pytest.fixture
    def data_with_missing_values(self):
        """创建有缺失值的测试数据"""
        data = {}
        for i, feature in enumerate(DataValidator.REQUIRED_FEATURES):
            values = np.random.choice([-1, 0, 1], size=100).astype(float)
            # 给某些特征添加缺失值
            if i % 3 == 0:
                indices = np.random.choice(100, 10, replace=False)
                values[indices] = np.nan
            data[feature] = values
        return pd.DataFrame(data)
    
    def test_get_feature_requirements(self, validator):
        """测试获取特征要求"""
        requirements = validator.get_feature_requirements()
        
        assert 'total_features' in requirements
        assert 'features' in requirements
        assert requirements['total_features'] == 30
        assert len(requirements['features']) == 30
        
        # 验证每个特征都有必要的字段
        for feature in requirements['features']:
            assert 'name' in feature
            assert 'description' in feature
            assert 'type' in feature
            assert feature['name'] in DataValidator.REQUIRED_FEATURES
    
    def test_validate_complete_data(self, validator, complete_data):
        """测试验证完整数据"""
        is_valid, report = validator.validate_features(complete_data)
        
        assert is_valid is True
        assert len(report['missing_features']) == 0
        assert report['is_valid'] is True
    
    def test_validate_incomplete_data(self, validator, incomplete_data):
        """测试验证不完整数据"""
        is_valid, report = validator.validate_features(incomplete_data)
        
        assert is_valid is False
        assert len(report['missing_features']) == 20
        assert report['is_valid'] is False
        
        # 应该有关于缺失特征的建议
        assert len(report['recommendations']) > 0
    
    def test_validate_data_with_missing_values(self, validator, data_with_missing_values):
        """测试验证有缺失值的数据"""
        is_valid, report = validator.validate_features(data_with_missing_values)
        
        assert 'missing_values' in report
        assert len(report['missing_values']) > 0
    
    def test_impute_missing_features_constant(self, validator, incomplete_data):
        """测试使用常数补全缺失特征"""
        df_imputed, report = validator.impute_missing_features(
            incomplete_data,
            strategy='constant',
            fill_value=0
        )
        
        # 验证所有特征都存在
        assert len(df_imputed.columns) == 30
        for feature in DataValidator.REQUIRED_FEATURES:
            assert feature in df_imputed.columns
        
        # 验证报告
        assert 'added_features' in report
        assert len(report['added_features']) == 20
        assert report['strategy'] == 'constant'
    
    def test_impute_missing_values_mean(self, validator, data_with_missing_values):
        """测试使用均值补全缺失值"""
        df_imputed, report = validator.impute_missing_features(
            data_with_missing_values,
            strategy='mean'
        )
        
        # 验证没有缺失值
        assert df_imputed.isnull().sum().sum() == 0
        assert 'imputed_values' in report
    
    def test_suggest_imputation_strategy_no_missing(self, validator, complete_data):
        """测试无缺失数据的补全策略建议"""
        suggestions = validator.suggest_imputation_strategy(complete_data)
        
        assert 'missing_percentage' in suggestions
        assert suggestions['missing_percentage'] == 0
        assert len(suggestions['suggestions']) > 0
    
    def test_suggest_imputation_strategy_with_missing(self, validator, data_with_missing_values):
        """测试有缺失数据的补全策略建议"""
        suggestions = validator.suggest_imputation_strategy(data_with_missing_values)
        
        assert 'missing_percentage' in suggestions
        assert suggestions['missing_percentage'] > 0
        assert len(suggestions['suggestions']) > 0
        
        # 验证建议有优先级
        for suggestion in suggestions['suggestions']:
            assert 'strategy' in suggestion
            assert 'reason' in suggestion
            assert 'priority' in suggestion
    
    def test_column_order_preserved(self, validator, incomplete_data):
        """测试补全后列顺序与REQUIRED_FEATURES一致"""
        df_imputed, _ = validator.impute_missing_features(incomplete_data)
        
        assert list(df_imputed.columns) == DataValidator.REQUIRED_FEATURES


# 运行测试
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

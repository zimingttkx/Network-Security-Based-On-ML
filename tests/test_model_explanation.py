"""
测试模型解释功能
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from networksecurity.utils.ml_utils.model_explanation import ModelExplainer


class TestModelExplainer:
    """模型解释器测试类"""
    
    @pytest.fixture
    def explainer(self):
        """创建解释器实例"""
        return ModelExplainer()
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        np.random.seed(42)
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(100) for i in range(10)
        })
        y = np.random.randint(0, 2, 100)
        return X, y
    
    @pytest.fixture
    def trained_rf_model(self, sample_data):
        """训练RandomForest模型"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def trained_xgb_model(self, sample_data):
        """训练XGBoost模型"""
        X, y = sample_data
        model = XGBClassifier(n_estimators=10, random_state=42, eval_metric='logloss')
        model.fit(X, y)
        return model
    
    def test_explainer_initialization(self, explainer):
        """测试解释器初始化"""
        assert explainer is not None
        assert explainer.feature_names is None
        assert explainer.explainer is None
    
    def test_explainer_with_feature_names(self):
        """测试带特征名称的解释器初始化"""
        feature_names = ['f1', 'f2', 'f3']
        explainer = ModelExplainer(feature_names=feature_names)
        assert explainer.feature_names == feature_names
    
    def test_is_tree_model_rf(self, explainer, trained_rf_model):
        """测试RandomForest模型类型检测"""
        assert explainer._is_tree_model(trained_rf_model) is True
    
    def test_is_tree_model_xgb(self, explainer, trained_xgb_model):
        """测试XGBoost模型类型检测"""
        assert explainer._is_tree_model(trained_xgb_model) is True
    
    def test_get_explanation_rf(self, explainer, trained_rf_model, sample_data):
        """测试RandomForest模型的SHAP解释"""
        X, _ = sample_data
        input_data = X.iloc[:5]
        
        result = explainer.get_explanation(trained_rf_model, input_data)
        
        assert result['success'] is True
        assert result['model_type'] == 'RandomForestClassifier'
        assert result['num_samples'] == 5
        assert len(result['explanations']) == 5
        
        # 验证解释结构
        explanation = result['explanations'][0]
        assert 'base_value' in explanation
        assert 'features' in explanation
        assert 'top_features' in explanation
        assert len(explanation['features']) == 10
    
    @pytest.mark.skip(reason="SHAP与XGBoost版本兼容性问题，跳过此测试")
    def test_get_explanation_xgb(self, explainer, trained_xgb_model, sample_data):
        """测试XGBoost模型的SHAP解释"""
        X, _ = sample_data
        input_data = X.iloc[:3]
        
        result = explainer.get_explanation(trained_xgb_model, input_data)
        
        assert result['success'] is True
        assert result['model_type'] == 'XGBClassifier'
        assert result['num_samples'] == 3
    
    def test_get_explanation_numpy_array(self, explainer, trained_rf_model, sample_data):
        """测试使用numpy数组输入"""
        X, _ = sample_data
        input_data = X.iloc[:2].values
        
        result = explainer.get_explanation(trained_rf_model, input_data)
        
        assert result['success'] is True
        assert result['num_samples'] == 2
    
    def test_get_explanation_single_sample(self, explainer, trained_rf_model, sample_data):
        """测试单个样本的解释"""
        X, _ = sample_data
        input_data = X.iloc[0].values
        
        result = explainer.get_explanation(trained_rf_model, input_data)
        
        assert result['success'] is True
        assert result['num_samples'] == 1
    
    def test_generate_summary_plot_rf(self, explainer, trained_rf_model, sample_data):
        """测试RandomForest模型的SHAP总结图生成"""
        X, _ = sample_data
        
        result = explainer.generate_summary_plot(trained_rf_model, X.iloc[:20])
        
        assert result['success'] is True
        assert result['model_type'] == 'RandomForestClassifier'
        assert 'image_base64' in result
        assert result['image_format'] == 'png'
        assert len(result['image_base64']) > 0
    
    def test_generate_summary_plot_bar(self, explainer, trained_rf_model, sample_data):
        """测试生成bar类型的总结图"""
        X, _ = sample_data
        
        result = explainer.generate_summary_plot(
            trained_rf_model, 
            X.iloc[:20], 
            plot_type='bar'
        )
        
        assert result['success'] is True
        assert result['plot_type'] == 'bar'
    
    def test_generate_force_plot(self, explainer, trained_rf_model, sample_data):
        """测试生成SHAP力图"""
        X, _ = sample_data
        
        result = explainer.generate_force_plot(trained_rf_model, X.iloc[:5], sample_index=0)
        
        assert result['success'] is True
        assert 'image_base64' in result
        assert result['sample_index'] == 0
    
    def test_feature_contribution_values(self, explainer, trained_rf_model, sample_data):
        """测试特征贡献值的正确性"""
        X, _ = sample_data
        input_data = X.iloc[:1]
        
        result = explainer.get_explanation(trained_rf_model, input_data)
        
        explanation = result['explanations'][0]
        for feature_name, feature_info in explanation['features'].items():
            assert 'value' in feature_info
            assert 'shap_value' in feature_info
            assert 'contribution' in feature_info
            assert feature_info['contribution'] in ['positive', 'negative']
    
    def test_top_features_sorted(self, explainer, trained_rf_model, sample_data):
        """测试top_features按SHAP值绝对值排序"""
        X, _ = sample_data
        input_data = X.iloc[:1]
        
        result = explainer.get_explanation(trained_rf_model, input_data)
        
        top_features = result['explanations'][0]['top_features']
        shap_values = [abs(f['shap_value']) for f in top_features]
        
        # 验证是降序排列
        assert shap_values == sorted(shap_values, reverse=True)


# 运行测试
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

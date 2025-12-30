"""
DL模型模块单元测试
"""

import pytest
import numpy as np
import pandas as pd
import os

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from networksecurity.models.dl.base import (
    DLModelBase, DLModelRegistry, DLModelInfo, DLModelType, DLTrainingResult
)


@pytest.fixture
def sample_data():
    """创建示例分类数据"""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({f'f{i}': np.random.randn(n) for i in range(10)})
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


@pytest.fixture
def train_test_data(sample_data):
    """划分训练测试数据"""
    X, y = sample_data
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


class TestDLModelInfo:
    """DL模型信息测试"""
    
    def test_info_creation(self):
        info = DLModelInfo(name="Test", type=DLModelType.DNN, description="测试")
        assert info.name == "Test"
        assert info.type == DLModelType.DNN
    
    def test_info_to_dict(self):
        info = DLModelInfo(name="Test", type=DLModelType.LSTM, description="Test")
        d = info.to_dict()
        assert d['name'] == "Test"
        assert d['type'] == 'lstm'


class TestDLModelRegistry:
    """DL模型注册表测试"""
    
    def test_list_models(self):
        models = DLModelRegistry.list_models()
        assert 'dnn' in models
        assert 'lstm' in models
    
    def test_create_model(self):
        model = DLModelRegistry.create('dnn', input_dim=10)
        assert model is not None
        assert model.input_dim == 10


class TestDNNClassifier:
    """DNN分类器测试"""
    
    def test_dnn_info(self):
        from networksecurity.models.dl.classifiers import DNNClassifier
        model = DNNClassifier(input_dim=10)
        assert model.info.name == "DNN"
        assert model.info.type == DLModelType.DNN
    
    def test_dnn_fit(self, train_test_data):
        from networksecurity.models.dl.classifiers import DNNClassifier
        X_train, X_test, y_train, y_test = train_test_data
        model = DNNClassifier(input_dim=X_train.shape[1], hidden_layers=[32, 16])
        result = model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=0)
        assert result.success
        assert model.is_fitted
    
    def test_dnn_predict(self, train_test_data):
        from networksecurity.models.dl.classifiers import DNNClassifier
        X_train, X_test, y_train, y_test = train_test_data
        model = DNNClassifier(input_dim=X_train.shape[1], hidden_layers=[16])
        model.fit(X_train, y_train, epochs=2, verbose=0)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)


class TestLSTMClassifier:
    """LSTM分类器测试"""
    
    def test_lstm_fit(self, train_test_data):
        from networksecurity.models.dl.classifiers import LSTMClassifier
        X_train, X_test, y_train, y_test = train_test_data
        model = LSTMClassifier(input_dim=X_train.shape[1], units=[16])
        result = model.fit(X_train, y_train, epochs=2, verbose=0)
        assert result.success
    
    def test_lstm_predict(self, train_test_data):
        from networksecurity.models.dl.classifiers import LSTMClassifier
        X_train, X_test, y_train, y_test = train_test_data
        model = LSTMClassifier(input_dim=X_train.shape[1], units=[16])
        model.fit(X_train, y_train, epochs=2, verbose=0)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)


class TestAutoEncoder:
    """AutoEncoder测试"""
    
    def test_autoencoder_fit(self, sample_data):
        from networksecurity.models.dl.autoencoder import AutoEncoderDetector
        X, _ = sample_data
        model = AutoEncoderDetector(input_dim=X.shape[1], encoding_dim=4, hidden_layers=[16])
        result = model.fit(X, epochs=2, verbose=0)
        assert result.success
        assert model.threshold is not None
    
    def test_autoencoder_predict(self, sample_data):
        from networksecurity.models.dl.autoencoder import AutoEncoderDetector
        X, _ = sample_data
        model = AutoEncoderDetector(input_dim=X.shape[1], encoding_dim=4, hidden_layers=[16])
        model.fit(X, epochs=2, verbose=0)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({-1, 1})


class TestDLTrainingResult:
    """训练结果测试"""
    
    def test_result_creation(self):
        result = DLTrainingResult(success=True, model_name="Test", train_time=1.5)
        assert result.success
        assert result.train_time == 1.5
    
    def test_result_to_dict(self):
        result = DLTrainingResult(success=True, model_name="Test", final_accuracy=0.95)
        d = result.to_dict()
        assert d['success'] == True
        assert d['final_accuracy'] == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

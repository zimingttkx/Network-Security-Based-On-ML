"""
ML模型模块单元测试
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from networksecurity.models.ml.base import (
    MLModelBase, ModelRegistry, ModelInfo, ModelType, TrainingResult
)
from networksecurity.models.ml.classifiers import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, SVMClassifier, LogisticRegressionClassifier,
    KNNClassifier, NaiveBayesClassifier, DecisionTreeClassifier
)
from networksecurity.models.ml.anomaly import (
    IsolationForestDetector, OneClassSVMDetector, LOFDetector
)


@pytest.fixture
def sample_data():
    """创建示例分类数据"""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n)
    })
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


@pytest.fixture
def train_test_data(sample_data):
    """划分训练测试数据"""
    X, y = sample_data
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


class TestModelInfo:
    """模型信息测试"""
    
    def test_info_creation(self):
        info = ModelInfo(
            name="Test Model",
            type=ModelType.ENSEMBLE,
            description="测试模型"
        )
        assert info.name == "Test Model"
        assert info.type == ModelType.ENSEMBLE
    
    def test_info_to_dict(self):
        info = ModelInfo(name="Test", type=ModelType.LINEAR, description="Test")
        d = info.to_dict()
        assert d['name'] == "Test"
        assert d['type'] == 'linear'


class TestModelRegistry:
    """模型注册表测试"""
    
    def test_list_models(self):
        models = ModelRegistry.list_models()
        assert 'random_forest' in models
        assert 'svm' in models
        assert 'knn' in models
    
    def test_get_model(self):
        model_class = ModelRegistry.get('random_forest')
        assert model_class == RandomForestClassifier
    
    def test_create_model(self):
        model = ModelRegistry.create('random_forest', n_estimators=10)
        assert isinstance(model, RandomForestClassifier)
    
    def test_get_nonexistent(self):
        model = ModelRegistry.get('nonexistent')
        assert model is None
    
    def test_get_all_info(self):
        infos = ModelRegistry.get_all_info()
        assert len(infos) > 0
        assert any(i['name'] == 'Random Forest' for i in infos)


class TestRandomForestClassifier:
    """随机森林测试"""
    
    def test_info(self):
        model = RandomForestClassifier()
        assert model.info.name == "Random Forest"
        assert model.info.type == ModelType.ENSEMBLE
    
    def test_fit(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = RandomForestClassifier(n_estimators=10)
        result = model.fit(X_train, y_train, X_test, y_test)
        assert result.success
        assert model.is_fitted
    
    def test_predict(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
    
    def test_predict_proba(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape[0] == len(X_test)
    
    def test_feature_importance(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = RandomForestClassifier(n_estimators=10)
        result = model.fit(X_train, y_train)
        assert result.feature_importance is not None
        assert 'f1' in result.feature_importance


class TestOtherClassifiers:
    """其他分类器测试"""
    
    @pytest.mark.parametrize("model_name,model_class", [
        ("gradient_boosting", GradientBoostingClassifier),
        ("adaboost", AdaBoostClassifier),
        ("extra_trees", ExtraTreesClassifier),
        ("logistic_regression", LogisticRegressionClassifier),
        ("knn", KNNClassifier),
        ("naive_bayes", NaiveBayesClassifier),
        ("decision_tree", DecisionTreeClassifier),
    ])
    def test_classifier_fit_predict(self, model_name, model_class, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = model_class()
        result = model.fit(X_train, y_train)
        assert result.success
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)


class TestSVMClassifier:
    """SVM分类器测试"""
    
    def test_svm_fit(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = SVMClassifier(C=1.0)
        result = model.fit(X_train, y_train)
        assert result.success
    
    def test_svm_predict_proba(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = SVMClassifier()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape[0] == len(X_test)


class TestAnomalyDetectors:
    """异常检测器测试"""
    
    def test_isolation_forest(self, sample_data):
        X, _ = sample_data
        model = IsolationForestDetector(n_estimators=10)
        result = model.fit(X)
        assert result.success
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({-1, 1})
    
    def test_one_class_svm(self, sample_data):
        X, _ = sample_data
        model = OneClassSVMDetector()
        result = model.fit(X)
        assert result.success
        preds = model.predict(X)
        assert len(preds) == len(X)
    
    def test_lof(self, sample_data):
        X, _ = sample_data
        model = LOFDetector(n_neighbors=5)
        result = model.fit(X)
        assert result.success
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestModelSaveLoad:
    """模型保存加载测试"""
    
    def test_save_load(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train, y_train)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        
        try:
            assert model.save(path)
            new_model = RandomForestClassifier()
            assert new_model.load(path)
            assert new_model.is_fitted
            preds = new_model.predict(X_test)
            assert len(preds) == len(X_test)
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestTrainingResult:
    """训练结果测试"""
    
    def test_result_creation(self):
        result = TrainingResult(
            success=True,
            model_name="Test",
            train_time=1.5,
            train_score=0.95
        )
        assert result.success
        assert result.train_time == 1.5
    
    def test_result_to_dict(self):
        result = TrainingResult(success=True, model_name="Test")
        d = result.to_dict()
        assert d['success'] == True
        assert d['model_name'] == "Test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

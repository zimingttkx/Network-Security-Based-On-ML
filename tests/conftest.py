"""
Pytest配置文件
定义测试fixtures和配置
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """创建示例测试数据"""
    np.random.seed(42)
    n_samples = 100

    data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randint(0, 10, n_samples),
        'feature_4': np.random.uniform(0, 1, n_samples),
        'Result': np.random.randint(0, 2, n_samples)
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_path(tmp_path, sample_data):
    """创建临时CSV文件"""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def project_root():
    """获取项目根目录"""
    return Path(__file__).parent.parent

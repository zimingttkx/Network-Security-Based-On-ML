"""
数据摄取模块测试
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig


class TestDataIngestion:
    """数据摄取测试类"""

    @pytest.fixture
    def training_config(self):
        """创建训练配置"""
        return TrainingPipelineConfig()

    @pytest.fixture
    def data_ingestion_config(self, training_config):
        """创建数据摄取配置"""
        return DataIngestionConfig(training_config)

    @pytest.fixture
    def data_ingestion(self, data_ingestion_config):
        """创建数据摄取实例"""
        return DataIngestion(data_ingestion_config)

    def test_export_data_into_feature_store(self, data_ingestion, sample_data, tmp_path):
        """测试导出数据到特征存储"""
        # 修改配置路径为临时目录
        data_ingestion.data_ingestion_config.feature_store_file_path = tmp_path / "feature_store.csv"

        # 执行导出
        result = data_ingestion.export_data_into_feature_store(sample_data)

        # 验证
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert (tmp_path / "feature_store.csv").exists()

    def test_split_data_as_train_test(self, data_ingestion, sample_data, tmp_path):
        """测试训练测试集分割"""
        # 修改配置路径
        data_ingestion.data_ingestion_config.training_file_path = tmp_path / "train.csv"
        data_ingestion.data_ingestion_config.testing_file_path = tmp_path / "test.csv"

        # 执行分割
        data_ingestion.split_data_as_train_test(sample_data)

        # 验证
        assert (tmp_path / "train.csv").exists()
        assert (tmp_path / "test.csv").exists()

        train_df = pd.read_csv(tmp_path / "train.csv")
        test_df = pd.read_csv(tmp_path / "test.csv")

        assert len(train_df) + len(test_df) == len(sample_data)
        assert len(test_df) / len(sample_data) == pytest.approx(0.2, rel=0.1)

    @patch('networksecurity.components.data_ingestion.pymongo.MongoClient')
    def test_export_collection_as_dataframe(self, mock_mongo, data_ingestion, sample_data):
        """测试从MongoDB导出数据"""
        # Mock MongoDB
        mock_collection = MagicMock()
        mock_collection.find.return_value = sample_data.to_dict('records')

        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        mock_mongo.return_value = mock_client

        # 执行
        result = data_ingestion.export_collection_as_dataframe()

        # 验证
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @patch('networksecurity.components.data_ingestion.pymongo.MongoClient')
    def test_export_collection_empty_data(self, mock_mongo, data_ingestion):
        """测试空数据库集合"""
        # Mock空集合
        mock_collection = MagicMock()
        mock_collection.find.return_value = []

        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        mock_client = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        mock_mongo.return_value = mock_client

        # 验证抛出异常
        with pytest.raises(Exception):
            data_ingestion.export_collection_as_dataframe()

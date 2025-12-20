"""
确保从数据摄取 数据转换 模型验证 模型评估 模型推送者

"""
import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# 确保所有组件都被正确导入
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

# 导入深度学习训练器
try:
    from networksecurity.components.dl_model_trainer import DLModelTrainer
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logging.warning("深度学习模块不可用")

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig, DataIngestionConfig,
    DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact, DataValidationArtifact,
    DataTransformationArtifact, ModelTrainerArtifact
)


class TrainingPipeline:
    def __init__(self, use_deep_learning=False, dl_model_type='dnn', dl_config=None):
        """
        初始化训练管道
        :param use_deep_learning: 是否使用深度学习
        :param dl_model_type: 深度学习模型类型 ('dnn', 'cnn', 'lstm')
        :param dl_config: 深度学习模型配置
        """
        self.training_pipeline_config = TrainingPipelineConfig()
        self.use_deep_learning = use_deep_learning
        self.dl_model_type = dl_model_type
        self.dl_config = dl_config

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        开始数据摄取
        :return: 返回数据摄取组件的产出物 (Artifact)
        """
        try:
            # 创建数据摄取配置
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("开始进行数据摄取")
            # 创建数据摄取对象
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            # 初始化数据摄取
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"数据摄取完成, 生成的 Data_ingestion_artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        这个函数用来进行数据验证
        :param data_ingestion_artifact:数据摄取函数的返回值 作为参数传入下一个数据处理步骤
        :return: 返回数据验证组件的产出物 (Artifact)
        """
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            logging.info("开始进行数据验证")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"数据验证完成, 生成的 Data_validation_artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        这个函数用来进行数据转换
        :param data_validation_artifact: 数据验证组件作为返回值传给数据转换的参数
        :return: 返回数据转化的组件的产出物 (Artifact)
        """
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config)
            logging.info("开始进行数据转化")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"数据转化完成, 生成的 Data_transformation_artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        这个函数用来返回模型训练组件
        :param data_transformation_artifact: 这个是数据转化的组件
        :return: 返回模型训练的组件
        """
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)

            # 根据配置选择使用机器学习或深度学习
            if self.use_deep_learning:
                if not DL_AVAILABLE:
                    logging.error("深度学习功能不可用，请安装TensorFlow")
                    raise ImportError("深度学习功能不可用，请先安装: pip install tensorflow keras")

                logging.info(f"开始进行深度学习模型训练 - 模型类型: {self.dl_model_type}")
                model_trainer = DLModelTrainer(
                    model_trainer_config=model_trainer_config,
                    data_transformation_artifact=data_transformation_artifact
                )
                model_trainer_artifact = model_trainer.initiate_model_trainer(
                    model_type=self.dl_model_type,
                    config=self.dl_config
                )
                logging.info(f"深度学习模型训练完成, 生成的 Model_trainer_artifact: {model_trainer_artifact}")
            else:
                logging.info("开始进行机器学习模型训练")
                model_trainer = ModelTrainer(
                    model_trainer_config=model_trainer_config,
                    data_transformation_artifact=data_transformation_artifact
                )
                model_trainer_artifact = model_trainer.initiate_model_trainer()
                logging.info(f"机器学习模型训练完成, 生成的 Model_trainer_artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def run_pipeline(self):
        """
        这个函数用来运行整个训练管道
        :return:
        """
        try:
            logging.info("====== 训练管道启动 ======")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info("====== 训练管道运行结束 ======")

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
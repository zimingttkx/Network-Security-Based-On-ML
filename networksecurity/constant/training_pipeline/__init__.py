import os
import sys
import numpy as np
import pandas as pd

"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "Result"
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "phisingData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


# schema文件存放整个架构 用来对比数据是否完整
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"


"""
数据摄取相关常量以 DATA_INGESTION VAR NAME 开头
"""
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
数据验证相关常量以 DATA_VALIDATION VAR NAME 开头
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

"""
数据转换通常以 DATA_TRANSFORMATION VAR NAME 开头
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

## KNN填充器的参数设置
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan, # 填充nan值
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"


"""
模型训练器通常以 MODE TRAINER VAR NAME 开头
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer" # 模型训练目录
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model" # 训练好的模型存放目录
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl" # 训练好的模型文件名
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6 # 预期得分
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05 # 过拟合和欠拟合的阈值

TRAINING_BUCKET_NAME = "networksecurity"
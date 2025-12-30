import sys,os
from platform import processor
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object
class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    @staticmethod
    def read_data(file_path: str):
        """
        读取指定路径下的数据文件
        :param file_path:  数据文件路径
        :return:  读取的数据
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def get_data_transformer_object(self)-> Pipeline:
        """
        使用在pipeline文件里面指定的参数来初始化KNN 填充器
        Args: cls:DataTransformation
        :return: 返回一个Pipeline对象
        """
        logging.info("进入数据转化类的 get_data_transformer_object 方法")
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS) #** 解包字典将键值对作为参数传给KNN
            logging.info(f"初始化KNN Imputer，参数为：{DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor:Pipeline= Pipeline([("imputer",imputer)])
            return processor

        except Exception as e:
            raise NetworkSecurityException(e,sys) from e


    def initiate_data_transformation(self)-> DataTransformationArtifact:
        logging.info("进入数据转化类的 initiate_data_transformation 方法")
        try:
            logging.info("开始进行数据转换")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            # training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)

            # testing dataframe
            input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            # 使用KNN处理缺失值
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_imput_train_feature= preprocessor_object.transform(input_feature_train_df)
            transformed_imput_test_feature = preprocessor_object.transform(input_feature_test_df)


            # 将结果保存为 numpy 数组格式
            train_arr = np.c_[transformed_imput_train_feature,np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_imput_test_feature,np.array(target_feature_test_df)]

            # 保存转换后的数据
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)

            save_object("models/preprocessor.pkl",preprocessor_object)

            # 准备组件
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)

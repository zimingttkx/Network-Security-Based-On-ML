from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

## configuration of the Data Ingestion Config

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self):
        """
        Read data from mongodb or local CSV file
        """
        try:
            # 如果MongoDB URL未配置或为空，使用本地CSV文件
            if not MONGO_DB_URL or MONGO_DB_URL.strip() == "":
                logging.info("MongoDB未配置，使用本地CSV文件")
                csv_path = "data/phisingData.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df.replace({"na": np.nan}, inplace=True)
                    logging.info(f"成功从本地CSV文件读取 {len(df)} 条记录")
                    return df
                else:
                    raise FileNotFoundError(f"本地数据文件不存在: {csv_path}")

            # 尝试从MongoDB读取
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            # 设置较短的超时时间
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, serverSelectionTimeoutMS=5000)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if df.empty:
                logging.warning("MongoDB中没有数据，尝试使用本地CSV文件")
                csv_path = "data/phisingData.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df.replace({"na": np.nan}, inplace=True)
                    logging.info(f"成功从本地CSV文件读取 {len(df)} 条记录")
                    return df
                else:
                    raise ValueError(
                        f"在数据库 '{database_name}' 的集合 '{collection_name}' 中没有找到任何数据，"
                        f"且本地CSV文件也不存在。"
                    )

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"成功从MongoDB导出 {len(df)} 条记录")
            return df
        except (pymongo.errors.ServerSelectionTimeoutError, pymongo.errors.ConnectionFailure) as e:
            # MongoDB连接失败，使用本地CSV文件
            logging.warning(f"MongoDB连接失败: {e}，使用本地CSV文件")
            csv_path = "data/phisingData.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df.replace({"na": np.nan}, inplace=True)
                logging.info(f"成功从本地CSV文件读取 {len(df)} 条记录")
                return df
            else:
                raise FileNotFoundError(f"MongoDB连接失败且本地数据文件不存在: {csv_path}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42
            )
            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")


        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

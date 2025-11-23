"""
配置管理模块
提供类型安全的配置加载和验证
"""
import os
import yaml
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path


class AppConfig(BaseModel):
    """应用配置"""
    name: str
    version: str
    environment: str
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


class MongoDBConfig(BaseModel):
    """MongoDB配置"""
    database_name: str
    collection_name: str
    connection_timeout: int = 5000
    max_pool_size: int = 50


class DataPipelineConfig(BaseModel):
    """数据管道配置"""
    artifact_dir: str
    pipeline_name: str


class ModelTrainingConfig(BaseModel):
    """模型训练配置"""
    expected_accuracy: float
    overfitting_threshold: float


class APIConfig(BaseModel):
    """API配置"""
    title: str
    description: str
    version: str


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "detailed"


class Config(BaseModel):
    """主配置类"""
    app: AppConfig
    database: Dict[str, MongoDBConfig]
    data_pipeline: DataPipelineConfig
    model_training: ModelTrainingConfig
    api: APIConfig
    logging: LoggingConfig

    class Config:
        arbitrary_types_allowed = True


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，默认为项目根目录的config/config.yaml
        """
        if config_path is None:
            # 获取项目根目录
            root_dir = Path(__file__).parent.parent.parent
            config_path = root_dir / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self._config_dict = self._load_yaml()
        self._merge_env_vars()
        self.config = self._validate_config()

    def _load_yaml(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _merge_env_vars(self):
        """从环境变量合并配置"""
        # MongoDB URL
        mongo_url = os.getenv("MONGO_DB_URL")
        if mongo_url:
            if 'database' not in self._config_dict:
                self._config_dict['database'] = {}
            if 'mongodb' not in self._config_dict['database']:
                self._config_dict['database']['mongodb'] = {}
            self._config_dict['database']['mongodb']['url'] = mongo_url

        # 其他环境变量
        env_mappings = {
            'APP_ENV': ('app', 'environment'),
            'APP_DEBUG': ('app', 'debug'),
            'APP_PORT': ('app', 'port'),
            'LOG_LEVEL': ('logging', 'level'),
        }

        for env_key, config_path in env_mappings.items():
            value = os.getenv(env_key)
            if value:
                self._set_nested_value(self._config_dict, config_path, value)

    def _set_nested_value(self, d: Dict, path: tuple, value: Any):
        """设置嵌套字典的值"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    def _validate_config(self) -> Config:
        """验证配置"""
        try:
            return Config(**self._config_dict)
        except Exception as e:
            raise ValueError(f"配置验证失败: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key_path: 配置键路径，使用点号分隔，如 "app.name"
            default: 默认值

        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self._config_dict

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_mongo_url(self) -> str:
        """获取MongoDB连接URL"""
        url = os.getenv("MONGO_DB_URL")
        if not url:
            raise ValueError("MONGO_DB_URL环境变量未设置")
        return url

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """获取模型参数"""
        return self.get(f"model_training.models.{model_name}.params", {})

    def is_model_enabled(self, model_name: str) -> bool:
        """检查模型是否启用"""
        return self.get(f"model_training.models.{model_name}.enabled", False)

    def get_enabled_models(self) -> List[str]:
        """获取所有启用的模型"""
        models = self.get("model_training.models", {})
        return [name for name, config in models.items() if config.get('enabled', False)]


# 全局配置实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reload_config():
    """重新加载配置"""
    global _config_manager
    _config_manager = ConfigManager()
    return _config_manager

"""
配置管理测试
"""
import pytest
import os
from pathlib import Path
from networksecurity.config.config_manager import ConfigManager


class TestConfigManager:
    """配置管理器测试类"""

    def test_config_loading(self):
        """测试配置加载"""
        config_manager = ConfigManager()

        assert config_manager is not None
        assert config_manager.config is not None
        assert config_manager.config.app is not None

    def test_get_config_value(self):
        """测试获取配置值"""
        config_manager = ConfigManager()

        app_name = config_manager.get('app.name')
        assert app_name is not None
        assert isinstance(app_name, str)

    def test_get_with_default(self):
        """测试带默认值的配置获取"""
        config_manager = ConfigManager()

        value = config_manager.get('nonexistent.key', 'default')
        assert value == 'default'

    def test_get_enabled_models(self):
        """测试获取启用的模型"""
        config_manager = ConfigManager()

        enabled_models = config_manager.get_enabled_models()
        assert isinstance(enabled_models, list)
        assert len(enabled_models) > 0

    def test_is_model_enabled(self):
        """测试检查模型是否启用"""
        config_manager = ConfigManager()

        # 假设至少有一个模型启用
        is_enabled = config_manager.is_model_enabled('xgboost')
        assert isinstance(is_enabled, bool)

    def test_env_var_override(self, monkeypatch):
        """测试环境变量覆盖"""
        # 设置环境变量
        monkeypatch.setenv('APP_ENV', 'testing')
        monkeypatch.setenv('APP_DEBUG', 'true')

        config_manager = ConfigManager()

        env = config_manager.get('app.environment')
        # 环境变量应该覆盖配置文件
        assert env == 'testing' or env == 'development'

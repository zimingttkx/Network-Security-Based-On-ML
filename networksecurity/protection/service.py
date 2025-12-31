"""
一键保护服务
提供类似VPN的一键开启/关闭防护功能
"""

import asyncio
import threading
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProtectionStatus(str, Enum):
    """保护状态"""
    OFF = "off"
    STARTING = "starting"
    ON = "on"
    STOPPING = "stopping"
    ERROR = "error"


class ProtectionLevel(str, Enum):
    """保护级别"""
    LOW = "low"           # 仅监控，不拦截
    MEDIUM = "medium"     # 拦截高风险
    HIGH = "high"         # 拦截中高风险
    STRICT = "strict"     # 严格模式，拦截所有可疑


@dataclass
class ProtectionStats:
    """保护统计"""
    start_time: Optional[datetime] = None
    total_requests: int = 0
    blocked_requests: int = 0          # 本系统二层拦截数
    allowed_requests: int = 0
    threats_detected: int = 0
    server_blocked: int = 0            # 服务器一层拦截数
    second_layer_blocked: int = 0      # 二层额外拦截数
    last_threat_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "allowed_requests": self.allowed_requests,
            "threats_detected": self.threats_detected,
            "server_blocked": self.server_blocked,
            "second_layer_blocked": self.second_layer_blocked,
            "block_rate": round(self.blocked_requests / max(self.total_requests, 1) * 100, 2),
            "last_threat_time": self.last_threat_time.isoformat() if self.last_threat_time else None
        }


class ProtectionService:
    """一键保护服务 - 单例模式"""
    
    _instance: Optional['ProtectionService'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._status = ProtectionStatus.OFF
        self._level = ProtectionLevel.MEDIUM
        self._stats = ProtectionStats()
        self._config = {
            "auto_block": True,
            "log_all_traffic": True,
            "alert_on_threat": True,
            "captcha_on_suspicious": True,
            "rate_limit_enabled": True,
            "rate_limit_per_minute": 100,
            "whitelist_ips": [],
            "blacklist_ips": [],
        }
        self._models_loaded = False
        self._monitor_task: Optional[asyncio.Task] = None
        logger.info("ProtectionService initialized")
    
    @property
    def status(self) -> ProtectionStatus:
        return self._status
    
    @property
    def level(self) -> ProtectionLevel:
        return self._level
    
    @property
    def is_active(self) -> bool:
        return self._status == ProtectionStatus.ON
    
    async def start(self, level: Optional[ProtectionLevel] = None) -> Dict[str, Any]:
        """启动保护服务"""
        if self._status == ProtectionStatus.ON:
            return {"success": True, "message": "保护已开启", "status": self._status.value}
        
        if self._status == ProtectionStatus.STARTING:
            return {"success": False, "message": "正在启动中...", "status": self._status.value}
        
        try:
            self._status = ProtectionStatus.STARTING
            logger.info("Starting protection service...")
            
            if level:
                self._level = level
            
            # 1. 加载检测模型
            await self._load_models()
            
            # 2. 初始化统计
            self._stats = ProtectionStats(start_time=datetime.now())
            
            # 3. 启动监控任务
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            self._status = ProtectionStatus.ON
            logger.info(f"Protection service started with level: {self._level.value}")
            
            return {
                "success": True,
                "message": f"保护已开启 - {self._get_level_description()}",
                "status": self._status.value,
                "level": self._level.value
            }
        except Exception as e:
            self._status = ProtectionStatus.ERROR
            logger.error(f"Failed to start protection: {e}")
            return {"success": False, "message": f"启动失败: {str(e)}", "status": self._status.value}
    
    async def stop(self) -> Dict[str, Any]:
        """停止保护服务"""
        if self._status == ProtectionStatus.OFF:
            return {"success": True, "message": "保护已关闭", "status": self._status.value}
        
        try:
            self._status = ProtectionStatus.STOPPING
            logger.info("Stopping protection service...")
            
            # 停止监控任务
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
                self._monitor_task = None
            
            self._status = ProtectionStatus.OFF
            logger.info("Protection service stopped")
            
            return {
                "success": True,
                "message": "保护已关闭",
                "status": self._status.value,
                "stats": self._stats.to_dict()
            }
        except Exception as e:
            self._status = ProtectionStatus.ERROR
            logger.error(f"Failed to stop protection: {e}")
            return {"success": False, "message": f"停止失败: {str(e)}", "status": self._status.value}
    
    async def toggle(self) -> Dict[str, Any]:
        """切换保护状态"""
        if self._status == ProtectionStatus.ON:
            return await self.stop()
        else:
            return await self.start()
    
    def set_level(self, level: ProtectionLevel) -> Dict[str, Any]:
        """设置保护级别"""
        old_level = self._level
        self._level = level
        logger.info(f"Protection level changed: {old_level.value} -> {level.value}")
        return {
            "success": True,
            "message": f"保护级别已设置为: {self._get_level_description()}",
            "level": level.value
        }
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "status": self._status.value,
            "is_active": self.is_active,
            "level": self._level.value,
            "level_description": self._get_level_description(),
            "config": self._config,
            "stats": self._stats.to_dict() if self._stats.start_time else None,
            "models_loaded": self._models_loaded
        }
    
    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """更新配置"""
        for key, value in config.items():
            if key in self._config:
                self._config[key] = value
        return {"success": True, "config": self._config}
    
    def record_request(self, is_threat: bool, blocked: bool):
        """记录请求统计"""
        if not self.is_active:
            return
        
        self._stats.total_requests += 1
        if blocked:
            self._stats.blocked_requests += 1
        else:
            self._stats.allowed_requests += 1
        
        if is_threat:
            self._stats.threats_detected += 1
            self._stats.last_threat_time = datetime.now()
    
    async def _load_models(self):
        """加载检测模型"""
        await asyncio.sleep(0.5)  # 模拟加载
        self._models_loaded = True
        logger.info("Detection models loaded")
    
    async def _monitor_loop(self):
        """监控循环"""
        while True:
            try:
                await asyncio.sleep(5)
                # 可以在这里添加定期检查逻辑
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
    
    def _get_level_description(self) -> str:
        """获取级别描述"""
        descriptions = {
            ProtectionLevel.LOW: "低级 - 仅监控记录",
            ProtectionLevel.MEDIUM: "中级 - 拦截高风险威胁",
            ProtectionLevel.HIGH: "高级 - 拦截中高风险威胁",
            ProtectionLevel.STRICT: "严格 - 拦截所有可疑流量"
        }
        return descriptions.get(self._level, "未知")


# 全局实例
_protection_service: Optional[ProtectionService] = None


def get_protection_service() -> ProtectionService:
    """获取保护服务实例"""
    global _protection_service
    if _protection_service is None:
        _protection_service = ProtectionService()
    return _protection_service

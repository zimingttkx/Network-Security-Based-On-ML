"""
保护服务API路由
提供一键开启/关闭防护的REST API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from networksecurity.protection.service import (
    get_protection_service, 
    ProtectionLevel,
    ProtectionStatus
)

protection_router = APIRouter(prefix="/api/v1/protection", tags=["Protection"])


class ToggleRequest(BaseModel):
    level: Optional[str] = None


class LevelRequest(BaseModel):
    level: str


class ConfigRequest(BaseModel):
    config: Dict[str, Any]


@protection_router.get("/state")
async def get_state():
    """获取保护状态"""
    service = get_protection_service()
    return service.get_state()


@protection_router.post("/start")
async def start_protection(request: Optional[ToggleRequest] = None):
    """启动保护服务"""
    service = get_protection_service()
    level = None
    if request and request.level:
        try:
            level = ProtectionLevel(request.level)
        except ValueError:
            raise HTTPException(400, f"无效的保护级别: {request.level}")
    return await service.start(level)


@protection_router.post("/stop")
async def stop_protection():
    """停止保护服务"""
    service = get_protection_service()
    return await service.stop()


@protection_router.post("/toggle")
async def toggle_protection():
    """切换保护状态"""
    service = get_protection_service()
    return await service.toggle()


@protection_router.post("/level")
async def set_level(request: LevelRequest):
    """设置保护级别"""
    service = get_protection_service()
    try:
        level = ProtectionLevel(request.level)
    except ValueError:
        raise HTTPException(400, f"无效的保护级别: {request.level}")
    return service.set_level(level)


@protection_router.post("/config")
async def update_config(request: ConfigRequest):
    """更新配置"""
    service = get_protection_service()
    return service.update_config(request.config)


@protection_router.get("/health")
async def health():
    """健康检查"""
    service = get_protection_service()
    return {
        "status": "healthy",
        "protection_active": service.is_active,
        "level": service.level.value
    }

"""
Protection Service Module
一键保护服务模块
"""

from networksecurity.protection.service import (
    ProtectionService,
    ProtectionStatus,
    ProtectionLevel,
    ProtectionStats,
    get_protection_service
)

from networksecurity.protection.api import protection_router

__all__ = [
    'ProtectionService',
    'ProtectionStatus', 
    'ProtectionLevel',
    'ProtectionStats',
    'get_protection_service',
    'protection_router'
]

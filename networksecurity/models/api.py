"""
模型选择API
提供模型列表、组合配置等功能
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging

from networksecurity.models.pretrained import (
    get_model_manager, ModelCategory, AttackType
)

logger = logging.getLogger(__name__)

model_router = APIRouter(prefix="/api/v1/models", tags=["Models"])


class ModelCombination(BaseModel):
    """模型组合配置"""
    name: str = Field(..., description="组合名称")
    ml_models: List[str] = Field(default=[], description="机器学习模型ID列表")
    dl_models: List[str] = Field(default=[], description="深度学习模型ID列表")
    anomaly_models: List[str] = Field(default=[], description="异常检测模型ID列表")
    rl_agent: Optional[str] = Field(None, description="强化学习代理ID")
    voting_strategy: str = Field("majority", description="投票策略: majority/weighted/any")
    threshold: float = Field(0.5, description="威胁判定阈值")


# 存储用户配置的模型组合
_active_combinations: Dict[str, ModelCombination] = {}


@model_router.get("/list")
async def list_all_models(category: Optional[str] = None):
    """列出所有可用模型"""
    manager = get_model_manager()
    cat = ModelCategory(category) if category else None
    models = manager.list_models(cat)
    return {"success": True, "models": models, "total": len(models)}


@model_router.get("/categories")
async def get_categories():
    """获取模型类别"""
    manager = get_model_manager()
    return {"success": True, "categories": manager.get_categories()}


@model_router.get("/recommend")
async def recommend_models(attack_type: str = "all"):
    """根据攻击类型推荐模型"""
    manager = get_model_manager()
    try:
        at = AttackType(attack_type)
    except ValueError:
        at = AttackType.ALL
    models = manager.get_models_by_attack(at)
    return {"success": True, "attack_type": attack_type, "recommended": models}


@model_router.get("/info/{model_id}")
async def get_model_info(model_id: str):
    """获取模型详情"""
    manager = get_model_manager()
    info = manager.get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="模型不存在")
    return {"success": True, "model": info}


@model_router.post("/combination")
async def create_combination(config: ModelCombination):
    """创建模型组合"""
    _active_combinations[config.name] = config
    return {"success": True, "message": f"模型组合 '{config.name}' 已创建", "config": config.dict()}


@model_router.get("/combinations")
async def list_combinations():
    """列出所有模型组合"""
    return {"success": True, "combinations": [c.dict() for c in _active_combinations.values()]}


@model_router.delete("/combination/{name}")
async def delete_combination(name: str):
    """删除模型组合"""
    if name in _active_combinations:
        del _active_combinations[name]
        return {"success": True, "message": f"已删除组合 '{name}'"}
    raise HTTPException(status_code=404, detail="组合不存在")


@model_router.get("/presets")
async def get_presets():
    """获取预设的模型组合"""
    presets = [
        {
            "id": "balanced",
            "name": "均衡防护",
            "description": "ML+DL+RL组合，平衡检测率和性能",
            "ml_models": ["rf_nslkdd", "xgb_nslkdd"],
            "dl_models": ["dnn_cicids"],
            "anomaly_models": ["isolation_forest"],
            "rl_agent": "dqn_firewall",
            "voting_strategy": "majority"
        },
        {
            "id": "high_security",
            "name": "高安全模式",
            "description": "多模型投票，最大化检测率",
            "ml_models": ["rf_nslkdd", "xgb_nslkdd"],
            "dl_models": ["dnn_cicids", "lstm_flow"],
            "anomaly_models": ["autoencoder_anomaly", "isolation_forest"],
            "rl_agent": "ppo_security",
            "voting_strategy": "any"
        },
        {
            "id": "fast",
            "name": "快速模式",
            "description": "单模型检测，最小延迟",
            "ml_models": ["xgb_nslkdd"],
            "dl_models": [],
            "anomaly_models": [],
            "rl_agent": None,
            "voting_strategy": "majority"
        },
        {
            "id": "anomaly_focus",
            "name": "异常检测优先",
            "description": "专注于检测未知攻击",
            "ml_models": [],
            "dl_models": [],
            "anomaly_models": ["autoencoder_anomaly", "isolation_forest"],
            "rl_agent": "dqn_firewall",
            "voting_strategy": "any"
        }
    ]
    return {"success": True, "presets": presets}

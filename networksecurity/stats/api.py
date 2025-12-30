"""
统计API路由
提供流量统计和日志查询的REST API
"""

from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from networksecurity.stats.models import (
    TrafficLog, ThreatType, ActionType, RiskLevel,
    GeoLocation, ModelPrediction
)
from networksecurity.stats.traffic_logger import TrafficLogger
from networksecurity.stats.aggregator import StatsAggregator


# 创建路由器
router = APIRouter(prefix="/api/v1/stats", tags=["Statistics"])

# 全局实例
_logger: Optional[TrafficLogger] = None
_aggregator: Optional[StatsAggregator] = None


def get_logger() -> TrafficLogger:
    """获取日志记录器单例"""
    global _logger
    if _logger is None:
        _logger = TrafficLogger()
    return _logger


def get_aggregator() -> StatsAggregator:
    """获取聚合器单例"""
    global _aggregator
    if _aggregator is None:
        _aggregator = StatsAggregator(get_logger())
    return _aggregator


# --- Pydantic 模型 ---

class TrafficLogInput(BaseModel):
    """流量日志输入模型"""
    source_ip: str = ""
    source_port: int = 0
    dest_ip: str = ""
    dest_port: int = 0
    protocol: str = "HTTP"
    method: str = "GET"
    url: str = ""
    user_agent: str = ""
    threat_type: str = "benign"
    risk_level: str = "safe"
    risk_score: float = 0.0
    action: str = "allow"
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None
    processing_time_ms: float = 0.0


class TrafficLogResponse(BaseModel):
    """流量日志响应模型"""
    id: str
    timestamp: str
    source_ip: str
    dest_ip: str
    dest_port: int
    url: str
    threat_type: str
    risk_level: str
    risk_score: float
    action: str


class StatsOverviewResponse(BaseModel):
    """统计概览响应模型"""
    total_requests: int
    blocked_requests: int
    allowed_requests: int
    challenged_requests: int
    threat_counts: dict
    action_counts: dict
    risk_level_counts: dict
    top_source_ips: list
    top_threat_types: list
    avg_risk_score: float
    avg_processing_time_ms: float


# --- API 端点 ---

@router.get("/overview", response_model=StatsOverviewResponse)
async def get_stats_overview(
    hours: int = Query(24, ge=1, le=720, description="统计时间范围（小时）")
):
    """
    获取流量统计概览
    
    - **hours**: 统计时间范围，默认24小时
    """
    try:
        aggregator = get_aggregator()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        stats = aggregator.get_overview(start_time=start_time, end_time=end_time)
        
        return StatsOverviewResponse(
            total_requests=stats.total_requests,
            blocked_requests=stats.blocked_requests,
            allowed_requests=stats.allowed_requests,
            challenged_requests=stats.challenged_requests,
            threat_counts=stats.threat_counts,
            action_counts=stats.action_counts,
            risk_level_counts=stats.risk_level_counts,
            top_source_ips=stats.top_source_ips,
            top_threat_types=stats.top_threat_types,
            avg_risk_score=stats.avg_risk_score,
            avg_processing_time_ms=stats.avg_processing_time_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threats")
async def get_threat_distribution(
    hours: int = Query(24, ge=1, le=720)
):
    """获取威胁类型分布"""
    try:
        aggregator = get_aggregator()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        distribution = aggregator.get_threat_distribution(
            start_time=start_time, 
            end_time=end_time
        )
        
        return {"success": True, "data": distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/actions")
async def get_action_distribution(
    hours: int = Query(24, ge=1, le=720)
):
    """获取处理动作分布"""
    try:
        aggregator = get_aggregator()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        distribution = aggregator.get_action_distribution(
            start_time=start_time,
            end_time=end_time
        )
        
        return {"success": True, "data": distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def get_top_sources(
    hours: int = Query(24, ge=1, le=720),
    limit: int = Query(10, ge=1, le=100),
    threat_only: bool = Query(False, description="仅统计威胁流量")
):
    """获取TOP源IP"""
    try:
        aggregator = get_aggregator()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        top_ips = aggregator.get_top_source_ips(
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            threat_only=threat_only
        )
        
        return {
            "success": True,
            "data": [{"ip": ip, "count": count} for ip, count in top_ips]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline")
async def get_timeline(
    hours: int = Query(24, ge=1, le=720),
    interval: int = Query(60, ge=5, le=1440, description="时间间隔（分钟）")
):
    """获取时间线统计"""
    try:
        aggregator = get_aggregator()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        timeline = aggregator.get_timeline(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval
        )
        
        return {"success": True, "data": timeline}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geo")
async def get_geo_distribution(
    hours: int = Query(24, ge=1, le=720)
):
    """获取地理位置分布"""
    try:
        aggregator = get_aggregator()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        distribution = aggregator.get_geo_distribution(
            start_time=start_time,
            end_time=end_time
        )
        
        return {"success": True, "data": distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent-threats")
async def get_recent_threats(
    limit: int = Query(20, ge=1, le=100)
):
    """获取最近的威胁日志"""
    try:
        aggregator = get_aggregator()
        threats = aggregator.get_recent_threats(limit=limit)
        
        return {
            "success": True,
            "data": [log.to_dict() for log in threats]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-distribution")
async def get_risk_distribution(
    hours: int = Query(24, ge=1, le=720),
    bins: int = Query(10, ge=5, le=20)
):
    """获取风险分数分布"""
    try:
        aggregator = get_aggregator()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        distribution = aggregator.get_risk_score_distribution(
            start_time=start_time,
            end_time=end_time,
            bins=bins
        )
        
        return {"success": True, "data": distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(
    hours: int = Query(24, ge=1, le=720),
    source_ip: Optional[str] = Query(None),
    threat_type: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    min_risk_score: Optional[float] = Query(None, ge=0, le=1),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """查询流量日志"""
    try:
        logger = get_logger()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 转换枚举类型
        threat_type_enum = ThreatType(threat_type) if threat_type else None
        action_enum = ActionType(action) if action else None
        
        logs = logger.query(
            start_time=start_time,
            end_time=end_time,
            source_ip=source_ip,
            threat_type=threat_type_enum,
            action=action_enum,
            min_risk_score=min_risk_score,
            limit=limit,
            offset=offset
        )
        
        total = logger.count(
            start_time=start_time,
            end_time=end_time,
            threat_type=threat_type_enum,
            action=action_enum
        )
        
        return {
            "success": True,
            "data": [log.to_dict() for log in logs],
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效的参数: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logs")
async def create_log(log_input: TrafficLogInput):
    """创建流量日志"""
    try:
        logger = get_logger()
        
        # 创建地理位置
        geo_location = None
        if log_input.geo_country:
            geo_location = GeoLocation(
                country=log_input.geo_country,
                city=log_input.geo_city or "Unknown"
            )
        
        # 创建日志对象
        traffic_log = TrafficLog(
            source_ip=log_input.source_ip,
            source_port=log_input.source_port,
            dest_ip=log_input.dest_ip,
            dest_port=log_input.dest_port,
            protocol=log_input.protocol,
            method=log_input.method,
            url=log_input.url,
            user_agent=log_input.user_agent,
            threat_type=ThreatType(log_input.threat_type),
            risk_level=RiskLevel(log_input.risk_level),
            risk_score=log_input.risk_score,
            action=ActionType(log_input.action),
            geo_location=geo_location,
            processing_time_ms=log_input.processing_time_ms
        )
        
        log_id = logger.log(traffic_log)
        
        return {"success": True, "id": log_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效的参数: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/{log_id}")
async def get_log_by_id(log_id: str):
    """根据ID获取日志"""
    try:
        logger = get_logger()
        log = logger.get_by_id(log_id)
        
        if log is None:
            raise HTTPException(status_code=404, detail="日志不存在")
        
        return {"success": True, "data": log.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/logs/cleanup")
async def cleanup_old_logs(
    days: int = Query(30, ge=1, le=365, description="删除多少天前的日志")
):
    """清理旧日志"""
    try:
        logger = get_logger()
        deleted = logger.delete_old_logs(days=days)
        
        return {
            "success": True,
            "message": f"已删除 {deleted} 条 {days} 天前的日志"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demo-data")
async def generate_demo_data(
    count: int = Query(500, ge=10, le=5000, description="生成数据条数"),
    hours_back: int = Query(24, ge=1, le=168, description="数据时间范围（小时）")
):
    """生成演示数据用于测试"""
    try:
        from networksecurity.stats.demo_data import generate_demo_data as gen_data
        logger = get_logger()
        generated = gen_data(logger, count=count, hours_back=hours_back)
        
        return {
            "success": True,
            "message": f"已生成 {generated} 条演示数据",
            "count": generated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

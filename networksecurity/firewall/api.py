"""
防火墙API路由
提供威胁检测和CAPTCHA验证的REST API
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging

from networksecurity.firewall.detector import ThreatDetector, DetectionResult, ThreatLevel, ActionType
from networksecurity.firewall.captcha import CaptchaService, CaptchaType

logger = logging.getLogger(__name__)

firewall_router = APIRouter(prefix="/api/v1/firewall", tags=["firewall"])

# 全局实例
_detector = ThreatDetector()
_captcha_service = CaptchaService()


def get_detector() -> ThreatDetector:
    return _detector


def get_captcha_service() -> CaptchaService:
    return _captcha_service


# 请求/响应模型
class DetectRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="特征字典")
    model_name: Optional[str] = Field(None, description="指定模型名称")


class DetectResponse(BaseModel):
    is_threat: bool
    threat_level: str
    confidence: float
    action: str
    model_name: str
    threat_type: str
    detection_time: float


class BatchDetectRequest(BaseModel):
    features_list: List[Dict[str, float]]
    model_name: Optional[str] = None


class CaptchaRequest(BaseModel):
    challenge_type: str = "math"


class CaptchaResponse(BaseModel):
    challenge_id: str
    challenge_type: str
    question: str
    expires_at: float


class VerifyRequest(BaseModel):
    challenge_id: str
    answer: str


class VerifyResponse(BaseModel):
    success: bool
    message: str


# API端点
@firewall_router.post("/detect", response_model=DetectResponse)
async def detect_threat(request: DetectRequest, detector: ThreatDetector = Depends(get_detector)):
    """检测单个请求的威胁"""
    result = detector.detect(request.features, request.model_name)
    return DetectResponse(
        is_threat=result.is_threat,
        threat_level=result.threat_level.value,
        confidence=result.confidence,
        action=result.action.value,
        model_name=result.model_name,
        threat_type=result.threat_type,
        detection_time=result.detection_time
    )


@firewall_router.post("/detect/batch")
async def detect_batch(request: BatchDetectRequest, detector: ThreatDetector = Depends(get_detector)):
    """批量检测威胁"""
    results = detector.detect_batch(request.features_list, request.model_name)
    return {"results": [r.to_dict() for r in results]}


@firewall_router.get("/models")
async def list_models(detector: ThreatDetector = Depends(get_detector)):
    """列出可用模型"""
    return detector.get_model_info()


@firewall_router.post("/captcha/generate", response_model=CaptchaResponse)
async def generate_captcha(request: CaptchaRequest, req: Request,
                          captcha: CaptchaService = Depends(get_captcha_service)):
    """生成验证码"""
    try:
        challenge_type = CaptchaType(request.challenge_type)
    except ValueError:
        challenge_type = CaptchaType.MATH
    
    client_ip = req.client.host if req.client else None
    challenge = captcha.generate_challenge(challenge_type, client_ip)
    
    return CaptchaResponse(
        challenge_id=challenge.challenge_id,
        challenge_type=challenge.challenge_type.value,
        question=challenge.question,
        expires_at=challenge.expires_at
    )


@firewall_router.post("/captcha/verify", response_model=VerifyResponse)
async def verify_captcha(request: VerifyRequest, captcha: CaptchaService = Depends(get_captcha_service)):
    """验证答案"""
    success = captcha.verify_challenge(request.challenge_id, request.answer)
    return VerifyResponse(
        success=success,
        message="验证成功" if success else "验证失败或已过期"
    )


@firewall_router.get("/captcha/stats")
async def captcha_stats(captcha: CaptchaService = Depends(get_captcha_service)):
    """获取验证码统计"""
    return captcha.get_stats()


@firewall_router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "firewall"}

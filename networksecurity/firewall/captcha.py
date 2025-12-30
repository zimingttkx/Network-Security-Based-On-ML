"""
CAPTCHA验证服务
提供人机验证功能
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum
import hashlib
import secrets
import time
import logging

logger = logging.getLogger(__name__)


class CaptchaType(str, Enum):
    """验证码类型"""
    MATH = "math"
    TEXT = "text"
    IMAGE = "image"
    RECAPTCHA = "recaptcha"


@dataclass
class CaptchaChallenge:
    """验证码挑战"""
    challenge_id: str
    challenge_type: CaptchaType
    question: str
    answer_hash: str
    created_at: float
    expires_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def verify(self, answer: str) -> bool:
        if self.is_expired():
            return False
        answer_hash = hashlib.sha256(answer.lower().strip().encode()).hexdigest()
        return answer_hash == self.answer_hash
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'challenge_id': self.challenge_id,
            'challenge_type': self.challenge_type.value,
            'question': self.question,
            'expires_at': self.expires_at
        }


class CaptchaService:
    """CAPTCHA服务"""
    
    def __init__(self, expiry_seconds: int = 300):
        self.expiry_seconds = expiry_seconds
        self.challenges: Dict[str, CaptchaChallenge] = {}
        self._cleanup_interval = 60
        self._last_cleanup = time.time()
    
    def generate_challenge(self, challenge_type: CaptchaType = CaptchaType.MATH,
                          client_ip: str = None) -> CaptchaChallenge:
        """生成验证码挑战"""
        self._cleanup_expired()
        
        challenge_id = secrets.token_urlsafe(16)
        now = time.time()
        
        if challenge_type == CaptchaType.MATH:
            question, answer = self._generate_math_challenge()
        elif challenge_type == CaptchaType.TEXT:
            question, answer = self._generate_text_challenge()
        else:
            question, answer = self._generate_math_challenge()
        
        answer_hash = hashlib.sha256(answer.lower().strip().encode()).hexdigest()
        
        challenge = CaptchaChallenge(
            challenge_id=challenge_id,
            challenge_type=challenge_type,
            question=question,
            answer_hash=answer_hash,
            created_at=now,
            expires_at=now + self.expiry_seconds,
            metadata={'client_ip': client_ip}
        )
        
        self.challenges[challenge_id] = challenge
        logger.info(f"生成验证码: {challenge_id}")
        return challenge
    
    def verify_challenge(self, challenge_id: str, answer: str) -> bool:
        """验证答案"""
        challenge = self.challenges.get(challenge_id)
        if not challenge:
            logger.warning(f"验证码不存在: {challenge_id}")
            return False
        
        result = challenge.verify(answer)
        
        # 验证后删除（一次性使用）
        del self.challenges[challenge_id]
        
        logger.info(f"验证码验证{'成功' if result else '失败'}: {challenge_id}")
        return result
    
    def _generate_math_challenge(self) -> tuple:
        """生成数学验证码"""
        import random
        ops = ['+', '-', '*']
        op = random.choice(ops)
        
        if op == '+':
            a, b = random.randint(1, 50), random.randint(1, 50)
            answer = str(a + b)
        elif op == '-':
            a, b = random.randint(10, 50), random.randint(1, 10)
            answer = str(a - b)
        else:
            a, b = random.randint(2, 10), random.randint(2, 10)
            answer = str(a * b)
        
        question = f"{a} {op} {b} = ?"
        return question, answer
    
    def _generate_text_challenge(self) -> tuple:
        """生成文本验证码"""
        chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
        code = ''.join(secrets.choice(chars) for _ in range(6))
        question = f"请输入验证码: {code}"
        return question, code
    
    def _cleanup_expired(self):
        """清理过期验证码"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        expired = [cid for cid, c in self.challenges.items() if c.is_expired()]
        for cid in expired:
            del self.challenges[cid]
        
        self._last_cleanup = now
        if expired:
            logger.info(f"清理过期验证码: {len(expired)}个")
    
    def get_challenge(self, challenge_id: str) -> Optional[CaptchaChallenge]:
        """获取验证码（不含答案）"""
        return self.challenges.get(challenge_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'active_challenges': len(self.challenges),
            'expiry_seconds': self.expiry_seconds
        }

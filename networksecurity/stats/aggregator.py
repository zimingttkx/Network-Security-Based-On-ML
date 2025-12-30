"""
统计聚合器
负责聚合和分析流量日志数据
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from networksecurity.stats.models import (
    TrafficLog, ThreatType, ActionType, RiskLevel, TrafficStats
)
from networksecurity.stats.traffic_logger import TrafficLogger


class StatsAggregator:
    """
    统计聚合器
    提供各种维度的流量统计分析
    """
    
    def __init__(self, logger: TrafficLogger):
        """
        初始化聚合器
        
        Args:
            logger: 流量日志记录器实例
        """
        self.logger = logger
    
    def get_overview(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> TrafficStats:
        """
        获取流量统计概览
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            TrafficStats对象
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        logs = self.logger.query(
            start_time=start_time,
            end_time=end_time,
            limit=100000
        )
        
        stats = TrafficStats(
            time_range_start=start_time,
            time_range_end=end_time
        )
        
        if not logs:
            return stats
        
        # 基础统计
        stats.total_requests = len(logs)
        
        # 按动作统计
        action_counts = defaultdict(int)
        threat_counts = defaultdict(int)
        risk_level_counts = defaultdict(int)
        ip_counts = defaultdict(int)
        
        total_risk_score = 0.0
        total_processing_time = 0.0
        
        for log in logs:
            action_counts[log.action.value] += 1
            threat_counts[log.threat_type.value] += 1
            risk_level_counts[log.risk_level.value] += 1
            ip_counts[log.source_ip] += 1
            total_risk_score += log.risk_score
            total_processing_time += log.processing_time_ms
        
        stats.blocked_requests = action_counts.get('block', 0)
        stats.allowed_requests = action_counts.get('allow', 0)
        stats.challenged_requests = action_counts.get('challenge', 0)
        
        stats.action_counts = dict(action_counts)
        stats.threat_counts = dict(threat_counts)
        stats.risk_level_counts = dict(risk_level_counts)
        
        # TOP IP
        stats.top_source_ips = sorted(
            ip_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # TOP 威胁类型
        stats.top_threat_types = sorted(
            [(k, v) for k, v in threat_counts.items() if k != 'benign'],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # 平均值
        stats.avg_risk_score = total_risk_score / len(logs)
        stats.avg_processing_time_ms = total_processing_time / len(logs)
        
        return stats
    
    def get_threat_distribution(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, int]:
        """获取威胁类型分布"""
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        logs = self.logger.query(start_time=start_time, end_time=end_time, limit=100000)
        
        distribution = defaultdict(int)
        for log in logs:
            distribution[log.threat_type.value] += 1
        
        return dict(distribution)
    
    def get_action_distribution(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, int]:
        """获取处理动作分布"""
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        logs = self.logger.query(start_time=start_time, end_time=end_time, limit=100000)
        
        distribution = defaultdict(int)
        for log in logs:
            distribution[log.action.value] += 1
        
        return dict(distribution)
    
    def get_top_source_ips(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        threat_only: bool = False
    ) -> List[Tuple[str, int]]:
        """
        获取TOP源IP
        
        Args:
            threat_only: 是否只统计威胁流量
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        logs = self.logger.query(start_time=start_time, end_time=end_time, limit=100000)
        
        ip_counts = defaultdict(int)
        for log in logs:
            if threat_only and log.threat_type == ThreatType.BENIGN:
                continue
            ip_counts[log.source_ip] += 1
        
        return sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_timeline(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        获取时间线统计
        
        Args:
            interval_minutes: 时间间隔（分钟）
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        logs = self.logger.query(start_time=start_time, end_time=end_time, limit=100000)
        
        # 按时间间隔分组
        interval = timedelta(minutes=interval_minutes)
        buckets = defaultdict(lambda: {'total': 0, 'blocked': 0, 'threats': 0})
        
        for log in logs:
            # 计算所属时间桶
            bucket_time = start_time + (
                (log.timestamp - start_time) // interval
            ) * interval
            bucket_key = bucket_time.isoformat()
            
            buckets[bucket_key]['total'] += 1
            if log.action == ActionType.BLOCK:
                buckets[bucket_key]['blocked'] += 1
            if log.threat_type != ThreatType.BENIGN:
                buckets[bucket_key]['threats'] += 1
        
        # 转换为列表并排序
        timeline = [
            {'timestamp': k, **v}
            for k, v in sorted(buckets.items())
        ]
        
        return timeline
    
    def get_geo_distribution(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, int]:
        """获取地理位置分布"""
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        logs = self.logger.query(start_time=start_time, end_time=end_time, limit=100000)
        
        country_counts = defaultdict(int)
        for log in logs:
            if log.geo_location:
                country_counts[log.geo_location.country] += 1
            else:
                country_counts['Unknown'] += 1
        
        return dict(country_counts)
    
    def get_recent_threats(
        self,
        limit: int = 20
    ) -> List[TrafficLog]:
        """获取最近的威胁日志"""
        # 查询非正常流量
        all_logs = self.logger.query(limit=limit * 5, order_desc=True)
        
        threats = [
            log for log in all_logs
            if log.threat_type != ThreatType.BENIGN
        ]
        
        return threats[:limit]
    
    def get_risk_score_distribution(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        bins: int = 10
    ) -> List[Dict[str, Any]]:
        """获取风险分数分布"""
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        logs = self.logger.query(start_time=start_time, end_time=end_time, limit=100000)
        
        # 创建分数区间
        bin_size = 1.0 / bins
        distribution = []
        
        for i in range(bins):
            low = i * bin_size
            high = (i + 1) * bin_size
            count = sum(1 for log in logs if low <= log.risk_score < high)
            distribution.append({
                'range': f'{low:.1f}-{high:.1f}',
                'low': low,
                'high': high,
                'count': count
            })
        
        return distribution

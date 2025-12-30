"""
流量日志记录器
负责记录、存储和查询流量日志
"""

import os
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager

from networksecurity.stats.models import (
    TrafficLog, ThreatType, ActionType, RiskLevel,
    GeoLocation, ModelPrediction
)


class TrafficLogger:
    """
    流量日志记录器
    使用SQLite存储日志，支持高并发写入和查询
    """
    
    def __init__(self, db_path: str = None):
        """
        初始化日志记录器
        
        Args:
            db_path: 数据库文件路径，默认为 logs/traffic.db
        """
        if db_path is None:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            db_path = str(logs_dir / "traffic.db")
        
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地的数据库连接"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    @contextmanager
    def _get_cursor(self):
        """获取数据库游标的上下文管理器"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def _init_database(self):
        """初始化数据库表"""
        with self._get_cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_logs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    source_ip TEXT,
                    source_port INTEGER,
                    dest_ip TEXT,
                    dest_port INTEGER,
                    protocol TEXT,
                    method TEXT,
                    url TEXT,
                    user_agent TEXT,
                    threat_type TEXT,
                    risk_level TEXT,
                    risk_score REAL,
                    action TEXT,
                    predictions TEXT,
                    ensemble_score REAL,
                    geo_country TEXT,
                    geo_city TEXT,
                    geo_latitude REAL,
                    geo_longitude REAL,
                    features TEXT,
                    metadata TEXT,
                    captcha_required INTEGER,
                    captcha_passed INTEGER,
                    processing_time_ms REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引以加速查询
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON traffic_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_ip ON traffic_logs(source_ip)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_threat_type ON traffic_logs(threat_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_action ON traffic_logs(action)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_level ON traffic_logs(risk_level)')
    
    def log(self, traffic_log: TrafficLog) -> str:
        """
        记录一条流量日志
        
        Args:
            traffic_log: 流量日志对象
            
        Returns:
            日志ID
        """
        with self._get_cursor() as cursor:
            cursor.execute('''
                INSERT INTO traffic_logs (
                    id, timestamp, source_ip, source_port, dest_ip, dest_port,
                    protocol, method, url, user_agent, threat_type, risk_level,
                    risk_score, action, predictions, ensemble_score,
                    geo_country, geo_city, geo_latitude, geo_longitude,
                    features, metadata, captcha_required, captcha_passed,
                    processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                traffic_log.id,
                traffic_log.timestamp.isoformat(),
                traffic_log.source_ip,
                traffic_log.source_port,
                traffic_log.dest_ip,
                traffic_log.dest_port,
                traffic_log.protocol,
                traffic_log.method,
                traffic_log.url,
                traffic_log.user_agent,
                traffic_log.threat_type.value,
                traffic_log.risk_level.value,
                traffic_log.risk_score,
                traffic_log.action.value,
                json.dumps([p.to_dict() if isinstance(p, ModelPrediction) else p for p in traffic_log.predictions]),
                traffic_log.ensemble_score,
                traffic_log.geo_location.country if traffic_log.geo_location else None,
                traffic_log.geo_location.city if traffic_log.geo_location else None,
                traffic_log.geo_location.latitude if traffic_log.geo_location else None,
                traffic_log.geo_location.longitude if traffic_log.geo_location else None,
                json.dumps(traffic_log.features),
                json.dumps(traffic_log.metadata),
                1 if traffic_log.captcha_required else 0,
                1 if traffic_log.captcha_passed else 0,
                traffic_log.processing_time_ms
            ))
        
        return traffic_log.id
    
    def log_batch(self, logs: List[TrafficLog]) -> int:
        """
        批量记录流量日志
        
        Args:
            logs: 流量日志列表
            
        Returns:
            成功记录的数量
        """
        count = 0
        with self._get_cursor() as cursor:
            for log in logs:
                try:
                    cursor.execute('''
                        INSERT INTO traffic_logs (
                            id, timestamp, source_ip, source_port, dest_ip, dest_port,
                            protocol, method, url, user_agent, threat_type, risk_level,
                            risk_score, action, predictions, ensemble_score,
                            geo_country, geo_city, geo_latitude, geo_longitude,
                            features, metadata, captcha_required, captcha_passed,
                            processing_time_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        log.id, log.timestamp.isoformat(), log.source_ip, log.source_port,
                        log.dest_ip, log.dest_port, log.protocol, log.method, log.url,
                        log.user_agent, log.threat_type.value, log.risk_level.value,
                        log.risk_score, log.action.value,
                        json.dumps([p.to_dict() if isinstance(p, ModelPrediction) else p for p in log.predictions]),
                        log.ensemble_score,
                        log.geo_location.country if log.geo_location else None,
                        log.geo_location.city if log.geo_location else None,
                        log.geo_location.latitude if log.geo_location else None,
                        log.geo_location.longitude if log.geo_location else None,
                        json.dumps(log.features), json.dumps(log.metadata),
                        1 if log.captcha_required else 0, 1 if log.captcha_passed else 0,
                        log.processing_time_ms
                    ))
                    count += 1
                except Exception:
                    continue
        return count
    
    def _row_to_traffic_log(self, row: sqlite3.Row) -> TrafficLog:
        """将数据库行转换为TrafficLog对象"""
        geo_location = None
        if row['geo_country']:
            geo_location = GeoLocation(
                country=row['geo_country'] or "Unknown",
                city=row['geo_city'] or "Unknown",
                latitude=row['geo_latitude'] or 0.0,
                longitude=row['geo_longitude'] or 0.0
            )
        
        predictions_data = json.loads(row['predictions']) if row['predictions'] else []
        predictions = [ModelPrediction.from_dict(p) if isinstance(p, dict) else p for p in predictions_data]
        
        return TrafficLog(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            source_ip=row['source_ip'] or "",
            source_port=row['source_port'] or 0,
            dest_ip=row['dest_ip'] or "",
            dest_port=row['dest_port'] or 0,
            protocol=row['protocol'] or "HTTP",
            method=row['method'] or "GET",
            url=row['url'] or "",
            user_agent=row['user_agent'] or "",
            threat_type=ThreatType(row['threat_type']) if row['threat_type'] else ThreatType.BENIGN,
            risk_level=RiskLevel(row['risk_level']) if row['risk_level'] else RiskLevel.SAFE,
            risk_score=row['risk_score'] or 0.0,
            action=ActionType(row['action']) if row['action'] else ActionType.ALLOW,
            predictions=predictions,
            ensemble_score=row['ensemble_score'] or 0.0,
            geo_location=geo_location,
            features=json.loads(row['features']) if row['features'] else {},
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            captcha_required=bool(row['captcha_required']),
            captcha_passed=bool(row['captcha_passed']),
            processing_time_ms=row['processing_time_ms'] or 0.0
        )
    
    def get_by_id(self, log_id: str) -> Optional[TrafficLog]:
        """根据ID获取日志"""
        with self._get_cursor() as cursor:
            cursor.execute('SELECT * FROM traffic_logs WHERE id = ?', (log_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_traffic_log(row)
        return None
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        source_ip: Optional[str] = None,
        threat_type: Optional[ThreatType] = None,
        action: Optional[ActionType] = None,
        risk_level: Optional[RiskLevel] = None,
        min_risk_score: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "timestamp",
        order_desc: bool = True
    ) -> List[TrafficLog]:
        """
        查询流量日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            source_ip: 源IP
            threat_type: 威胁类型
            action: 处理动作
            risk_level: 风险等级
            min_risk_score: 最小风险分数
            limit: 返回数量限制
            offset: 偏移量
            order_by: 排序字段
            order_desc: 是否降序
            
        Returns:
            流量日志列表
        """
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        if source_ip:
            conditions.append("source_ip = ?")
            params.append(source_ip)
        if threat_type:
            conditions.append("threat_type = ?")
            params.append(threat_type.value)
        if action:
            conditions.append("action = ?")
            params.append(action.value)
        if risk_level:
            conditions.append("risk_level = ?")
            params.append(risk_level.value)
        if min_risk_score is not None:
            conditions.append("risk_score >= ?")
            params.append(min_risk_score)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order_direction = "DESC" if order_desc else "ASC"
        
        query = f'''
            SELECT * FROM traffic_logs 
            WHERE {where_clause}
            ORDER BY {order_by} {order_direction}
            LIMIT ? OFFSET ?
        '''
        params.extend([limit, offset])
        
        results = []
        with self._get_cursor() as cursor:
            cursor.execute(query, params)
            for row in cursor.fetchall():
                results.append(self._row_to_traffic_log(row))
        
        return results
    
    def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        threat_type: Optional[ThreatType] = None,
        action: Optional[ActionType] = None
    ) -> int:
        """统计日志数量"""
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        if threat_type:
            conditions.append("threat_type = ?")
            params.append(threat_type.value)
        if action:
            conditions.append("action = ?")
            params.append(action.value)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_cursor() as cursor:
            cursor.execute(f'SELECT COUNT(*) FROM traffic_logs WHERE {where_clause}', params)
            return cursor.fetchone()[0]
    
    def delete_old_logs(self, days: int = 30) -> int:
        """删除指定天数之前的日志"""
        cutoff = datetime.now() - timedelta(days=days)
        with self._get_cursor() as cursor:
            cursor.execute('DELETE FROM traffic_logs WHERE timestamp < ?', (cutoff.isoformat(),))
            return cursor.rowcount
    
    def clear_all(self) -> int:
        """清空所有日志（谨慎使用）"""
        with self._get_cursor() as cursor:
            cursor.execute('DELETE FROM traffic_logs')
            return cursor.rowcount
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

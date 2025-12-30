"""
测试整合的GitHub开源算法
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKitsune:
    """测试Kitsune模块"""
    
    def test_afterimage_import(self):
        """测试AfterImage导入"""
        from networksecurity.models.kitsune.afterimage import AfterImage, IncStat, IncStatDB
        assert AfterImage is not None
        assert IncStat is not None
    
    def test_incstat_basic(self):
        """测试增量统计基本功能"""
        from networksecurity.models.kitsune.afterimage import IncStat
        stat = IncStat(lambda_=1.0)
        stat.insert(10.0, timestamp=1.0)
        stat.insert(20.0, timestamp=2.0)
        stat.insert(30.0, timestamp=3.0)
        
        assert stat.weight > 0
        assert stat.mean() > 0
    
    def test_afterimage_features(self):
        """测试AfterImage特征提取"""
        from networksecurity.models.kitsune.afterimage import AfterImage
        ai = AfterImage()
        
        features = ai.update_get_stats(
            src_mac="00:11:22:33:44:55",
            dst_mac="66:77:88:99:aa:bb",
            src_ip="192.168.1.1",
            dst_ip="10.0.0.1",
            src_port=12345,
            dst_port=80,
            packet_size=1024,
            timestamp=1.0
        )
        
        assert features.shape == (115,)
        assert ai.packet_count == 1
    
    def test_kitnet_import(self):
        """测试KitNET导入"""
        from networksecurity.models.kitsune.kitnet import KitNET, AutoEncoder
        assert KitNET is not None
        assert AutoEncoder is not None
    
    def test_autoencoder_basic(self):
        """测试自编码器基本功能"""
        from networksecurity.models.kitsune.kitnet import AutoEncoder
        ae = AutoEncoder(input_dim=10, hidden_ratio=0.5)
        
        x = np.random.randn(10)
        rmse = ae.train_step(x)
        assert rmse >= 0
    
    def test_kitsune_process(self):
        """测试Kitsune处理"""
        from networksecurity.models.kitsune import Kitsune
        
        kitsune = Kitsune(fm_grace_period=10, ad_grace_period=20)
        
        # 处理一些数据包
        for i in range(50):
            data = np.random.randn(115)
            result = kitsune.process(data)
            assert result is not None
        
        state = kitsune.get_state()
        assert state['packet_count'] == 50


class TestLUCID:
    """测试LUCID模块"""
    
    def test_lucid_import(self):
        """测试LUCID导入"""
        from networksecurity.models.lucid import LucidCNN, LucidDatasetParser, LucidDetector
        assert LucidCNN is not None
        assert LucidDatasetParser is not None
        assert LucidDetector is not None
    
    def test_dataset_parser(self):
        """测试数据集解析器"""
        from networksecurity.models.lucid.dataset_parser import LucidDatasetParser, FlowSample
        
        parser = LucidDatasetParser(time_window=10.0, packets_per_flow=5)
        
        # 模拟数据包
        packets = []
        for i in range(20):
            packets.append({
                'src_ip': '192.168.1.1',
                'dst_ip': '10.0.0.1',
                'src_port': 12345,
                'dst_port': 80,
                'protocol': 6,
                'packet_size': 500 + i * 10,
                'timestamp': i * 0.1
            })
        
        X, y = parser.parse_batch(packets)
        assert X.shape[1] == 5  # packets_per_flow
        assert X.shape[2] == 11  # n_features
    
    def test_lucid_detector(self):
        """测试LUCID检测器"""
        from networksecurity.models.lucid import LucidDetector
        
        detector = LucidDetector(time_window=5.0, packets_per_flow=5)
        
        # 处理数据包
        for i in range(10):
            packet = {
                'src_ip': '192.168.1.1',
                'dst_ip': '10.0.0.1',
                'src_port': 12345,
                'dst_port': 80,
                'packet_size': 500,
                'timestamp': i * 0.5
            }
            result = detector.process_packet(packet)
        
        stats = detector.get_stats()
        assert stats['total_packets'] == 10


class TestSlips:
    """测试Slips模块"""
    
    def test_slips_import(self):
        """测试Slips导入"""
        from networksecurity.models.slips import (
            BehaviorAnalyzer, ThreatIntelligence, FlowAnalyzer, SlipsDetector
        )
        assert BehaviorAnalyzer is not None
        assert ThreatIntelligence is not None
    
    def test_behavior_analyzer(self):
        """测试行为分析器"""
        from networksecurity.models.slips.behavior_analyzer import BehaviorAnalyzer
        
        analyzer = BehaviorAnalyzer()
        
        # 模拟流量
        for i in range(100):
            flow = {
                'src_ip': '192.168.1.1',
                'dst_ip': f'10.0.0.{i % 10}',
                'src_port': 12345,
                'dst_port': 80 + i,
                'protocol': 6,
                'bytes_sent': 1000,
                'bytes_recv': 500,
                'packets_sent': 10,
                'packets_recv': 5,
                'timestamp': i
            }
            scores = analyzer.analyze_flow(flow)
            assert 'port_scan' in scores
    
    def test_threat_intelligence(self):
        """测试威胁情报"""
        from networksecurity.models.slips.threat_intelligence import ThreatIntelligence, ThreatCategory
        
        ti = ThreatIntelligence()
        
        # 查询IP
        rep = ti.query_ip('8.8.8.8')
        assert rep.score == 1.0  # 白名单
        
        # 添加黑名单
        ti.add_to_blacklist('1.2.3.4', ThreatCategory.MALWARE)
        rep = ti.query_ip('1.2.3.4')
        assert rep.is_malicious()
    
    def test_slips_detector(self):
        """测试Slips检测器"""
        from networksecurity.models.slips import SlipsDetector
        
        detector = SlipsDetector(threat_threshold=0.5)
        
        # 处理正常流量
        result = detector.process_packet({
            'src_ip': '192.168.1.1',
            'dst_ip': '8.8.8.8',
            'src_port': 12345,
            'dst_port': 53,
            'packet_size': 100,
            'timestamp': 1.0
        })
        
        assert result is not None
        assert hasattr(result, 'is_threat')


class TestRLSecurity:
    """测试RL安全模块"""
    
    def test_rl_import(self):
        """测试RL导入"""
        from networksecurity.models.rl_security import (
            NetworkSecurityEnv, SecurityState, SecurityAction,
            DQNAgent, PPOAgent, DoubleDQNAgent
        )
        assert NetworkSecurityEnv is not None
        assert DQNAgent is not None
    
    def test_security_env(self):
        """测试安全环境"""
        from networksecurity.models.rl_security.environment import NetworkSecurityEnv, SecurityAction
        
        env = NetworkSecurityEnv(episode_length=100)
        
        state, info = env.reset()
        assert state.shape == (15,)
        
        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(SecurityAction.BLOCK)
        assert next_state.shape == (15,)
        assert isinstance(reward, float)
    
    def test_reward_calculator(self):
        """测试奖励计算器"""
        from networksecurity.models.rl_security.reward import RewardCalculator
        
        calc = RewardCalculator()
        
        # 正确阻断攻击
        reward = calc.calculate(action=1, is_attack=True, threat_score=0.8)
        assert reward > 0
        
        # 误报
        reward = calc.calculate(action=1, is_attack=False, threat_score=0.2)
        assert reward < 0


class TestPipeline:
    """测试统一流水线"""
    
    def test_pipeline_import(self):
        """测试流水线导入"""
        from networksecurity.models.pipeline import (
            UnifiedPreprocessor, ModelAdapter, UnifiedDetector
        )
        assert UnifiedPreprocessor is not None
        assert UnifiedDetector is not None
    
    def test_preprocessor(self):
        """测试预处理器"""
        from networksecurity.models.pipeline.preprocessor import UnifiedPreprocessor, OutputFormat
        
        preprocessor = UnifiedPreprocessor()
        
        # 测试原始数据包
        packet = {
            'src_ip': '192.168.1.1',
            'dst_ip': '10.0.0.1',
            'src_port': 12345,
            'dst_port': 80,
            'packet_size': 1024
        }
        
        result = preprocessor.preprocess(packet, OutputFormat.UNIFIED)
        assert result.features.shape == (50,)
        
        # 测试Kitsune格式
        result = preprocessor.preprocess(packet, OutputFormat.KITSUNE)
        assert result.features.shape == (115,)
    
    def test_unified_detector(self):
        """测试统一检测器"""
        from networksecurity.models.pipeline import UnifiedDetector
        from networksecurity.models.slips import SlipsDetector
        
        detector = UnifiedDetector(mode="cascade")
        
        # 添加Slips检测器
        slips = SlipsDetector()
        detector.add_detector("slips", slips, "slips")
        
        # 检测
        result = detector.detect({
            'src_ip': '192.168.1.1',
            'dst_ip': '10.0.0.1',
            'src_port': 12345,
            'dst_port': 80,
            'packet_size': 1024
        })
        
        assert result is not None
        assert hasattr(result, 'is_threat')
        assert hasattr(result, 'threat_score')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

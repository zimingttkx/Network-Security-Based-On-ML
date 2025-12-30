#!/usr/bin/env python
"""
算法功能演示脚本
验证整合的GitHub开源算法是否正常运行
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_kitsune():
    """测试Kitsune算法"""
    print("\n" + "="*60)
    print("1. 测试 Kitsune (AfterImage + KitNET)")
    print("="*60)
    
    from networksecurity.models.kitsune import Kitsune
    
    # 创建Kitsune实例 (减少训练期以便快速演示)
    kitsune = Kitsune(fm_grace_period=100, ad_grace_period=200)
    
    print("   生成模拟流量数据...")
    
    # 模拟正常流量 (训练期)
    normal_data = []
    for i in range(350):
        features = np.random.randn(115) * 0.5 + 0.5  # 正常流量特征
        normal_data.append(features)
    
    print("   训练阶段 (300个样本)...")
    start = time.time()
    for data in normal_data[:300]:
        result = kitsune.process(data)
    train_time = time.time() - start
    
    print(f"   训练完成! 耗时: {train_time:.3f}秒")
    print(f"   模型状态: {kitsune.get_state()}")
    
    # 测试正常流量
    print("\n   测试正常流量 (50个样本)...")
    normal_scores = []
    for data in normal_data[300:]:
        result = kitsune.process(data)
        normal_scores.append(result.rmse)
    
    # 模拟攻击流量
    print("   测试攻击流量 (50个样本)...")
    attack_scores = []
    for i in range(50):
        # 攻击流量: 异常特征
        attack_data = np.random.randn(115) * 3 + 5
        result = kitsune.process(attack_data)
        attack_scores.append(result.rmse)
    
    print(f"\n   结果:")
    print(f"   - 正常流量 RMSE: 均值={np.mean(normal_scores):.4f}, 标准差={np.std(normal_scores):.4f}")
    print(f"   - 攻击流量 RMSE: 均值={np.mean(attack_scores):.4f}, 标准差={np.std(attack_scores):.4f}")
    print(f"   - 阈值: {kitsune.kitnet.threshold:.4f}")
    
    # 检测率
    if kitsune.kitnet.threshold:
        normal_detected = sum(1 for s in normal_scores if s > kitsune.kitnet.threshold)
        attack_detected = sum(1 for s in attack_scores if s > kitsune.kitnet.threshold)
        print(f"   - 正常流量误报率: {normal_detected/50*100:.1f}%")
        print(f"   - 攻击流量检测率: {attack_detected/50*100:.1f}%")
    
    return True


def test_lucid():
    """测试LUCID CNN"""
    print("\n" + "="*60)
    print("2. 测试 LUCID CNN (DDoS检测)")
    print("="*60)
    
    from networksecurity.models.lucid import LucidDetector
    
    detector = LucidDetector(time_window=5.0, packets_per_flow=10)
    
    print("   生成模拟数据包...")
    
    # 模拟正常流量
    normal_packets = []
    for i in range(100):
        normal_packets.append({
            'src_ip': f'192.168.1.{i % 10}',
            'dst_ip': '10.0.0.1',
            'src_port': 10000 + i,
            'dst_port': 80,
            'protocol': 6,
            'packet_size': np.random.randint(200, 1400),
            'timestamp': i * 0.1,
            'tcp_flags': 0x10  # ACK
        })
    
    # 模拟DDoS流量
    ddos_packets = []
    for i in range(100):
        ddos_packets.append({
            'src_ip': f'10.{i%256}.{i%256}.{i%256}',  # 大量不同源IP
            'dst_ip': '192.168.1.1',
            'src_port': np.random.randint(1024, 65535),
            'dst_port': 80,
            'protocol': 6,
            'packet_size': 40,  # 小包
            'timestamp': i * 0.001,  # 高速率
            'tcp_flags': 0x02  # SYN
        })
    
    print("   处理正常流量...")
    for pkt in normal_packets:
        detector.process_packet(pkt)
    
    print("   处理DDoS流量...")
    for pkt in ddos_packets:
        detector.process_packet(pkt)
    
    stats = detector.get_stats()
    print(f"\n   结果:")
    print(f"   - 总数据包: {stats['total_packets']}")
    print(f"   - 检测次数: {stats['total_detections']}")
    print(f"   - 模型状态: {'已训练' if stats['is_trained'] else '未训练(需要训练数据)'}")
    
    return True


def test_slips():
    """测试Slips行为分析"""
    print("\n" + "="*60)
    print("3. 测试 Slips (行为分析)")
    print("="*60)
    
    from networksecurity.models.slips import SlipsDetector
    
    detector = SlipsDetector(threat_threshold=0.3)
    
    print("   模拟正常用户行为...")
    # 正常用户: 访问少量目标
    for i in range(50):
        result = detector.process_packet({
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',  # Google DNS (白名单)
            'src_port': 50000 + i,
            'dst_port': 53,
            'packet_size': 100,
            'timestamp': i
        })
    
    print("   模拟端口扫描行为...")
    # 端口扫描: 访问大量端口
    for i in range(100):
        result = detector.process_packet({
            'src_ip': '10.0.0.100',
            'dst_ip': '192.168.1.1',
            'src_port': 12345,
            'dst_port': i + 1,  # 扫描端口1-100
            'packet_size': 40,
            'timestamp': 100 + i * 0.01
        })
    
    print("   模拟DDoS行为...")
    # DDoS: 高速率大量数据
    for i in range(200):
        result = detector.process_packet({
            'src_ip': '10.0.0.200',
            'dst_ip': '192.168.1.1',
            'src_port': np.random.randint(1024, 65535),
            'dst_port': 80,
            'packet_size': 1400,
            'timestamp': 200 + i * 0.001
        })
    
    stats = detector.get_stats()
    suspicious = detector.get_suspicious_ips(threshold=0.3)
    
    print(f"\n   结果:")
    print(f"   - 总数据包: {stats['total_packets']}")
    print(f"   - 检测到威胁: {stats['total_threats']}")
    print(f"   - 威胁类型: {stats['threat_by_type']}")
    print(f"   - 可疑IP数: {len(suspicious)}")
    
    if suspicious:
        print(f"\n   可疑IP详情:")
        for ip_info in suspicious[:3]:
            print(f"   - {ip_info['ip']}: 威胁分数={ip_info['threat_score']:.2f}, "
                  f"端口扫描={ip_info['port_scan_score']:.2f}, DDoS={ip_info['ddos_score']:.2f}")
    
    return True


def test_rl_security():
    """测试RL安全代理"""
    print("\n" + "="*60)
    print("4. 测试 RL Security (强化学习代理)")
    print("="*60)
    
    from networksecurity.models.rl_security import NetworkSecurityEnv, DQNAgent
    
    env = NetworkSecurityEnv(episode_length=100)
    
    print("   创建DQN代理...")
    agent = DQNAgent(state_dim=15, action_dim=7, epsilon=0.5)
    
    print("   运行一个episode...")
    state, _ = env.reset()
    total_reward = 0
    actions_taken = {i: 0 for i in range(7)}
    
    for step in range(100):
        action = agent.select_action(state)
        actions_taken[action] += 1
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        
        if terminated or truncated:
            break
    
    metrics = env.get_metrics()
    action_names = ['ALLOW', 'BLOCK', 'ALERT', 'LOG', 'CHALLENGE', 'RATE_LIMIT', 'QUARANTINE']
    
    print(f"\n   结果:")
    print(f"   - 总奖励: {total_reward:.2f}")
    print(f"   - 准确率: {metrics['accuracy']*100:.1f}%")
    print(f"   - 精确率: {metrics['precision']*100:.1f}%")
    print(f"   - 召回率: {metrics['recall']*100:.1f}%")
    print(f"   - F1分数: {metrics['f1_score']*100:.1f}%")
    print(f"\n   动作分布:")
    for i, name in enumerate(action_names):
        print(f"   - {name}: {actions_taken[i]}次")
    
    return True


def test_unified_pipeline():
    """测试统一检测流水线"""
    print("\n" + "="*60)
    print("5. 测试统一检测流水线")
    print("="*60)
    
    from networksecurity.models.pipeline import UnifiedDetector, UnifiedPreprocessor
    from networksecurity.models.slips import SlipsDetector
    from networksecurity.models.kitsune import Kitsune
    
    print("   创建统一检测器...")
    detector = UnifiedDetector(mode="cascade")
    
    # 添加检测器
    slips = SlipsDetector(threat_threshold=0.3)
    detector.add_detector("slips", slips, "slips")
    
    kitsune = Kitsune(fm_grace_period=50, ad_grace_period=100)
    detector.add_detector("kitsune", kitsune, "kitsune")
    
    print("   训练Kitsune (150个样本)...")
    for i in range(150):
        detector.detect({'src_ip': '192.168.1.1', 'dst_ip': '10.0.0.1',
                        'src_port': 12345, 'dst_port': 80, 'packet_size': 500})
    
    print("\n   测试正常流量...")
    normal_results = []
    for i in range(20):
        result = detector.detect({
            'src_ip': '192.168.1.1',
            'dst_ip': '8.8.8.8',
            'src_port': 50000 + i,
            'dst_port': 53,
            'packet_size': 100
        })
        normal_results.append(result)
    
    print("   测试可疑流量...")
    suspicious_results = []
    for i in range(20):
        result = detector.detect({
            'src_ip': '10.0.0.100',
            'dst_ip': '192.168.1.1',
            'src_port': 12345,
            'dst_port': i + 1,  # 端口扫描
            'packet_size': 40
        })
        suspicious_results.append(result)
    
    stats = detector.get_stats()
    
    print(f"\n   结果:")
    print(f"   - 总处理: {stats['total_processed']}")
    print(f"   - 检测到威胁: {stats['total_threats']}")
    print(f"   - 威胁率: {stats['threat_rate']*100:.1f}%")
    print(f"   - 平均检测时间: {stats['avg_detection_time_ms']:.2f}ms")
    print(f"   - 使用的检测器: {stats['detectors']}")
    
    # 显示几个检测结果示例
    print(f"\n   检测结果示例:")
    print(f"   - 正常流量: is_threat={normal_results[0].is_threat}, score={normal_results[0].threat_score:.3f}")
    print(f"   - 可疑流量: is_threat={suspicious_results[-1].is_threat}, score={suspicious_results[-1].threat_score:.3f}")
    
    return True


def main():
    print("\n" + "#"*60)
    print("#  网络安全算法功能验证")
    print("#  整合自GitHub开源项目")
    print("#"*60)
    
    results = {}
    
    try:
        results['Kitsune'] = test_kitsune()
    except Exception as e:
        print(f"   错误: {e}")
        results['Kitsune'] = False
    
    try:
        results['LUCID'] = test_lucid()
    except Exception as e:
        print(f"   错误: {e}")
        results['LUCID'] = False
    
    try:
        results['Slips'] = test_slips()
    except Exception as e:
        print(f"   错误: {e}")
        results['Slips'] = False
    
    try:
        results['RL Security'] = test_rl_security()
    except Exception as e:
        print(f"   错误: {e}")
        results['RL Security'] = False
    
    try:
        results['Pipeline'] = test_unified_pipeline()
    except Exception as e:
        print(f"   错误: {e}")
        results['Pipeline'] = False
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, passed in results.items():
        status = "通过" if passed else "失败"
        print(f"   {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n   总体结果: {'全部通过!' if all_passed else '存在失败'}")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

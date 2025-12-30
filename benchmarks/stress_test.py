#!/usr/bin/env python3
"""
高并发压力极限测试脚本
测试系统在极端负载下的稳定性和性能
"""

import requests
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
import random

BASE_URL = "http://localhost:8000"

# 测试结果统计
results = {
    "success": 0,
    "failed": 0,
    "timeout": 0,
    "latencies": [],
    "errors": defaultdict(int)
}
results_lock = threading.Lock()

def generate_features(malicious=True):
    """生成测试特征"""
    if malicious:
        return {
            "having_IP_Address": 1, "URL_Length": 1, "Shortining_Service": 1,
            "having_At_Symbol": 1, "double_slash_redirecting": 1, "Prefix_Suffix": 1,
            "having_Sub_Domain": 1, "SSLfinal_State": -1, "Domain_registeration_length": -1,
            "Favicon": -1, "port": 1, "HTTPS_token": 1, "Request_URL": 1,
            "URL_of_Anchor": 1, "Links_in_tags": 1, "SFH": 1, "Submitting_to_email": 1,
            "Abnormal_URL": 1, "Redirect": 1, "on_mouseover": 1, "RightClick": 1,
            "popUpWidnow": 1, "Iframe": 1, "age_of_domain": -1, "DNSRecord": -1,
            "web_traffic": -1, "Page_Rank": -1, "Google_Index": -1,
            "Links_pointing_to_page": -1, "Statistical_report": 1
        }
    else:
        return {
            "having_IP_Address": -1, "URL_Length": -1, "Shortining_Service": -1,
            "having_At_Symbol": -1, "double_slash_redirecting": -1, "Prefix_Suffix": -1,
            "having_Sub_Domain": -1, "SSLfinal_State": 1, "Domain_registeration_length": 1,
            "Favicon": 1, "port": -1, "HTTPS_token": -1, "Request_URL": -1,
            "URL_of_Anchor": -1, "Links_in_tags": -1, "SFH": -1, "Submitting_to_email": -1,
            "Abnormal_URL": -1, "Redirect": 0, "on_mouseover": -1, "RightClick": -1,
            "popUpWidnow": -1, "Iframe": -1, "age_of_domain": 1, "DNSRecord": 1,
            "web_traffic": 1, "Page_Rank": 1, "Google_Index": 1,
            "Links_pointing_to_page": 1, "Statistical_report": -1
        }

ATTACKER_IPS = ["185.220.101.1", "45.155.205.233", "89.248.167.131", "193.32.162.79", "141.98.10.121"]
ATTACK_TYPES = ["ddos", "sql_injection", "xss", "brute_force", "port_scan"]

def log_to_stats(is_threat=True):
    """记录到统计系统"""
    try:
        attack_type = random.choice(ATTACK_TYPES) if is_threat else "benign"
        log_data = {
            "source_ip": random.choice(ATTACKER_IPS) if is_threat else f"192.168.1.{random.randint(1,254)}",
            "source_port": random.randint(1024, 65535),
            "dest_ip": "127.0.0.1",
            "dest_port": 8000,
            "protocol": "HTTP",
            "method": random.choice(["GET", "POST"]),
            "url": f"/api/test/{random.randint(1,1000)}",
            "user_agent": "StressTest/1.0",
            "threat_type": attack_type,
            "risk_level": "high" if is_threat else "safe",
            "risk_score": random.uniform(0.7, 0.99) if is_threat else random.uniform(0.01, 0.2),
            "action": "block" if is_threat else "allow",
            "processing_time_ms": random.uniform(5, 50)
        }
        requests.post(f"{BASE_URL}/api/v1/stats/logs", json=log_data, timeout=1)
    except:
        pass

def single_request(endpoint, method="GET", data=None, timeout=5, log_stats=True):
    """执行单个请求并记录结果"""
    start = time.time()
    try:
        if method == "GET":
            resp = requests.get(f"{BASE_URL}{endpoint}", timeout=timeout)
        else:
            resp = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=timeout)
        
        latency = (time.time() - start) * 1000  # ms
        
        with results_lock:
            if resp.status_code == 200:
                results["success"] += 1
                results["latencies"].append(latency)
            else:
                results["failed"] += 1
                results["errors"][f"HTTP_{resp.status_code}"] += 1
        
        # 记录到统计系统供前端显示
        if log_stats:
            log_to_stats(is_threat=random.random() > 0.3)
        
        return True, latency
    except requests.exceptions.Timeout:
        with results_lock:
            results["timeout"] += 1
        return False, None
    except Exception as e:
        with results_lock:
            results["failed"] += 1
            results["errors"][type(e).__name__] += 1
        return False, None

def stress_test_predict(num_requests, concurrency):
    """压力测试预测接口"""
    print(f"\n[压测] /predict_live - {num_requests}请求, {concurrency}并发")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            features = generate_features(malicious=random.choice([True, False]))
            futures.append(executor.submit(single_request, "/predict_live", "POST", features))
        
        for f in as_completed(futures):
            pass

def stress_test_health(num_requests, concurrency):
    """压力测试健康检查接口"""
    print(f"\n[压测] /health - {num_requests}请求, {concurrency}并发")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request, "/health") for _ in range(num_requests)]
        for f in as_completed(futures):
            pass

def stress_test_stats(num_requests, concurrency):
    """压力测试统计接口"""
    print(f"\n[压测] /api/v1/stats/overview - {num_requests}请求, {concurrency}并发")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request, "/api/v1/stats/overview?hours=1") for _ in range(num_requests)]
        for f in as_completed(futures):
            pass

def stress_test_firewall(num_requests, concurrency):
    """压力测试防火墙检测接口"""
    print(f"\n[压测] /api/v1/firewall/detect - {num_requests}请求, {concurrency}并发")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for _ in range(num_requests):
            data = {"features": {"packet_rate": random.uniform(100, 10000), "byte_rate": random.uniform(1000, 100000)}}
            futures.append(executor.submit(single_request, "/api/v1/firewall/detect", "POST", data))
        for f in as_completed(futures):
            pass

def print_results(test_name, duration):
    """打印测试结果"""
    total = results["success"] + results["failed"] + results["timeout"]
    
    print(f"\n{'='*60}")
    print(f"  {test_name} - 测试结果")
    print(f"{'='*60}")
    print(f"  总请求数: {total}")
    print(f"  成功: {results['success']} ({results['success']/total*100:.1f}%)")
    print(f"  失败: {results['failed']} ({results['failed']/total*100:.1f}%)")
    print(f"  超时: {results['timeout']} ({results['timeout']/total*100:.1f}%)")
    print(f"  耗时: {duration:.2f}秒")
    print(f"  QPS: {total/duration:.1f} 请求/秒")
    
    if results["latencies"]:
        print(f"\n  [延迟统计]")
        print(f"  最小: {min(results['latencies']):.1f}ms")
        print(f"  最大: {max(results['latencies']):.1f}ms")
        print(f"  平均: {statistics.mean(results['latencies']):.1f}ms")
        print(f"  P50: {statistics.median(results['latencies']):.1f}ms")
        sorted_lat = sorted(results['latencies'])
        p95_idx = int(len(sorted_lat) * 0.95)
        p99_idx = int(len(sorted_lat) * 0.99)
        if p95_idx < len(sorted_lat):
            print(f"  P95: {sorted_lat[p95_idx]:.1f}ms")
        if p99_idx < len(sorted_lat):
            print(f"  P99: {sorted_lat[p99_idx]:.1f}ms")
    
    if results["errors"]:
        print(f"\n  [错误分布]")
        for err, count in results["errors"].items():
            print(f"  {err}: {count}")
    
    print(f"{'='*60}")

def reset_results():
    """重置结果"""
    global results
    results = {"success": 0, "failed": 0, "timeout": 0, "latencies": [], "errors": defaultdict(int)}

def run_extreme_test():
    """运行极限压力测试"""
    print("\n" + "="*60)
    print("       高并发极限压力测试")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 测试1: 健康检查 - 轻量级接口
    reset_results()
    start = time.time()
    stress_test_health(1000, 100)
    print_results("健康检查接口", time.time() - start)
    
    time.sleep(2)
    
    # 测试2: 预测接口 - 核心ML推理
    reset_results()
    start = time.time()
    stress_test_predict(500, 50)
    print_results("ML预测接口", time.time() - start)
    
    time.sleep(2)
    
    # 测试3: 统计接口 - 数据聚合
    reset_results()
    start = time.time()
    stress_test_stats(300, 30)
    print_results("统计聚合接口", time.time() - start)
    
    time.sleep(2)
    
    # 测试4: 防火墙检测 - 实时检测
    reset_results()
    start = time.time()
    stress_test_firewall(500, 50)
    print_results("防火墙检测接口", time.time() - start)
    
    time.sleep(2)
    
    # 测试5: 混合负载极限测试
    print("\n" + "="*60)
    print("  混合负载极限测试 (所有接口同时压测)")
    print("="*60)
    
    reset_results()
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=200) as executor:
        futures = []
        # 混合各种请求
        for _ in range(300):
            futures.append(executor.submit(single_request, "/health"))
        for _ in range(200):
            features = generate_features(malicious=random.choice([True, False]))
            futures.append(executor.submit(single_request, "/predict_live", "POST", features))
        for _ in range(100):
            futures.append(executor.submit(single_request, "/api/v1/stats/overview?hours=1"))
        for _ in range(200):
            data = {"features": {"packet_rate": random.uniform(100, 10000)}}
            futures.append(executor.submit(single_request, "/api/v1/firewall/detect", "POST", data))
        
        for f in as_completed(futures):
            pass
    
    print_results("混合负载极限测试", time.time() - start)
    
    # 测试6: 持续高压测试
    print("\n" + "="*60)
    print("  持续高压测试 (30秒持续压力)")
    print("="*60)
    
    reset_results()
    start = time.time()
    end_time = time.time() + 30  # 30秒
    
    def continuous_load():
        while time.time() < end_time:
            choice = random.randint(1, 4)
            if choice == 1:
                single_request("/health", timeout=2)
            elif choice == 2:
                features = generate_features(malicious=random.choice([True, False]))
                single_request("/predict_live", "POST", features, timeout=3)
            elif choice == 3:
                single_request("/api/v1/stats/overview?hours=1", timeout=3)
            else:
                data = {"features": {"packet_rate": random.uniform(100, 10000)}}
                single_request("/api/v1/firewall/detect", "POST", data, timeout=3)
    
    threads = [threading.Thread(target=continuous_load) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print_results("持续高压测试(30秒)", time.time() - start)
    
    print("\n" + "="*60)
    print("       极限压力测试完成")
    print("="*60)

if __name__ == "__main__":
    run_extreme_test()

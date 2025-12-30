#!/usr/bin/env python3
"""
网络攻击模拟脚本
模拟多种攻击类型来测试网络安全检测系统
"""

import requests
import random
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:8000"

# 攻击类型配置
ATTACK_TYPES = {
    "benign": {"risk_level": "safe", "risk_score": 0.1},
    "ddos": {"risk_level": "critical", "risk_score": 0.95},
    "sql_injection": {"risk_level": "high", "risk_score": 0.85},
    "xss": {"risk_level": "high", "risk_score": 0.8},
    "brute_force": {"risk_level": "high", "risk_score": 0.75},
    "port_scan": {"risk_level": "medium", "risk_score": 0.6},
    "phishing": {"risk_level": "high", "risk_score": 0.88},
}

# 模拟攻击源IP
ATTACKER_IPS = [
    "185.220.101.1", "45.155.205.233", "89.248.167.131",
    "193.32.162.79", "141.98.10.121", "45.129.56.200",
    "185.156.73.54", "194.26.192.64", "45.134.26.174",
]

NORMAL_IPS = [
    "192.168.1.100", "192.168.1.101", "10.0.0.50",
    "172.16.0.25", "192.168.2.200", "10.10.10.10",
]

def generate_malicious_features():
    """生成恶意URL特征"""
    return {
        "having_IP_Address": 1,
        "URL_Length": 1,
        "Shortining_Service": 1,
        "having_At_Symbol": 1,
        "double_slash_redirecting": 1,
        "Prefix_Suffix": 1,
        "having_Sub_Domain": 1,
        "SSLfinal_State": -1,
        "Domain_registeration_length": -1,
        "Favicon": -1,
        "port": 1,
        "HTTPS_token": 1,
        "Request_URL": 1,
        "URL_of_Anchor": 1,
        "Links_in_tags": 1,
        "SFH": 1,
        "Submitting_to_email": 1,
        "Abnormal_URL": 1,
        "Redirect": 1,
        "on_mouseover": 1,
        "RightClick": 1,
        "popUpWidnow": 1,
        "Iframe": 1,
        "age_of_domain": -1,
        "DNSRecord": -1,
        "web_traffic": -1,
        "Page_Rank": -1,
        "Google_Index": -1,
        "Links_pointing_to_page": -1,
        "Statistical_report": 1
    }

def generate_normal_features():
    """生成正常URL特征"""
    return {
        "having_IP_Address": -1,
        "URL_Length": -1,
        "Shortining_Service": -1,
        "having_At_Symbol": -1,
        "double_slash_redirecting": -1,
        "Prefix_Suffix": -1,
        "having_Sub_Domain": -1,
        "SSLfinal_State": 1,
        "Domain_registeration_length": 1,
        "Favicon": 1,
        "port": -1,
        "HTTPS_token": -1,
        "Request_URL": -1,
        "URL_of_Anchor": -1,
        "Links_in_tags": -1,
        "SFH": -1,
        "Submitting_to_email": -1,
        "Abnormal_URL": -1,
        "Redirect": 0,
        "on_mouseover": -1,
        "RightClick": -1,
        "popUpWidnow": -1,
        "Iframe": -1,
        "age_of_domain": 1,
        "DNSRecord": 1,
        "web_traffic": 1,
        "Page_Rank": 1,
        "Google_Index": 1,
        "Links_pointing_to_page": 1,
        "Statistical_report": -1
    }

def log_traffic(attack_type, source_ip, is_threat):
    """记录流量到统计系统"""
    config = ATTACK_TYPES.get(attack_type, ATTACK_TYPES["benign"])
    
    urls = {
        "benign": ["/", "/api/products", "/login", "/about"],
        "ddos": ["/api/search?q=" + "x"*1000, "/api/heavy-query"],
        "sql_injection": ["/api/users?id=1' OR '1'='1", "/login?user=admin'--"],
        "xss": ["/search?q=<script>alert(1)</script>", "/comment?text=<img onerror=alert(1)>"],
        "brute_force": ["/login", "/admin/login", "/api/auth"],
        "port_scan": ["/", "/.env", "/wp-admin", "/.git/config"],
        "phishing": ["/secure-login", "/verify-account", "/update-payment"],
    }
    
    log_data = {
        "source_ip": source_ip,
        "source_port": random.randint(1024, 65535),
        "dest_ip": "127.0.0.1",
        "dest_port": 8000,
        "protocol": "HTTP",
        "method": random.choice(["GET", "POST"]),
        "url": random.choice(urls.get(attack_type, ["/"])),
        "user_agent": "Mozilla/5.0 (Attack Simulator)",
        "threat_type": attack_type,
        "risk_level": config["risk_level"],
        "risk_score": config["risk_score"] + random.uniform(-0.1, 0.1),
        "action": "block" if is_threat else "allow",
        "processing_time_ms": random.uniform(5, 50)
    }
    
    try:
        requests.post(f"{BASE_URL}/api/v1/stats/logs", json=log_data, timeout=2)
    except:
        pass

def simulate_prediction(is_malicious):
    """模拟URL预测"""
    features = generate_malicious_features() if is_malicious else generate_normal_features()
    
    try:
        response = requests.post(f"{BASE_URL}/predict_live", json=features, timeout=5)
        result = response.json()
        
        # 记录预测结果
        is_correct = (result["raw_prediction"] == 1) == is_malicious
        requests.post(
            f"{BASE_URL}/api/record-prediction",
            params={"is_correct": is_correct, "is_threat": is_malicious},
            timeout=2
        )
        return result
    except Exception as e:
        return {"error": str(e)}

def ddos_attack(duration=10, intensity=50):
    """模拟DDoS攻击"""
    print(f"\n[!] 开始DDoS攻击模拟 (持续{duration}秒, 强度{intensity}请求/秒)")
    
    end_time = time.time() + duration
    count = 0
    
    def single_request():
        nonlocal count
        ip = random.choice(ATTACKER_IPS)
        log_traffic("ddos", ip, True)
        simulate_prediction(True)
        count += 1
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        while time.time() < end_time:
            for _ in range(intensity // 10):
                executor.submit(single_request)
            time.sleep(0.1)
    
    print(f"[+] DDoS攻击完成: 发送 {count} 个请求")

def sql_injection_attack(count=20):
    """模拟SQL注入攻击"""
    print(f"\n[!] 开始SQL注入攻击模拟 ({count}次)")
    
    for i in range(count):
        ip = random.choice(ATTACKER_IPS)
        log_traffic("sql_injection", ip, True)
        simulate_prediction(True)
        time.sleep(0.1)
    
    print(f"[+] SQL注入攻击完成: {count} 次尝试")

def xss_attack(count=15):
    """模拟XSS攻击"""
    print(f"\n[!] 开始XSS攻击模拟 ({count}次)")
    
    for i in range(count):
        ip = random.choice(ATTACKER_IPS)
        log_traffic("xss", ip, True)
        simulate_prediction(True)
        time.sleep(0.1)
    
    print(f"[+] XSS攻击完成: {count} 次尝试")

def brute_force_attack(count=30):
    """模拟暴力破解攻击"""
    print(f"\n[!] 开始暴力破解攻击模拟 ({count}次)")
    
    ip = random.choice(ATTACKER_IPS)  # 同一IP多次尝试
    for i in range(count):
        log_traffic("brute_force", ip, True)
        simulate_prediction(True)
        time.sleep(0.05)
    
    print(f"[+] 暴力破解攻击完成: {count} 次尝试")

def port_scan_attack(count=25):
    """模拟端口扫描"""
    print(f"\n[!] 开始端口扫描模拟 ({count}次)")
    
    ip = random.choice(ATTACKER_IPS)
    for i in range(count):
        log_traffic("port_scan", ip, True)
        time.sleep(0.05)
    
    print(f"[+] 端口扫描完成: {count} 个端口")

def normal_traffic(count=50):
    """模拟正常流量"""
    print(f"\n[*] 生成正常流量 ({count}个请求)")
    
    for i in range(count):
        ip = random.choice(NORMAL_IPS)
        log_traffic("benign", ip, False)
        simulate_prediction(False)
        time.sleep(0.05)
    
    print(f"[+] 正常流量完成: {count} 个请求")

def mixed_attack_scenario():
    """混合攻击场景"""
    print("\n" + "="*60)
    print("       网络攻击模拟测试 - 混合场景")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("请打开浏览器访问: http://localhost:8000/dashboard")
    print("观察实时统计数据变化")
    print("="*60)
    
    # 阶段1: 正常流量基线
    print("\n[阶段1] 建立正常流量基线...")
    normal_traffic(30)
    time.sleep(2)
    
    # 阶段2: SQL注入攻击
    print("\n[阶段2] SQL注入攻击波...")
    sql_injection_attack(15)
    time.sleep(2)
    
    # 阶段3: 混合正常流量
    print("\n[阶段3] 混合正常流量...")
    normal_traffic(20)
    time.sleep(2)
    
    # 阶段4: XSS攻击
    print("\n[阶段4] XSS攻击波...")
    xss_attack(12)
    time.sleep(2)
    
    # 阶段5: 暴力破解
    print("\n[阶段5] 暴力破解攻击...")
    brute_force_attack(25)
    time.sleep(2)
    
    # 阶段6: 端口扫描
    print("\n[阶段6] 端口扫描...")
    port_scan_attack(20)
    time.sleep(2)
    
    # 阶段7: DDoS攻击
    print("\n[阶段7] DDoS攻击波...")
    ddos_attack(duration=8, intensity=30)
    time.sleep(2)
    
    # 阶段8: 恢复正常
    print("\n[阶段8] 恢复正常流量...")
    normal_traffic(30)
    
    # 打印统计
    print_stats()

def print_stats():
    """打印统计信息"""
    print("\n" + "="*60)
    print("       攻击模拟完成 - 统计报告")
    print("="*60)
    
    try:
        # 系统统计
        response = requests.get(f"{BASE_URL}/api/system-stats", timeout=5)
        stats = response.json()
        print(f"\n[系统统计]")
        print(f"  总预测次数: {stats.get('total_predictions', 0)}")
        print(f"  威胁检测数: {stats.get('threat_detections', 0)}")
        print(f"  准确率: {stats.get('accuracy', 0)}%")
        
        # 流量统计
        response = requests.get(f"{BASE_URL}/api/v1/stats/overview?hours=1", timeout=5)
        overview = response.json()
        print(f"\n[流量统计 (最近1小时)]")
        print(f"  总请求数: {overview.get('total_requests', 0)}")
        print(f"  阻止请求: {overview.get('blocked_requests', 0)}")
        print(f"  允许请求: {overview.get('allowed_requests', 0)}")
        
        # 威胁分布
        threat_counts = overview.get('threat_counts', {})
        if threat_counts:
            print(f"\n[威胁类型分布]")
            for threat_type, count in sorted(threat_counts.items(), key=lambda x: -x[1]):
                if count > 0:
                    print(f"  {threat_type}: {count}")
        
        # TOP攻击源
        top_ips = overview.get('top_source_ips', [])
        if top_ips:
            print(f"\n[TOP攻击源IP]")
            for ip, count in top_ips[:5]:
                print(f"  {ip}: {count} 次")
                
    except Exception as e:
        print(f"获取统计失败: {e}")
    
    print("\n" + "="*60)
    print("请查看仪表盘: http://localhost:8000/dashboard")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "ddos":
            ddos_attack(duration=15, intensity=50)
        elif cmd == "sql":
            sql_injection_attack(30)
        elif cmd == "xss":
            xss_attack(20)
        elif cmd == "brute":
            brute_force_attack(40)
        elif cmd == "scan":
            port_scan_attack(30)
        elif cmd == "normal":
            normal_traffic(100)
        elif cmd == "stats":
            print_stats()
        else:
            print("用法: python attack_simulation.py [ddos|sql|xss|brute|scan|normal|stats]")
    else:
        # 默认运行混合攻击场景
        mixed_attack_scenario()

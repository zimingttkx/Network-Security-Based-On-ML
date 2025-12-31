#!/usr/bin/env python3
"""
淘宝级别高并发流量模拟器
实现二层防御架构：服务器默认防御(一层) + 本系统应用层防御(二层)
"""

import asyncio
import aiohttp
import random
import time
import argparse
from datetime import datetime
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"
API_ENDPOINT = "/api/v1/stats/logs"
PROTECTION_STATE_ENDPOINT = "/api/v1/protection/state"

TRAFFIC_DISTRIBUTION = {"normal_user": 0.70, "crawler": 0.12, "bot": 0.08, "attacker": 0.10}
ATTACK_TYPES = ["ddos", "sql_injection", "xss", "brute_force", "probe", "phishing", "botnet", "web_attack"]
TAOBAO_URLS = ["/", "/search", "/item/detail", "/cart", "/order/confirm", "/pay", "/user/login", "/api/search"]
NORMAL_UAS = ["Mozilla/5.0 Chrome/120.0.0.0", "Mozilla/5.0 iPhone Safari", "Mozilla/5.0 Mac Safari"]
BOT_UAS = ["Googlebot/2.1", "python-requests/2.28", "scrapy/2.8", "", "Java/1.8"]

GEO = {"CN": (0.75, ["北京", "上海", "广州"]), "US": (0.10, ["New York", "LA"]), "XX": (0.15, ["Unknown"])}


def gen_ip(geo="CN"):
    prefixes = {"CN": ["116.25", "180.101"], "US": ["8.8", "104.16"], "XX": ["185.220", "45.33"]}
    p = random.choice(prefixes.get(geo, prefixes["CN"]))
    return f"{p}.{random.randint(1,254)}.{random.randint(1,254)}"


def select_geo():
    r, c = random.random(), 0
    for code, (w, cities) in GEO.items():
        c += w
        if r <= c: return code, random.choice(cities)
    return "CN", "北京"


def generate_traffic(protection_active: bool, protection_level: str) -> Dict[str, Any]:
    """
    二层防御逻辑：
    - 一层(服务器默认): 对明显恶意流量(critical级别)进行拦截
    - 二层(本系统): 仅在protection_active时启用，对中高风险进行二次筛选
    """
    r, cumulative, traffic_type = random.random(), 0, "normal_user"
    for t, w in TRAFFIC_DISTRIBUTION.items():
        cumulative += w
        if r <= cumulative:
            traffic_type = t
            break
    
    geo, city = select_geo()
    is_threat = traffic_type == "attacker" or (traffic_type == "bot" and random.random() < 0.6) or \
                (traffic_type == "crawler" and random.random() < 0.3)
    
    data = {
        "source_ip": gen_ip("XX" if is_threat else geo),
        "source_port": random.randint(10000, 65535),
        "dest_ip": "10.0.0.1",
        "dest_port": random.choice([80, 443]),
        "protocol": "HTTPS",
        "method": random.choice(["GET", "POST"]),
        "url": random.choice(TAOBAO_URLS),
        "geo_country": "XX" if is_threat else geo,
        "geo_city": "Unknown" if is_threat else city,
        "processing_time_ms": random.uniform(5, 50),
    }
    
    if is_threat:
        attack = random.choice(ATTACK_TYPES)
        risk_score = random.uniform(0.5, 1.0)
        data.update({
            "threat_type": attack,
            "risk_score": risk_score,
            "user_agent": random.choice(BOT_UAS),
        })
        
        # 根据风险分数确定风险等级
        if risk_score >= 0.85:
            data["risk_level"] = "critical"
        elif risk_score >= 0.7:
            data["risk_level"] = "high"
        elif risk_score >= 0.5:
            data["risk_level"] = "medium"
        else:
            data["risk_level"] = "low"
        
        # === 二层防御逻辑 ===
        # 一层(服务器默认防御): 只拦截critical级别的明显攻击
        if data["risk_level"] == "critical":
            data["action"] = "block"
            data["blocked_by"] = "server_default"  # 服务器一层拦截
        elif protection_active:
            # 二层(本系统): 根据保护级别进行二次筛选
            if protection_level == "strict":
                data["action"] = "block"
                data["blocked_by"] = "second_layer"
            elif protection_level == "high" and data["risk_level"] in ["high", "medium"]:
                data["action"] = "block"
                data["blocked_by"] = "second_layer"
            elif protection_level == "medium" and data["risk_level"] == "high":
                data["action"] = "block"
                data["blocked_by"] = "second_layer"
            elif protection_level == "low":
                data["action"] = "log"  # 低级只记录
                data["blocked_by"] = "none"
            else:
                data["action"] = random.choice(["challenge", "alert"])
                data["blocked_by"] = "none"
        else:
            # 保护未开启: 非critical的威胁只记录不拦截
            data["action"] = "log"
            data["blocked_by"] = "none"
    else:
        data.update({
            "threat_type": "benign",
            "risk_score": random.uniform(0.0, 0.15),
            "risk_level": "safe",
            "user_agent": random.choice(NORMAL_UAS),
            "action": "allow",
            "blocked_by": "none",
        })
    
    return data


class TrafficSimulator:
    """二层防御架构流量模拟器"""
    
    def __init__(self, base_url: str = BASE_URL, concurrency: int = 500):
        self.base_url = base_url
        self.concurrency = concurrency
        self.total_sent = 0
        self.success_count = 0
        self.start_time = None
        self.protection_active = False
        self.protection_level = "medium"
        # 统计
        self.server_blocked = 0      # 一层拦截
        self.second_layer_blocked = 0  # 二层拦截
        self.threats_logged = 0      # 仅记录的威胁
        
    async def check_protection_state(self, session: aiohttp.ClientSession):
        try:
            async with session.get(f"{self.base_url}{PROTECTION_STATE_ENDPOINT}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.protection_active = data.get("is_active", False)
                    self.protection_level = data.get("level", "medium")
        except: pass
    
    async def send_log(self, session: aiohttp.ClientSession, log_data: Dict) -> bool:
        try:
            # 统计拦截来源
            if log_data.get("action") == "block":
                if log_data.get("blocked_by") == "server_default":
                    self.server_blocked += 1
                elif log_data.get("blocked_by") == "second_layer":
                    self.second_layer_blocked += 1
            elif log_data.get("threat_type") != "benign" and log_data.get("action") == "log":
                self.threats_logged += 1
            
            async with session.post(f"{self.base_url}{API_ENDPOINT}", json=log_data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    self.success_count += 1
                    return True
                return False
        except: return False
    
    async def send_batch(self, session: aiohttp.ClientSession, batch_size: int):
        tasks = [self.send_log(session, generate_traffic(self.protection_active, self.protection_level)) for _ in range(batch_size)]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.total_sent += batch_size
    
    def print_progress(self, total_target: int):
        elapsed = time.time() - self.start_time
        rate = self.total_sent / elapsed if elapsed > 0 else 0
        pct = (self.total_sent / total_target) * 100
        status = f"二层防御开启({self.protection_level})" if self.protection_active else "仅一层防御"
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {status} | 进度:{pct:.1f}% | 一层拦截:{self.server_blocked} | 二层拦截:{self.second_layer_blocked} | 仅记录:{self.threats_logged} | 速率:{rate:.0f}/s", end="", flush=True)
    
    async def run(self, total_requests: int, batch_size: int = 100):
        print(f"\n{'='*70}")
        print(f"  淘宝级流量模拟器 - 二层防御架构")
        print(f"  目标: {total_requests:,} 请求 | 并发: {self.concurrency}")
        print(f"{'='*70}")
        print("\n防御架构说明:")
        print("  一层(服务器默认): 拦截critical级别的明显攻击")
        print("  二层(本系统): 开启后对中高风险进行二次筛选\n")
        
        self.start_time = time.time()
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            await self.check_protection_state(session)
            print(f"当前状态: {'二层防御已开启 - ' + self.protection_level if self.protection_active else '二层防御未开启(仅一层默认防御)'}\n")
            
            batches = total_requests // batch_size
            check_interval = max(1, batches // 20)
            
            for i in range(batches):
                await self.send_batch(session, batch_size)
                self.print_progress(total_requests)
                if i % check_interval == 0:
                    await self.check_protection_state(session)
            
            if total_requests % batch_size > 0:
                await self.send_batch(session, total_requests % batch_size)
        
        elapsed = time.time() - self.start_time
        print(f"\n\n{'='*70}")
        print(f"  完成! 耗时:{elapsed:.1f}s | 速率:{self.total_sent/elapsed:.0f}/s")
        print(f"  一层拦截:{self.server_blocked} | 二层拦截:{self.second_layer_blocked} | 仅记录:{self.threats_logged}")
        print(f"{'='*70}\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--requests", type=int, default=1000000)
    parser.add_argument("-c", "--concurrency", type=int, default=500)
    parser.add_argument("-b", "--batch", type=int, default=100)
    parser.add_argument("--url", type=str, default=BASE_URL)
    args = parser.parse_args()
    await TrafficSimulator(args.url, args.concurrency).run(args.requests, args.batch)


if __name__ == "__main__":
    print("\n启动流量模拟器 - 二层防御架构测试")
    print("仪表盘: http://127.0.0.1:8000/dashboard")
    print("一键保护: http://127.0.0.1:8000/protection\n")
    asyncio.run(main())

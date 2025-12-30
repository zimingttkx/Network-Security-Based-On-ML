#!/usr/bin/env python3
"""
网络安全威胁检测系统 - 功能测试脚本
测试所有API端点和功能
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

# 测试结果统计
test_results = {
    "passed": 0,
    "failed": 0,
    "total": 0,
    "details": []
}

def log_test(test_name: str, passed: bool, message: str = ""):
    """记录测试结果"""
    test_results["total"] += 1
    if passed:
        test_results["passed"] += 1
        status = "✅ PASS"
    else:
        test_results["failed"] += 1
        status = "❌ FAIL"

    result = f"{status} - {test_name}"
    if message:
        result += f": {message}"

    print(result)
    test_results["details"].append({
        "name": test_name,
        "passed": passed,
        "message": message
    })

def test_page_access():
    """测试前端页面访问"""
    print("\n" + "="*60)
    print("测试 1: 前端页面访问")
    print("="*60)

    pages = {
        "首页": "/",
        "预测页面": "/predict",
        "训练页面": "/train",
        "教程页面": "/tutorial",
        "API文档": "/docs"
    }

    for page_name, path in pages.items():
        try:
            response = requests.get(f"{BASE_URL}{path}", timeout=5)
            passed = response.status_code == 200
            log_test(f"访问{page_name}", passed, f"状态码: {response.status_code}")
        except Exception as e:
            log_test(f"访问{page_name}", False, str(e))

def test_prediction_api():
    """测试威胁预测API"""
    print("\n" + "="*60)
    print("测试 2: 威胁预测功能")
    print("="*60)

    # 测试数据 - 安全样本
    safe_data = {
        "having_IP_Address": 1,
        "URL_Length": 1,
        "Shortining_Service": 1,
        "having_At_Symbol": 1,
        "double_slash_redirecting": 1,
        "Prefix_Suffix": -1,
        "having_Sub_Domain": 1,
        "SSLfinal_State": 1,
        "Domain_registeration_length": 1,
        "Favicon": 1,
        "port": 1,
        "HTTPS_token": 1,
        "Request_URL": 1,
        "URL_of_Anchor": 1,
        "Links_in_tags": 1,
        "SFH": 1,
        "Submitting_to_email": 1,
        "Abnormal_URL": 1,
        "Redirect": 0,
        "on_mouseover": 1,
        "RightClick": 1,
        "popUpWidnow": 1,
        "Iframe": 1,
        "age_of_domain": 1,
        "DNSRecord": 1,
        "web_traffic": 1,
        "Page_Rank": 1,
        "Google_Index": 1,
        "Links_pointing_to_page": 1,
        "Statistical_report": 1
    }

    # 测试数据 - 威胁样本
    threat_data = {
        "having_IP_Address": -1,
        "URL_Length": -1,
        "Shortining_Service": -1,
        "having_At_Symbol": -1,
        "double_slash_redirecting": -1,
        "Prefix_Suffix": -1,
        "having_Sub_Domain": -1,
        "SSLfinal_State": -1,
        "Domain_registeration_length": -1,
        "Favicon": -1,
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
        "age_of_domain": -1,
        "DNSRecord": -1,
        "web_traffic": -1,
        "Page_Rank": -1,
        "Google_Index": -1,
        "Links_pointing_to_page": -1,
        "Statistical_report": -1
    }

    # 测试安全样本预测
    try:
        response = requests.post(f"{BASE_URL}/predict_live", json=safe_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            log_test("预测安全样本", True, f"预测结果: {result.get('prediction', 'N/A')}")
        else:
            log_test("预测安全样本", False, f"状态码: {response.status_code}")
    except Exception as e:
        log_test("预测安全样本", False, str(e))

    # 测试威胁样本预测
    try:
        response = requests.post(f"{BASE_URL}/predict_live", json=threat_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            log_test("预测威胁样本", True, f"预测结果: {result.get('prediction', 'N/A')}")
        else:
            log_test("预测威胁样本", False, f"状态码: {response.status_code}")
    except Exception as e:
        log_test("预测威胁样本", False, str(e))

    # 测试缺少字段的情况
    try:
        incomplete_data = {"having_IP_Address": 1}
        response = requests.post(f"{BASE_URL}/predict_live", json=incomplete_data, timeout=10)
        # 应该返回错误
        passed = response.status_code in [400, 422]
        log_test("处理不完整数据", passed, f"状态码: {response.status_code}")
    except Exception as e:
        log_test("处理不完整数据", False, str(e))

def test_feature_requirements():
    """测试特征要求API"""
    print("\n" + "="*60)
    print("测试 3: 特征要求API")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/api/features/requirements", timeout=5)
        if response.status_code == 200:
            result = response.json()
            total_features = result.get("total_features", 0)
            passed = total_features == 30
            log_test("获取特征要求", passed, f"特征数量: {total_features}")
        else:
            log_test("获取特征要求", False, f"状态码: {response.status_code}")
    except Exception as e:
        log_test("获取特征要求", False, str(e))

def test_data_validation():
    """测试数据验证功能"""
    print("\n" + "="*60)
    print("测试 4: 数据验证功能")
    print("="*60)

    # 创建测试CSV文件
    import pandas as pd
    import os

    # 创建一个简单的测试数据
    test_data = pd.DataFrame({
        "having_IP_Address": [1, -1, 1],
        "URL_Length": [1, -1, 0],
        "Shortining_Service": [1, -1, 1]
    })

    test_file = "test_data.csv"
    test_data.to_csv(test_file, index=False)

    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/api/data/validate", files=files, timeout=10)

        if response.status_code == 200:
            result = response.json()
            is_valid = result.get("is_valid", False)
            log_test("数据验证API", True, f"验证结果: {'有效' if is_valid else '无效'}")
        else:
            log_test("数据验证API", False, f"状态码: {response.status_code}")
    except Exception as e:
        log_test("数据验证API", False, str(e))
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)

def test_static_files():
    """测试静态文件访问"""
    print("\n" + "="*60)
    print("测试 5: 静态文件访问")
    print("="*60)

    static_files = [
        "/static/css/style.css",
        "/static/js/script.js"
    ]

    for file_path in static_files:
        try:
            response = requests.get(f"{BASE_URL}{file_path}", timeout=5)
            passed = response.status_code == 200
            log_test(f"访问{file_path}", passed, f"状态码: {response.status_code}")
        except Exception as e:
            log_test(f"访问{file_path}", False, str(e))

def print_summary():
    """打印测试总结"""
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"总测试数: {test_results['total']}")
    print(f"通过: {test_results['passed']} ✅")
    print(f"失败: {test_results['failed']} ❌")

    if test_results['total'] > 0:
        pass_rate = (test_results['passed'] / test_results['total']) * 100
        print(f"通过率: {pass_rate:.1f}%")

    print("="*60)

    # 保存测试报告
    with open("test_report.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    print("\n测试报告已保存到: test_report.json")

def main():
    """主测试函数"""
    print("="*60)
    print("网络安全威胁检测系统 - 功能测试")
    print("="*60)
    print(f"测试目标: {BASE_URL}")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 等待服务器启动
    print("\n检查服务器状态...")
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"✅ 服务器正在运行 (状态码: {response.status_code})")
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        print("请确保应用正在运行: python app.py")
        return

    # 运行所有测试
    test_page_access()
    test_prediction_api()
    test_feature_requirements()
    test_data_validation()
    test_static_files()

    # 打印总结
    print_summary()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
网络攻击防御压力测试 - 优化版
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# 加载模型
print("[*] 加载网络攻击检测模型...")
with open('models/network_attack_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/network_attack_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/network_attack_features.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# 加载并预处理数据
df = pd.read_csv('data/nsl_kdd_test.csv')
print(f"[+] 加载测试数据: {len(df)} 条")

from sklearn.preprocessing import LabelEncoder
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

df['label'] = (df['attack_type'] != 'normal').astype(int)
X = df[[c for c in feature_cols if c in df.columns]].select_dtypes(include=[np.number])
y = df['label'].values
attack_types = df['attack_type'].values

# 预先标准化所有数据
print("[*] 预处理数据...")
X_scaled = scaler.transform(X)
print(f"[+] 特征数: {X_scaled.shape[1]}")

# 统计
stats = defaultdict(int)
latencies = []

def run_stress_test(sample_size=10000, batch_size=100):
    """运行压力测试 - 批量预测"""
    print("\n" + "="*70)
    print("         网络攻击防御压力测试")
    print("="*70)
    
    # 采样
    indices = np.random.choice(len(X_scaled), min(sample_size, len(X_scaled)), replace=False)
    X_test = X_scaled[indices]
    y_test = y[indices]
    attack_test = attack_types[indices]
    
    print(f"样本数: {len(X_test)} | 批量大小: {batch_size}")
    print("="*70)
    
    print(f"\n[*] 开始压力测试...")
    start_time = time.time()
    
    # 批量预测
    total_batches = (len(X_test) + batch_size - 1) // batch_size
    all_preds = []
    
    for i in range(0, len(X_test), batch_size):
        batch_start = time.time()
        batch_X = X_test[i:i+batch_size]
        
        preds = model.predict(batch_X)
        all_preds.extend(preds)
        
        batch_time = (time.time() - batch_start) * 1000
        latencies.append(batch_time / len(batch_X))  # 平均每条延迟
        
        if (i // batch_size + 1) % 20 == 0:
            elapsed = time.time() - start_time
            qps = (i + batch_size) / elapsed
            print(f"  进度: {min(i+batch_size, len(X_test))}/{len(X_test)} | QPS: {qps:,.0f}")
    
    elapsed = time.time() - start_time
    all_preds = np.array(all_preds)
    
    # 计算指标
    for i, (pred, true_label, attack_type) in enumerate(zip(all_preds, y_test, attack_test)):
        stats['total'] += 1
        if pred == true_label:
            stats['correct'] += 1
        
        if true_label == 1:
            stats['actual_attack'] += 1
            if pred == 1:
                stats['tp'] += 1
            else:
                stats['fn'] += 1
        else:
            stats['actual_normal'] += 1
            if pred == 1:
                stats['fp'] += 1
            else:
                stats['tn'] += 1
        
        stats[f'attack_{attack_type}'] += 1
    
    total = stats['total']
    tp, tn, fp, fn = stats['tp'], stats['tn'], stats['fp'], stats['fn']
    
    accuracy = stats['correct'] / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    qps = total / elapsed
    
    print("\n" + "="*70)
    print("                    测试结果报告")
    print("="*70)
    
    print(f"\n[性能指标]")
    print(f"  总请求数: {total:,}")
    print(f"  总耗时: {elapsed:.2f}秒")
    print(f"  吞吐量 (QPS): {qps:,.0f}")
    
    print(f"\n[检测性能]")
    print(f"  准确率 (Accuracy):  {accuracy:.2f}%")
    print(f"  精确率 (Precision): {precision:.2f}%")
    print(f"  召回率 (Recall):    {recall:.2f}%")
    print(f"  F1分数:             {f1:.2f}%")
    print(f"  误报率 (FPR):       {fpr:.2f}%")
    print(f"  漏报率 (FNR):       {fnr:.2f}%")
    
    print(f"\n[混淆矩阵]")
    print(f"  真阳性 (TP): {tp:,} - 正确检测的攻击")
    print(f"  真阴性 (TN): {tn:,} - 正确放行的正常流量")
    print(f"  假阳性 (FP): {fp:,} - 误报")
    print(f"  假阴性 (FN): {fn:,} - 漏报")
    
    print(f"\n[延迟统计]")
    if latencies:
        print(f"  平均每条: {sum(latencies)/len(latencies):.3f}ms")
    
    print(f"\n[攻击类型检测统计 TOP10]")
    attack_stats = {k.replace('attack_', ''): v for k, v in stats.items() if k.startswith('attack_')}
    for attack, count in sorted(attack_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {attack:20s}: {count:,}")
    
    print("="*70)
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'qps': qps}

if __name__ == "__main__":
    result = run_stress_test(sample_size=20000, batch_size=500)
    print(f"\n[结论] 系统在 {result['qps']:,.0f} QPS 下保持 {result['accuracy']:.2f}% 准确率")

#!/usr/bin/env python3
"""
算法对比测试 - 找出最佳检测算法
测试所有可用的ML/DL算法组合，找出效果最好的作为默认算法
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# 尝试导入XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[!] XGBoost未安装，跳过该算法")

# 定义所有算法
ALGORITHMS = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'NaiveBayes': GaussianNB(),
}

if HAS_XGBOOST:
    ALGORITHMS['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')


def load_phishing_data():
    """加载钓鱼网站数据集"""
    data_path = "data/phisingData.csv"
    if not os.path.exists(data_path):
        # 尝试其他路径
        for alt_path in ["artifacts/data_ingestion/phisingData.csv", "Network_Data/phisingData.csv"]:
            if os.path.exists(alt_path):
                data_path = alt_path
                break
    
    if not os.path.exists(data_path):
        print(f"[!] 找不到数据文件，尝试从artifacts加载...")
        # 列出可能的数据文件
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.endswith('.csv') and 'phising' in f.lower():
                    data_path = os.path.join(root, f)
                    print(f"[+] 找到数据文件: {data_path}")
                    break
    
    df = pd.read_csv(data_path)
    print(f"[+] 加载数据: {len(df)} 条记录, {len(df.columns)} 个特征")
    return df


def load_nslkdd_data():
    """加载NSL-KDD数据集"""
    data_path = "data/nsl_kdd_test.csv"
    if not os.path.exists(data_path):
        print("[!] NSL-KDD数据集不存在")
        return None
    
    df = pd.read_csv(data_path)
    print(f"[+] 加载NSL-KDD数据: {len(df)} 条记录")
    return df


def prepare_phishing_data(df):
    """准备钓鱼数据集"""
    # 分离特征和标签
    if 'Result' in df.columns:
        X = df.drop('Result', axis=1)
        y = df['Result']
    elif 'class' in df.columns:
        X = df.drop('class', axis=1)
        y = df['class']
    else:
        # 假设最后一列是标签
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # 转换标签为0/1
    if y.min() == -1:
        y = (y + 1) // 2  # -1->0, 1->1
    
    return X, y


def prepare_nslkdd_data(df, sample_size=10000):
    """准备NSL-KDD数据集"""
    # 采样
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # 编码分类特征
    categorical_cols = ['protocol_type', 'service', 'flag']
    le = LabelEncoder()
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    # 创建二分类标签
    df['label'] = (df['attack_type'] != 'normal').astype(int)
    
    # 选择数值特征
    feature_cols = [c for c in df.columns if c not in ['attack_type', 'difficulty', 'label']]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['label']
    
    return X, y


def benchmark_algorithm(name, model, X_train, X_test, y_train, y_test):
    """测试单个算法"""
    print(f"\n  测试 {name}...", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        # 训练
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 预测
        start_pred = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_pred
        
        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"完成 (训练: {train_time:.2f}s)")
        
        return {
            'name': name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'fpr': fpr,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'train_time': train_time,
            'pred_time': pred_time,
            'model': model
        }
    except Exception as e:
        print(f"失败: {e}")
        return None


def run_benchmark():
    """运行完整的算法对比测试"""
    print("="*70)
    print("           算法对比测试 - 寻找最佳检测算法")
    print("="*70)
    
    results = []
    
    # 测试1: 钓鱼网站数据集
    print("\n[1/2] 钓鱼网站数据集测试")
    print("-"*50)
    
    try:
        df_phishing = load_phishing_data()
        X, y = prepare_phishing_data(df_phishing)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  训练集: {len(X_train)} | 测试集: {len(X_test)}")
        
        phishing_results = []
        for name, model in ALGORITHMS.items():
            result = benchmark_algorithm(name, model, X_train, X_test, y_train, y_test)
            if result:
                result['dataset'] = 'phishing'
                phishing_results.append(result)
                results.append(result)
        
        # 显示钓鱼数据集结果
        print("\n  钓鱼数据集结果排名 (按F1分数):")
        phishing_results.sort(key=lambda x: x['f1'], reverse=True)
        for i, r in enumerate(phishing_results[:5], 1):
            print(f"  {i}. {r['name']:20s} F1={r['f1']:.4f} Acc={r['accuracy']:.4f} Prec={r['precision']:.4f} Rec={r['recall']:.4f}")
        
    except Exception as e:
        print(f"  [!] 钓鱼数据集测试失败: {e}")
    
    # 测试2: NSL-KDD数据集
    print("\n[2/2] NSL-KDD数据集测试")
    print("-"*50)
    
    try:
        df_nslkdd = load_nslkdd_data()
        if df_nslkdd is not None:
            X, y = prepare_nslkdd_data(df_nslkdd)
            
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"  训练集: {len(X_train)} | 测试集: {len(X_test)}")
            
            nslkdd_results = []
            for name, model_template in ALGORITHMS.items():
                # 创建新模型实例
                model = model_template.__class__(**model_template.get_params())
                result = benchmark_algorithm(name, model, X_train, X_test, y_train, y_test)
                if result:
                    result['dataset'] = 'nslkdd'
                    nslkdd_results.append(result)
                    results.append(result)
            
            # 显示NSL-KDD结果
            print("\n  NSL-KDD数据集结果排名 (按F1分数):")
            nslkdd_results.sort(key=lambda x: x['f1'], reverse=True)
            for i, r in enumerate(nslkdd_results[:5], 1):
                print(f"  {i}. {r['name']:20s} F1={r['f1']:.4f} Acc={r['accuracy']:.4f} Prec={r['precision']:.4f} Rec={r['recall']:.4f}")
    except Exception as e:
        print(f"  [!] NSL-KDD数据集测试失败: {e}")
    
    # 综合排名
    print("\n" + "="*70)
    print("                    综合排名")
    print("="*70)
    
    # 按算法聚合结果
    algo_scores = {}
    for r in results:
        name = r['name']
        if name not in algo_scores:
            algo_scores[name] = {'f1_sum': 0, 'acc_sum': 0, 'count': 0}
        algo_scores[name]['f1_sum'] += r['f1']
        algo_scores[name]['acc_sum'] += r['accuracy']
        algo_scores[name]['count'] += 1
    
    # 计算平均分
    rankings = []
    for name, scores in algo_scores.items():
        avg_f1 = scores['f1_sum'] / scores['count']
        avg_acc = scores['acc_sum'] / scores['count']
        rankings.append((name, avg_f1, avg_acc))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print("\n算法综合排名 (按平均F1分数):")
    print("-"*50)
    print(f"{'排名':<4} {'算法':<20} {'平均F1':<10} {'平均准确率':<10}")
    print("-"*50)
    
    for i, (name, avg_f1, avg_acc) in enumerate(rankings, 1):
        marker = " ★ 推荐" if i == 1 else ""
        print(f"{i:<4} {name:<20} {avg_f1:.4f}     {avg_acc:.4f}{marker}")
    
    # 返回最佳算法
    best_algo = rankings[0][0] if rankings else "RandomForest"
    print(f"\n[结论] 最佳算法: {best_algo}")
    print("="*70)
    
    return best_algo, results


if __name__ == "__main__":
    best_algo, results = run_benchmark()
    
    # 保存结果
    result_df = pd.DataFrame([{
        'algorithm': r['name'],
        'dataset': r['dataset'],
        'accuracy': r['accuracy'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1': r['f1'],
        'fpr': r['fpr'],
        'train_time': r['train_time']
    } for r in results])
    
    result_df.to_csv('algorithm_benchmark_results.csv', index=False)
    print(f"\n[+] 结果已保存到 algorithm_benchmark_results.csv")

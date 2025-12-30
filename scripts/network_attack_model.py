#!/usr/bin/env python3
"""
网络攻击检测模型训练
使用NSL-KDD数据集训练专门的网络攻击防御模型
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_nslkdd():
    """加载NSL-KDD数据集"""
    df = pd.read_csv('data/nsl_kdd_test.csv')
    print(f"[+] 加载NSL-KDD: {len(df)} 条记录")
    return df

def prepare_data(df):
    """准备训练数据"""
    # 编码分类特征
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # 攻击类型映射到大类
    attack_map = {
        'normal': 'normal',
        'neptune': 'dos', 'smurf': 'dos', 'pod': 'dos', 'teardrop': 'dos',
        'land': 'dos', 'back': 'dos', 'apache2': 'dos', 'processtable': 'dos',
        'mailbomb': 'dos', 'udpstorm': 'dos',
        'ipsweep': 'probe', 'portsweep': 'probe', 'nmap': 'probe', 'satan': 'probe',
        'mscan': 'probe', 'saint': 'probe',
        'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l',
        'multihop': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l',
        'spy': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpguess': 'r2l',
        'snmpgetattack': 'r2l', 'httptunnel': 'r2l', 'sendmail': 'r2l',
        'named': 'r2l', 'worm': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r',
        'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r',
    }
    
    df['attack_category'] = df['attack_type'].map(lambda x: attack_map.get(x, 'unknown'))
    df['label'] = (df['attack_type'] != 'normal').astype(int)
    
    # 选择特征
    feature_cols = [c for c in df.columns if c not in ['attack_type', 'difficulty', 'label', 'attack_category']]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['label']
    
    print(f"[+] 特征数: {X.shape[1]}")
    print(f"[+] 正常流量: {(y==0).sum()} | 攻击流量: {(y==1).sum()}")
    
    return X, y, encoders, feature_cols

def train_model(X, y):
    """训练模型"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练RandomForest
    print("\n[*] 训练RandomForest模型...")
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # 评估
    y_pred = model.predict(X_test_scaled)
    
    print("\n" + "="*50)
    print("         模型评估结果")
    print("="*50)
    print(f"准确率: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"精确率: {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"召回率: {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"F1分数: {f1_score(y_test, y_pred)*100:.2f}%")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '攻击']))
    
    return model, scaler

def save_model(model, scaler, feature_cols):
    """保存模型"""
    os.makedirs('models', exist_ok=True)
    
    # 保存网络攻击检测模型
    with open('models/network_attack_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/network_attack_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/network_attack_features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("\n[+] 模型已保存:")
    print("  - models/network_attack_model.pkl")
    print("  - models/network_attack_scaler.pkl")
    print("  - models/network_attack_features.pkl")

if __name__ == "__main__":
    print("="*50)
    print("    网络攻击检测模型训练")
    print("="*50)
    
    df = load_nslkdd()
    X, y, encoders, feature_cols = prepare_data(df)
    model, scaler = train_model(X, y)
    save_model(model, scaler, list(X.columns))
    
    print("\n[+] 训练完成!")

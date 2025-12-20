"""
深度学习功能单元测试
测试深度学习模型训练器的各项功能
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from networksecurity.components.dl_model_trainer import DLModelTrainer
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact


def create_test_data():
    """创建测试数据"""
    print("\n" + "="*60)
    print("创建测试数据...")
    print("="*60)

    # 读取真实数据
    csv_path = "Network_Data/phisingData.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ 成功读取数据: {df.shape[0]} 行 x {df.shape[1]} 列")

    # 分离特征和标签
    X = df.drop(columns=['Result'])
    y = df['Result'].replace(-1, 0)  # 将-1转换为0

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"✓ 训练集: {X_train.shape[0]} 样本")
    print(f"✓ 测试集: {X_test.shape[0]} 样本")

    return X_train, X_test, y_train, y_test


def test_dnn_model():
    """测试DNN模型"""
    print("\n" + "="*60)
    print("测试 1: DNN (深度神经网络) 模型")
    print("="*60)

    try:
        # 创建测试数据
        X_train, X_test, y_train, y_test = create_test_data()

        # 创建DNN配置
        dl_config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 5,  # 测试时使用较少的epoch
            'optimizer': 'adam',
            'dropout_rate': 0.3,
            'l2_reg': 0.001,
            'early_stopping_patience': 3,
            'use_batch_norm': True,
            'activation': 'relu',
            'hidden_layers': [128, 64, 32]
        }

        print(f"\n配置参数:")
        for key, value in dl_config.items():
            print(f"  {key}: {value}")

        # 创建训练器
        trainer = DLModelTrainer(
            model_type='dnn',
            config=dl_config
        )

        print("\n开始训练DNN模型...")
        model, metrics = trainer.train(X_train, y_train, X_test, y_test)

        print(f"\n✓ DNN模型训练完成!")
        print(f"  准确率: {metrics.f1_score:.4f}")
        print(f"  精确率: {metrics.precision_score:.4f}")
        print(f"  召回率: {metrics.recall_score:.4f}")
        print(f"  F1分数: {metrics.f1_score:.4f}")

        # 测试预测
        sample = X_test.iloc[:5]
        predictions = model.predict(sample)
        print(f"\n✓ 预测测试通过，预测结果: {predictions}")

        return True

    except Exception as e:
        print(f"\n✗ DNN模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn_model():
    """测试CNN模型"""
    print("\n" + "="*60)
    print("测试 2: CNN (卷积神经网络) 模型")
    print("="*60)

    try:
        # 创建测试数据
        X_train, X_test, y_train, y_test = create_test_data()

        # 创建CNN配置
        dl_config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 5,
            'optimizer': 'adam',
            'dropout_rate': 0.3,
            'l2_reg': 0.001,
            'early_stopping_patience': 3,
            'use_batch_norm': True,
            'activation': 'relu',
            'conv_filters': [64, 32],
            'dense_layers': [64, 32],
            'kernel_size': 3
        }

        print(f"\n配置参数:")
        for key, value in dl_config.items():
            print(f"  {key}: {value}")

        # 创建训练器
        trainer = DLModelTrainer(
            model_type='cnn',
            config=dl_config
        )

        print("\n开始训练CNN模型...")
        model, metrics = trainer.train(X_train, y_train, X_test, y_test)

        print(f"\n✓ CNN模型训练完成!")
        print(f"  准确率: {metrics.f1_score:.4f}")
        print(f"  精确率: {metrics.precision_score:.4f}")
        print(f"  召回率: {metrics.recall_score:.4f}")
        print(f"  F1分数: {metrics.f1_score:.4f}")

        # 测试预测
        sample = X_test.iloc[:5]
        predictions = model.predict(sample)
        print(f"\n✓ 预测测试通过，预测结果: {predictions}")

        return True

    except Exception as e:
        print(f"\n✗ CNN模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_lstm_model():
    """测试LSTM模型"""
    print("\n" + "="*60)
    print("测试 3: LSTM (长短期记忆网络) 模型")
    print("="*60)

    try:
        # 创建测试数据
        X_train, X_test, y_train, y_test = create_test_data()

        # 创建LSTM配置
        dl_config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 5,
            'optimizer': 'adam',
            'dropout_rate': 0.3,
            'l2_reg': 0.001,
            'early_stopping_patience': 3,
            'use_batch_norm': True,
            'activation': 'relu',
            'lstm_units': [64, 32],
            'dense_layers': [32],
            'recurrent_dropout': 0.2
        }

        print(f"\n配置参数:")
        for key, value in dl_config.items():
            print(f"  {key}: {value}")

        # 创建训练器
        trainer = DLModelTrainer(
            model_type='lstm',
            config=dl_config
        )

        print("\n开始训练LSTM模型...")
        model, metrics = trainer.train(X_train, y_train, X_test, y_test)

        print(f"\n✓ LSTM模型训练完成!")
        print(f"  准确率: {metrics.f1_score:.4f}")
        print(f"  精确率: {metrics.precision_score:.4f}")
        print(f"  召回率: {metrics.recall_score:.4f}")
        print(f"  F1分数: {metrics.f1_score:.4f}")

        # 测试预测
        sample = X_test.iloc[:5]
        predictions = model.predict(sample)
        print(f"\n✓ 预测测试通过，预测结果: {predictions}")

        return True

    except Exception as e:
        print(f"\n✗ LSTM模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_default_configs():
    """测试默认配置"""
    print("\n" + "="*60)
    print("测试 4: 默认配置获取")
    print("="*60)

    try:
        trainer = DLModelTrainer(model_type='dnn')

        # 测试DNN默认配置
        dnn_config = trainer.get_default_config('dnn')
        print(f"\n✓ DNN默认配置:")
        for key, value in dnn_config.items():
            print(f"  {key}: {value}")

        # 测试CNN默认配置
        cnn_config = trainer.get_default_config('cnn')
        print(f"\n✓ CNN默认配置:")
        for key, value in cnn_config.items():
            print(f"  {key}: {value}")

        # 测试LSTM默认配置
        lstm_config = trainer.get_default_config('lstm')
        print(f"\n✓ LSTM默认配置:")
        for key, value in lstm_config.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"\n✗ 默认配置测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("深度学习功能单元测试")
    print("="*60)

    results = {
        'DNN模型': False,
        'CNN模型': False,
        'LSTM模型': False,
        '默认配置': False
    }

    # 运行测试
    results['默认配置'] = test_default_configs()
    results['DNN模型'] = test_dnn_model()
    results['CNN模型'] = test_cnn_model()
    results['LSTM模型'] = test_lstm_model()

    # 输出测试结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 所有测试通过！深度学习功能正常工作。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit(main())

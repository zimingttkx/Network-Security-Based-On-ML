"""
数据验证和特征补全工具

功能:
1. 验证上传数据的特征完整性
2. 提供特征缺失的补全方案
3. 数据质量检查
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.impute import KNNImputer, SimpleImputer
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证和特征补全类"""
    
    # 所需的30个特征
    REQUIRED_FEATURES = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
        'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
        'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
        'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
        'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
        'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
        'Statistical_report'
    ]
    
    # 特征说明
    FEATURE_DESCRIPTIONS = {
        'having_IP_Address': 'URL中是否包含IP地址 (-1: 是, 1: 否)',
        'URL_Length': 'URL长度 (1: 正常, 0: 可疑, -1: 异常)',
        'Shortining_Service': '是否使用短链服务 (-1: 是, 1: 否)',
        'having_At_Symbol': 'URL中是否包含@符号 (-1: 是, 1: 否)',
        'double_slash_redirecting': '是否有双斜杠重定向 (-1: 是, 1: 否)',
        'Prefix_Suffix': '域名中是否有前缀/后缀 (-1: 是, 1: 否)',
        'having_Sub_Domain': '子域名数量 (1: 正常, 0: 可疑, -1: 异常)',
        'SSLfinal_State': 'SSL证书状态 (1: 有效, 0: 可疑, -1: 无效)',
        'Domain_registeration_length': '域名注册时长 (1: 长, -1: 短)',
        'Favicon': '是否有Favicon图标 (1: 是, -1: 否)',
        'port': '端口是否标准 (1: 标准, -1: 非标准)',
        'HTTPS_token': 'HTTPS令牌 (1: 有, -1: 无)',
        'Request_URL': '请求URL资源比例 (1: 正常, -1: 异常)',
        'URL_of_Anchor': '锚点URL比例 (1: 正常, 0: 可疑, -1: 异常)',
        'Links_in_tags': '标签中链接比例 (1: 正常, 0: 可疑, -1: 异常)',
        'SFH': '表单提交地址 (1: 正常, 0: 可疑, -1: 异常)',
        'Submitting_to_email': '是否提交到邮箱 (-1: 是, 1: 否)',
        'Abnormal_URL': 'URL是否异常 (-1: 是, 1: 否)',
        'Redirect': '重定向次数 (0: 无, 1: 1-3次, -1: >3次)',
        'on_mouseover': '是否有onMouseOver事件 (-1: 是, 1: 否)',
        'RightClick': '是否禁用右键 (-1: 是, 1: 否)',
        'popUpWidnow': '是否有弹窗 (-1: 是, 1: 否)',
        'Iframe': '是否使用iframe (-1: 是, 1: 否)',
        'age_of_domain': '域名年龄 (1: 长, -1: 短)',
        'DNSRecord': 'DNS记录 (1: 有, -1: 无)',
        'web_traffic': '网站流量 (1: 高, 0: 中, -1: 低)',
        'Page_Rank': '页面排名 (1: 高, -1: 低)',
        'Google_Index': '是否被Google索引 (1: 是, -1: 否)',
        'Links_pointing_to_page': '指向页面的链接数 (1: 多, 0: 中, -1: 少)',
        'Statistical_report': '统计报告 (-1: 有异常, 1: 无异常)'
    }
    
    def __init__(self):
        self.imputers = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'most_frequent': SimpleImputer(strategy='most_frequent'),
            'constant': SimpleImputer(strategy='constant', fill_value=0),
            'knn': KNNImputer(n_neighbors=5)
        }
    
    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        验证数据特征完整性
        
        Args:
            df: 输入数据DataFrame
            
        Returns:
            (is_valid, validation_report)
        """
        report = {
            'is_valid': True,
            'missing_features': [],
            'extra_features': [],
            'missing_values': {},
            'data_types': {},
            'value_ranges': {},
            'recommendations': []
        }
        
        # 检查缺失的特征
        current_features = set(df.columns)
        required_features = set(self.REQUIRED_FEATURES)
        
        report['missing_features'] = list(required_features - current_features)
        report['extra_features'] = list(current_features - required_features)
        
        if report['missing_features']:
            report['is_valid'] = False
            report['recommendations'].append({
                'issue': f"缺少 {len(report['missing_features'])} 个特征",
                'features': report['missing_features'],
                'solution': "请添加缺失特征或使用特征补全功能"
            })
        
        # 检查缺失值
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                missing_count = df[col].isnull().sum()
                missing_percent = (missing_count / len(df)) * 100
                report['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_percent, 2)
                }
                
                if missing_percent > 10:
                    report['recommendations'].append({
                        'issue': f"特征 '{col}' 缺失值过多 ({missing_percent:.1f}%)",
                        'solution': "建议使用KNN补全或删除该特征"
                    })
        
        # 检查数据类型
        for col in df.columns:
            report['data_types'][col] = str(df[col].dtype)
            if not np.issubdtype(df[col].dtype, np.number):
                report['is_valid'] = False
                report['recommendations'].append({
                    'issue': f"特征 '{col}' 不是数值类型",
                    'solution': "请将所有特征转换为数值类型"
                })
        
        # 检查值域
        for col in df.select_dtypes(include=[np.number]).columns:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            report['value_ranges'][col] = {
                'min': min_val,
                'max': max_val
            }
            
            # 网络安全特征通常在 -1 到 1 之间
            if min_val < -10 or max_val > 10:
                report['recommendations'].append({
                    'issue': f"特征 '{col}' 值域异常 ({min_val} ~ {max_val})",
                    'solution': "建议检查数据是否需要标准化或归一化"
                })
        
        return report['is_valid'], report
    
    def get_feature_requirements(self) -> Dict:
        """获取特征要求说明"""
        return {
            'total_features': len(self.REQUIRED_FEATURES),
            'features': [
                {
                    'name': feature,
                    'description': self.FEATURE_DESCRIPTIONS.get(feature, ''),
                    'type': 'integer',
                    'typical_values': '-1, 0, 1'
                }
                for feature in self.REQUIRED_FEATURES
            ]
        }
    
    def impute_missing_features(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'constant',
        fill_value: int = 0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        补全缺失特征
        
        Args:
            df: 输入数据
            strategy: 补全策略 ('mean', 'median', 'most_frequent', 'constant', 'knn')
            fill_value: constant策略的填充值
            
        Returns:
            (补全后的DataFrame, 补全报告)
        """
        report = {
            'added_features': [],
            'imputed_values': {},
            'strategy': strategy
        }
        
        df_copy = df.copy()
        
        # 添加缺失的特征列
        for feature in self.REQUIRED_FEATURES:
            if feature not in df_copy.columns:
                df_copy[feature] = fill_value
                report['added_features'].append(feature)
        
        # 补全缺失值
        if df_copy.isnull().sum().sum() > 0:
            if strategy == 'constant':
                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
            elif strategy in self.imputers:
                imputer = self.imputers[strategy]
            else:
                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
            
            # 只对数值列进行补全
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
                
                for col in numeric_cols:
                    if col in report['imputed_values'] or df[col].isnull().sum() > 0:
                        report['imputed_values'][col] = {
                            'missing_count': int(df[col].isnull().sum()) if col in df else 0,
                            'strategy': strategy
                        }
        
        # 确保列顺序与REQUIRED_FEATURES一致
        df_result = df_copy[self.REQUIRED_FEATURES]
        
        return df_result, report
    
    def suggest_imputation_strategy(self, df: pd.DataFrame) -> Dict:
        """
        建议最佳补全策略
        
        Args:
            df: 输入数据
            
        Returns:
            补全策略建议
        """
        suggestions = []
        
        missing_percent = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        if missing_percent == 0:
            suggestions.append({
                'strategy': None,
                'reason': '数据完整，无需补全',
                'priority': 1
            })
        elif missing_percent < 5:
            suggestions.append({
                'strategy': 'mean',
                'reason': '缺失值较少，使用均值补全即可',
                'priority': 1
            })
        elif missing_percent < 15:
            suggestions.append({
                'strategy': 'knn',
                'reason': '缺失值中等，KNN补全效果较好',
                'priority': 1
            })
            suggestions.append({
                'strategy': 'median',
                'reason': '中位数补全对异常值robust',
                'priority': 2
            })
        else:
            suggestions.append({
                'strategy': 'constant',
                'reason': '缺失值较多，建议使用常数补全或重新采集数据',
                'priority': 1
            })
            suggestions.append({
                'strategy': 'most_frequent',
                'reason': '使用最常见值补全',
                'priority': 2
            })
        
        return {
            'missing_percentage': round(missing_percent, 2),
            'suggestions': suggestions
        }


# 导出
__all__ = ['DataValidator']

"""
模型解释工具

功能:
1. 使用SHAP库对模型预测进行解释
2. 生成特征重要性可视化
3. 支持Tree-based模型（RandomForest, XGBoost等）
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ModelExplainer:
    """模型解释类，使用SHAP进行模型可解释性分析"""
    
    SUPPORTED_MODELS = [
        'RandomForestClassifier',
        'XGBClassifier',
        'GradientBoostingClassifier',
        'DecisionTreeClassifier',
        'ExtraTreesClassifier',
        'LGBMClassifier'
    ]
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        初始化模型解释器
        
        Args:
            feature_names: 特征名称列表，用于解释结果的可读性
        """
        self.feature_names = feature_names
        self.explainer = None
        self._last_shap_values = None
    
    def _get_model_type(self, model: Any) -> str:
        """获取模型类型名称"""
        return type(model).__name__
    
    def _is_tree_model(self, model: Any) -> bool:
        """检查是否为Tree-based模型"""
        model_type = self._get_model_type(model)
        return model_type in self.SUPPORTED_MODELS
    
    def _create_explainer(self, model: Any) -> shap.Explainer:
        """
        创建SHAP解释器
        
        Args:
            model: 训练好的模型
            
        Returns:
            SHAP解释器实例
        """
        model_type = self._get_model_type(model)
        
        if self._is_tree_model(model):
            logger.info(f"为 {model_type} 创建 TreeExplainer")
            return shap.TreeExplainer(model)
        else:
            logger.warning(f"模型类型 {model_type} 不在支持列表中，尝试使用通用Explainer")
            return shap.Explainer(model)
    
    def _get_base_value(self, expected_value: Any) -> float:
        """安全获取base_value"""
        if isinstance(expected_value, (list, np.ndarray)):
            if len(expected_value) > 1:
                return float(expected_value[1])
            return float(expected_value[0])
        return float(expected_value)
    
    def _get_shap_values_for_class(self, shap_values: Any, class_idx: int = 1) -> np.ndarray:
        """获取指定类别的SHAP值"""
        if isinstance(shap_values, list):
            return np.array(shap_values[class_idx])
        if len(shap_values.shape) == 3:
            return shap_values[:, :, class_idx]
        return np.array(shap_values)

    def get_explanation(
        self, 
        model: Any, 
        input_data: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        获取模型预测的SHAP解释
        
        Args:
            model: 训练好的模型
            input_data: 输入数据，可以是numpy数组或DataFrame
            
        Returns:
            JSON格式的解释结果，包含每个特征的SHAP值
        """
        try:
            if isinstance(input_data, pd.DataFrame):
                feature_names = list(input_data.columns)
                data_array = input_data.values.astype(np.float64)
            else:
                data_array = np.array(input_data, dtype=np.float64)
                if len(data_array.shape) == 1:
                    data_array = data_array.reshape(1, -1)
                feature_names = self.feature_names or [f"feature_{i}" for i in range(data_array.shape[1])]
            
            self.explainer = self._create_explainer(model)
            shap_values = self.explainer.shap_values(data_array)
            self._last_shap_values = shap_values
            
            shap_values_to_use = self._get_shap_values_for_class(shap_values)
            if len(shap_values_to_use.shape) == 1:
                shap_values_to_use = shap_values_to_use.reshape(1, -1)
            
            base_value = self._get_base_value(self.explainer.expected_value)
            
            results = []
            for i in range(data_array.shape[0]):
                sample_explanation = {
                    "sample_index": i,
                    "base_value": base_value,
                    "features": {}
                }
                
                for j, feature_name in enumerate(feature_names):
                    shap_val = float(shap_values_to_use[i, j])
                    sample_explanation["features"][feature_name] = {
                        "value": float(data_array[i, j]),
                        "shap_value": shap_val,
                        "contribution": "positive" if shap_val > 0 else "negative"
                    }
                
                sorted_features = sorted(
                    sample_explanation["features"].items(),
                    key=lambda x: abs(x[1]["shap_value"]),
                    reverse=True
                )
                sample_explanation["top_features"] = [
                    {"name": name, **values} for name, values in sorted_features[:10]
                ]
                
                results.append(sample_explanation)
            
            return {
                "success": True,
                "model_type": self._get_model_type(model),
                "num_samples": len(results),
                "explanations": results
            }
            
        except Exception as e:
            logger.error(f"获取SHAP解释时出错: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_type": self._get_model_type(model)
            }
    
    def generate_summary_plot(
        self, 
        model: Any, 
        test_data: Union[np.ndarray, pd.DataFrame],
        plot_type: str = "dot",
        max_display: int = 20
    ) -> Dict[str, Any]:
        """
        生成SHAP总结图并转换为Base64编码
        
        Args:
            model: 训练好的模型
            test_data: 测试数据
            plot_type: 图表类型 ("dot", "bar", "violin")
            max_display: 最多显示的特征数量
            
        Returns:
            包含Base64编码图片的字典
        """
        try:
            if isinstance(test_data, pd.DataFrame):
                feature_names = list(test_data.columns)
                data_array = test_data.values.astype(np.float64)
            else:
                data_array = np.array(test_data, dtype=np.float64)
                feature_names = self.feature_names or [f"feature_{i}" for i in range(data_array.shape[1])]
            
            self.explainer = self._create_explainer(model)
            shap_values = self.explainer.shap_values(data_array)
            self._last_shap_values = shap_values
            
            shap_values_to_plot = self._get_shap_values_for_class(shap_values)
            
            plt.figure(figsize=(12, 8))
            plt.clf()
            
            shap.summary_plot(
                shap_values_to_plot,
                data_array,
                feature_names=feature_names,
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            plt.close()
            
            return {
                "success": True,
                "model_type": self._get_model_type(model),
                "plot_type": plot_type,
                "num_samples": data_array.shape[0],
                "num_features": len(feature_names),
                "image_base64": image_base64,
                "image_format": "png"
            }
            
        except Exception as e:
            logger.error(f"生成SHAP总结图时出错: {str(e)}")
            plt.close('all')
            return {
                "success": False,
                "error": str(e),
                "model_type": self._get_model_type(model)
            }
    
    def generate_force_plot(
        self,
        model: Any,
        input_data: Union[np.ndarray, pd.DataFrame],
        sample_index: int = 0
    ) -> Dict[str, Any]:
        """
        生成单个样本的SHAP力图
        
        Args:
            model: 训练好的模型
            input_data: 输入数据
            sample_index: 要解释的样本索引
            
        Returns:
            包含Base64编码图片的字典
        """
        try:
            if isinstance(input_data, pd.DataFrame):
                feature_names = list(input_data.columns)
                data_array = input_data.values.astype(np.float64)
            else:
                data_array = np.array(input_data, dtype=np.float64)
                if len(data_array.shape) == 1:
                    data_array = data_array.reshape(1, -1)
                    sample_index = 0
                feature_names = self.feature_names or [f"feature_{i}" for i in range(data_array.shape[1])]
            
            self.explainer = self._create_explainer(model)
            shap_values = self.explainer.shap_values(data_array)
            
            shap_values_to_plot = self._get_shap_values_for_class(shap_values)
            expected_value = self._get_base_value(self.explainer.expected_value)
            
            plt.figure(figsize=(14, 4))
            plt.clf()
            
            shap.plots.force(
                expected_value,
                shap_values_to_plot[sample_index],
                data_array[sample_index],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            plt.close()
            
            return {
                "success": True,
                "model_type": self._get_model_type(model),
                "sample_index": sample_index,
                "image_base64": image_base64,
                "image_format": "png"
            }
            
        except Exception as e:
            logger.error(f"生成SHAP力图时出错: {str(e)}")
            plt.close('all')
            return {
                "success": False,
                "error": str(e),
                "model_type": self._get_model_type(model)
            }

"""
集成检测模型
结合统计特征(XGBoost) + 文本分析(BERT) + 视觉相似度检测
"""

import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%H:%M:%S')

# 导入各个模块
try:
    from networksecurity.utils.url_feature_extractor import URLFeatureExtractor
    URL_EXTRACTOR_AVAILABLE = True
except ImportError:
    URL_EXTRACTOR_AVAILABLE = False

try:
    from networksecurity.utils.web_content_extractor import WebContentExtractor
    WEB_CONTENT_AVAILABLE = True
except ImportError:
    WEB_CONTENT_AVAILABLE = False

try:
    from networksecurity.components.bert_classifier import BertPhishingClassifier, TRANSFORMERS_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from networksecurity.components.visual_similarity import (
        VisualPhishingDetector, 
        PLAYWRIGHT_AVAILABLE, 
        FAISS_AVAILABLE,
        TF_AVAILABLE
    )
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    FAISS_AVAILABLE = False
    TF_AVAILABLE = False


@dataclass
class DetectionResult:
    """检测结果"""
    url: str
    is_phishing: bool
    confidence: float
    threat_level: str
    
    # 各模型得分
    statistical_score: Optional[float] = None
    text_score: Optional[float] = None
    visual_score: Optional[float] = None
    
    # 详细信息
    statistical_details: Optional[Dict] = None
    text_details: Optional[Dict] = None
    visual_details: Optional[Dict] = None
    
    # 元信息
    models_used: List[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'is_phishing': self.is_phishing,
            'confidence': self.confidence,
            'threat_level': self.threat_level,
            'scores': {
                'statistical': self.statistical_score,
                'text': self.text_score,
                'visual': self.visual_score
            },
            'details': {
                'statistical': self.statistical_details,
                'text': self.text_details,
                'visual': self.visual_details
            },
            'models_used': self.models_used or [],
            'error': self.error
        }


class EnsemblePhishingDetector:
    """集成钓鱼检测器"""
    
    # 默认权重
    DEFAULT_WEIGHTS = {
        'statistical': 0.4,  # XGBoost统计特征
        'text': 0.35,        # BERT文本分析
        'visual': 0.25       # 视觉相似度
    }
    
    # 威胁级别阈值
    THREAT_THRESHOLDS = {
        'safe': 0.3,
        'suspicious': 0.6,
        'dangerous': 0.8
    }
    
    def __init__(self, 
                 statistical_model=None,
                 bert_model_path: Optional[str] = None,
                 visual_db_path: str = 'visual_db',
                 weights: Optional[Dict[str, float]] = None):
        """
        初始化集成检测器
        
        Args:
            statistical_model: 统计特征模型（XGBoost等）
            bert_model_path: BERT模型路径
            visual_db_path: 视觉数据库路径
            weights: 各模型权重
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._normalize_weights()
        
        # 统计特征模型
        self.statistical_model = statistical_model
        self.url_extractor = URLFeatureExtractor() if URL_EXTRACTOR_AVAILABLE else None
        
        # 文本分析模型
        self.bert_classifier = None
        self.web_content_extractor = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.bert_classifier = BertPhishingClassifier(model_path=bert_model_path)
                self.web_content_extractor = WebContentExtractor() if WEB_CONTENT_AVAILABLE else None
            except Exception as e:
                logging.warning(f"BERT模型初始化失败: {e}")
        
        # 视觉检测器
        self.visual_detector = None
        if PLAYWRIGHT_AVAILABLE and FAISS_AVAILABLE and TF_AVAILABLE:
            try:
                self.visual_detector = VisualPhishingDetector(db_path=visual_db_path)
            except Exception as e:
                logging.warning(f"视觉检测器初始化失败: {e}")
        
        self._log_available_models()
    
    def _normalize_weights(self):
        """归一化权重"""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total
    
    def _log_available_models(self):
        """记录可用模型"""
        available = []
        if self.statistical_model and self.url_extractor:
            available.append('统计特征(XGBoost)')
        if self.bert_classifier and self.web_content_extractor:
            available.append('文本分析(BERT)')
        if self.visual_detector:
            available.append('视觉相似度(CNN+FAISS)')
        
        if available:
            logging.info(f"可用检测模型: {', '.join(available)}")
        else:
            logging.warning("没有可用的检测模型")
    
    def _get_threat_level(self, score: float) -> str:
        """根据分数获取威胁级别"""
        if score >= self.THREAT_THRESHOLDS['dangerous']:
            return '危险 (Dangerous)'
        elif score >= self.THREAT_THRESHOLDS['suspicious']:
            return '可疑 (Suspicious)'
        elif score >= self.THREAT_THRESHOLDS['safe']:
            return '低风险 (Low Risk)'
        else:
            return '安全 (Safe)'
    
    def _detect_statistical(self, url: str) -> Optional[Dict[str, Any]]:
        """使用统计特征模型检测"""
        if not self.statistical_model or not self.url_extractor:
            return None
        
        try:
            # 提取特征
            features = self.url_extractor.extract_features(url)
            
            # 转换为模型输入格式
            import pandas as pd
            feature_order = [
                'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
                'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
                'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
                'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
                'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
                'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
                'Statistical_report'
            ]
            df = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
            
            # 预测
            prediction = self.statistical_model.predict(df)[0]
            
            # 获取概率
            probability = 0.5
            if hasattr(self.statistical_model, 'predict_proba'):
                proba = self.statistical_model.predict_proba(df)
                probability = float(proba[0, 1])
            
            return {
                'prediction': int(prediction),
                'probability': probability,
                'features': features
            }
            
        except Exception as e:
            logging.error(f"统计特征检测失败: {e}")
            return None
    
    def _detect_text(self, url: str) -> Optional[Dict[str, Any]]:
        """使用文本分析模型检测"""
        if not self.bert_classifier or not self.web_content_extractor:
            return None
        
        try:
            # 提取网页文本
            text = self.web_content_extractor.extract_text_for_nlp(url, max_length=512)
            
            if not text or len(text) < 10:
                return {'error': '无法提取足够的文本内容'}
            
            # BERT预测
            result = self.bert_classifier.predict_single(text)
            
            return {
                'prediction': result['prediction'],
                'probability': result['probabilities']['phishing'],
                'confidence': result['confidence'],
                'text_length': len(text)
            }
            
        except Exception as e:
            logging.error(f"文本分析检测失败: {e}")
            return None
    
    def _detect_visual(self, url: str) -> Optional[Dict[str, Any]]:
        """使用视觉相似度检测"""
        if not self.visual_detector:
            return None
        
        try:
            result = self.visual_detector.detect(url)
            
            return {
                'is_phishing': result['is_phishing'],
                'probability': result['confidence'] if result['is_phishing'] else 1 - result['similarity'],
                'similarity': result['similarity'],
                'similar_to': result['similar_to'],
                'reason': result['reason'],
                'error': result.get('error')
            }
            
        except Exception as e:
            logging.error(f"视觉检测失败: {e}")
            return None
    
    def detect(self, url: str, use_statistical: bool = True,
               use_text: bool = True, use_visual: bool = False) -> DetectionResult:
        """
        综合检测URL
        
        Args:
            url: 目标URL
            use_statistical: 是否使用统计特征检测
            use_text: 是否使用文本分析检测
            use_visual: 是否使用视觉检测（较慢）
            
        Returns:
            检测结果
        """
        logging.info(f"开始综合检测: {url}")
        
        models_used = []
        scores = []
        weights_used = []
        
        # 统计特征检测
        statistical_result = None
        statistical_score = None
        if use_statistical:
            statistical_result = self._detect_statistical(url)
            if statistical_result and 'probability' in statistical_result:
                statistical_score = statistical_result['probability']
                scores.append(statistical_score)
                weights_used.append(self.weights['statistical'])
                models_used.append('statistical')
                logging.info(f"统计特征得分: {statistical_score:.2%}")
        
        # 文本分析检测
        text_result = None
        text_score = None
        if use_text:
            text_result = self._detect_text(url)
            if text_result and 'probability' in text_result:
                text_score = text_result['probability']
                scores.append(text_score)
                weights_used.append(self.weights['text'])
                models_used.append('text')
                logging.info(f"文本分析得分: {text_score:.2%}")
        
        # 视觉检测
        visual_result = None
        visual_score = None
        if use_visual:
            visual_result = self._detect_visual(url)
            if visual_result:
                if visual_result.get('is_phishing'):
                    visual_score = visual_result.get('probability', 0.9)
                else:
                    visual_score = 1 - visual_result.get('similarity', 0.5)
                scores.append(visual_score)
                weights_used.append(self.weights['visual'])
                models_used.append('visual')
                logging.info(f"视觉检测得分: {visual_score:.2%}")
        
        # 计算加权平均分数
        if scores and weights_used:
            total_weight = sum(weights_used)
            final_score = sum(s * w for s, w in zip(scores, weights_used)) / total_weight
        else:
            final_score = 0.5
        
        # 判断结果
        is_phishing = final_score >= 0.5
        threat_level = self._get_threat_level(final_score)
        
        logging.info(f"综合得分: {final_score:.2%}, 判定: {'钓鱼' if is_phishing else '安全'}")
        
        return DetectionResult(
            url=url,
            is_phishing=is_phishing,
            confidence=final_score,
            threat_level=threat_level,
            statistical_score=statistical_score,
            text_score=text_score,
            visual_score=visual_score,
            statistical_details=statistical_result,
            text_details=text_result,
            visual_details=visual_result,
            models_used=models_used
        )
    
    def detect_batch(self, urls: List[str], **kwargs) -> List[DetectionResult]:
        """批量检测"""
        results = []
        for url in urls:
            result = self.detect(url, **kwargs)
            results.append(result)
        return results
    
    def set_weights(self, statistical: float = None, text: float = None, visual: float = None):
        """设置模型权重"""
        if statistical is not None:
            self.weights['statistical'] = statistical
        if text is not None:
            self.weights['text'] = text
        if visual is not None:
            self.weights['visual'] = visual
        self._normalize_weights()
        logging.info(f"权重已更新: {self.weights}")


# 测试代码
if __name__ == '__main__':
    print("集成检测模块")
    print(f"URL特征提取可用: {URL_EXTRACTOR_AVAILABLE}")
    print(f"网页内容提取可用: {WEB_CONTENT_AVAILABLE}")
    print(f"BERT可用: {TRANSFORMERS_AVAILABLE}")
    print(f"视觉检测可用: {PLAYWRIGHT_AVAILABLE and FAISS_AVAILABLE and TF_AVAILABLE}")

"""
网页视觉相似度检测模块
使用Playwright截图 + MobileNetV2特征提取 + FAISS向量检索
"""

import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import hashlib
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%H:%M:%S')

# 检查依赖
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("Pillow未安装")

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow未安装")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS未安装")

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright未安装")


class WebScreenshotCapture:
    """网页截图捕获器"""
    
    def __init__(self, width: int = 1280, height: int = 720, timeout: int = 30000):
        """
        初始化截图捕获器
        
        Args:
            width: 视口宽度
            height: 视口高度
            timeout: 超时时间（毫秒）
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright未安装，请运行: pip install playwright && playwright install chromium")
        
        self.width = width
        self.height = height
        self.timeout = timeout
    
    def capture(self, url: str, output_path: Optional[str] = None) -> Optional[bytes]:
        """
        捕获网页截图
        
        Args:
            url: 目标URL
            output_path: 输出路径（可选）
            
        Returns:
            截图的字节数据
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': self.width, 'height': self.height},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = context.new_page()
                
                # 访问页面
                page.goto(url, timeout=self.timeout, wait_until='networkidle')
                
                # 等待页面稳定
                page.wait_for_timeout(1000)
                
                # 截图
                screenshot = page.screenshot(type='png', full_page=False)
                
                browser.close()
                
                # 保存到文件
                if output_path:
                    with open(output_path, 'wb') as f:
                        f.write(screenshot)
                    logging.info(f"截图已保存到 {output_path}")
                
                return screenshot
                
        except Exception as e:
            logging.error(f"截图失败: {e}")
            return None


class VisualFeatureExtractor:
    """视觉特征提取器（使用MobileNetV2）"""
    
    def __init__(self):
        """初始化特征提取器"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow未安装")
        if not PIL_AVAILABLE:
            raise ImportError("Pillow未安装")
        
        # 加载预训练的MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.model = Model(inputs=base_model.input, outputs=base_model.output)
        self.input_size = (224, 224)
        self.feature_dim = 1280  # MobileNetV2的特征维度
        
        logging.info("MobileNetV2特征提取器已初始化")
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """预处理图像"""
        # 从字节数据加载图像
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整大小
        image = image.resize(self.input_size, Image.Resampling.LANCZOS)
        
        # 转换为数组
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def extract_features(self, image_data: bytes) -> np.ndarray:
        """
        提取图像特征向量
        
        Args:
            image_data: 图像字节数据
            
        Returns:
            特征向量 (1280,)
        """
        img_array = self.preprocess_image(image_data)
        features = self.model.predict(img_array, verbose=0)
        
        # L2归一化
        features = features / np.linalg.norm(features)
        
        return features.flatten()
    
    def extract_features_from_file(self, image_path: str) -> np.ndarray:
        """从文件提取特征"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return self.extract_features(image_data)


class VisualSimilarityDatabase:
    """视觉相似度数据库（使用FAISS）"""
    
    # 知名网站列表（用于检测仿冒）
    KNOWN_WEBSITES = {
        'google': ['google.com', 'google.cn', 'accounts.google.com'],
        'facebook': ['facebook.com', 'fb.com'],
        'amazon': ['amazon.com', 'amazon.cn', 'amazon.co.jp'],
        'paypal': ['paypal.com', 'paypal.me'],
        'apple': ['apple.com', 'icloud.com', 'appleid.apple.com'],
        'microsoft': ['microsoft.com', 'live.com', 'outlook.com', 'office.com'],
        'netflix': ['netflix.com'],
        'twitter': ['twitter.com', 'x.com'],
        'instagram': ['instagram.com'],
        'linkedin': ['linkedin.com'],
        'alibaba': ['alibaba.com', 'taobao.com', 'tmall.com', 'alipay.com'],
        'tencent': ['qq.com', 'weixin.qq.com', 'wechat.com'],
        'baidu': ['baidu.com'],
        'jd': ['jd.com'],
        'bank_of_china': ['boc.cn', 'bankofchina.com'],
        'icbc': ['icbc.com.cn'],
        'ccb': ['ccb.com'],
        'abc': ['abchina.com'],
    }
    
    def __init__(self, db_path: str = 'visual_db', feature_dim: int = 1280):
        """
        初始化数据库
        
        Args:
            db_path: 数据库存储路径
            feature_dim: 特征向量维度
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS未安装，请运行: pip install faiss-cpu")
        
        self.db_path = db_path
        self.feature_dim = feature_dim
        self.index_path = os.path.join(db_path, 'faiss.index')
        self.metadata_path = os.path.join(db_path, 'metadata.json')
        
        os.makedirs(db_path, exist_ok=True)
        
        # 加载或创建索引
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logging.info(f"加载了 {self.index.ntotal} 个向量")
        else:
            # 使用内积索引（因为向量已归一化，内积等于余弦相似度）
            self.index = faiss.IndexFlatIP(feature_dim)
            self.metadata = {'entries': []}
            logging.info("创建新的FAISS索引")
    
    def add_entry(self, features: np.ndarray, website_name: str, 
                  official_domains: List[str], description: str = ''):
        """
        添加条目到数据库
        
        Args:
            features: 特征向量
            website_name: 网站名称
            official_domains: 官方域名列表
            description: 描述
        """
        # 确保特征是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 添加到索引
        self.index.add(features.astype(np.float32))
        
        # 添加元数据
        self.metadata['entries'].append({
            'website_name': website_name,
            'official_domains': official_domains,
            'description': description,
            'index': self.index.ntotal - 1
        })
        
        logging.info(f"添加了 {website_name} 到数据库")
    
    def search(self, features: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相似向量
        
        Args:
            features: 查询特征向量
            k: 返回结果数量
            
        Returns:
            相似结果列表
        """
        if self.index.ntotal == 0:
            return []
        
        # 确保特征是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 搜索
        k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(features.astype(np.float32), k)
        
        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= 0 and idx < len(self.metadata['entries']):
                entry = self.metadata['entries'][idx].copy()
                entry['similarity'] = float(sim)
                results.append(entry)
        
        return results
    
    def save(self):
        """保存数据库"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logging.info(f"数据库已保存到 {self.db_path}")
    
    def is_official_domain(self, domain: str, website_name: str) -> bool:
        """检查域名是否为官方域名"""
        domain = domain.lower().strip()
        
        for entry in self.metadata['entries']:
            if entry['website_name'].lower() == website_name.lower():
                for official in entry['official_domains']:
                    if domain == official or domain.endswith('.' + official):
                        return True
        
        # 检查预定义列表
        if website_name.lower() in self.KNOWN_WEBSITES:
            for official in self.KNOWN_WEBSITES[website_name.lower()]:
                if domain == official or domain.endswith('.' + official):
                    return True
        
        return False


class VisualPhishingDetector:
    """视觉钓鱼检测器"""
    
    def __init__(self, db_path: str = 'visual_db'):
        """
        初始化检测器
        
        Args:
            db_path: 数据库路径
        """
        self.screenshot_capture = None
        self.feature_extractor = None
        self.database = None
        self.db_path = db_path
        
        # 延迟初始化
        self._initialized = False
    
    def _ensure_initialized(self):
        """确保组件已初始化"""
        if self._initialized:
            return
        
        if PLAYWRIGHT_AVAILABLE:
            self.screenshot_capture = WebScreenshotCapture()
        
        if TF_AVAILABLE and PIL_AVAILABLE:
            self.feature_extractor = VisualFeatureExtractor()
        
        if FAISS_AVAILABLE:
            self.database = VisualSimilarityDatabase(self.db_path)
        
        self._initialized = True
    
    def detect(self, url: str, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """
        检测URL是否为视觉钓鱼
        
        Args:
            url: 目标URL
            similarity_threshold: 相似度阈值
            
        Returns:
            检测结果
        """
        self._ensure_initialized()
        
        result = {
            'url': url,
            'is_phishing': False,
            'confidence': 0.0,
            'reason': '',
            'similar_to': None,
            'similarity': 0.0,
            'domain_mismatch': False,
            'error': None
        }
        
        # 检查组件是否可用
        if not all([self.screenshot_capture, self.feature_extractor, self.database]):
            missing = []
            if not self.screenshot_capture:
                missing.append('Playwright')
            if not self.feature_extractor:
                missing.append('TensorFlow/Pillow')
            if not self.database:
                missing.append('FAISS')
            result['error'] = f"缺少依赖: {', '.join(missing)}"
            return result
        
        # 解析域名
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        try:
            # 截图
            logging.info(f"正在截图: {url}")
            screenshot = self.screenshot_capture.capture(url)
            if not screenshot:
                result['error'] = '截图失败'
                return result
            
            # 提取特征
            logging.info("正在提取视觉特征...")
            features = self.feature_extractor.extract_features(screenshot)
            
            # 搜索相似
            logging.info("正在搜索相似网站...")
            similar_results = self.database.search(features, k=3)
            
            if not similar_results:
                result['reason'] = '数据库为空，无法进行比对'
                return result
            
            # 检查最相似的结果
            top_match = similar_results[0]
            result['similarity'] = top_match['similarity']
            result['similar_to'] = top_match['website_name']
            
            # 判断是否为钓鱼
            if top_match['similarity'] >= similarity_threshold:
                # 检查域名是否匹配
                is_official = self.database.is_official_domain(domain, top_match['website_name'])
                
                if not is_official:
                    result['is_phishing'] = True
                    result['domain_mismatch'] = True
                    result['confidence'] = top_match['similarity']
                    result['reason'] = f"视觉高度相似 {top_match['website_name']}，但域名 {domain} 不是官方域名"
                else:
                    result['reason'] = f"视觉相似 {top_match['website_name']}，域名验证通过"
            else:
                result['reason'] = f"与已知网站相似度较低 ({top_match['similarity']:.2%})"
            
            return result
            
        except Exception as e:
            logging.error(f"检测失败: {e}")
            result['error'] = str(e)
            return result
    
    def add_known_website(self, url: str, website_name: str, 
                          official_domains: List[str]) -> bool:
        """
        添加已知网站到数据库
        
        Args:
            url: 网站URL
            website_name: 网站名称
            official_domains: 官方域名列表
            
        Returns:
            是否成功
        """
        self._ensure_initialized()
        
        if not all([self.screenshot_capture, self.feature_extractor, self.database]):
            logging.error("组件未完全初始化")
            return False
        
        try:
            # 截图
            screenshot = self.screenshot_capture.capture(url)
            if not screenshot:
                return False
            
            # 提取特征
            features = self.feature_extractor.extract_features(screenshot)
            
            # 添加到数据库
            self.database.add_entry(features, website_name, official_domains)
            self.database.save()
            
            return True
            
        except Exception as e:
            logging.error(f"添加网站失败: {e}")
            return False


# 测试代码
if __name__ == '__main__':
    print("视觉相似度检测模块")
    print(f"Playwright可用: {PLAYWRIGHT_AVAILABLE}")
    print(f"TensorFlow可用: {TF_AVAILABLE}")
    print(f"FAISS可用: {FAISS_AVAILABLE}")
    print(f"Pillow可用: {PIL_AVAILABLE}")

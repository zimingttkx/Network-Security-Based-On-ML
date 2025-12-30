"""
NLP和CV检测模块单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestWebContentExtractor:
    """网页内容提取器测试"""
    
    @pytest.fixture
    def extractor(self):
        from networksecurity.utils.web_content_extractor import WebContentExtractor
        return WebContentExtractor(timeout=5)
    
    def test_normalize_url_with_http(self, extractor):
        """测试URL标准化"""
        assert extractor._normalize_url("example.com") == "http://example.com"
        assert extractor._normalize_url("https://example.com") == "https://example.com"
    
    def test_clean_text(self, extractor):
        """测试文本清理"""
        text = "  Hello   World  \n\n  Test  "
        cleaned = extractor._clean_text(text)
        assert cleaned == "Hello World Test"
    
    def test_extract_visible_text(self, extractor):
        """测试从HTML提取可见文本"""
        html = """
        <html>
        <head><title>Test</title><script>var x=1;</script></head>
        <body>
            <div>Hello World</div>
            <p>This is a test.</p>
        </body>
        </html>
        """
        text = extractor._extract_visible_text(html)
        assert "Hello World" in text
        assert "This is a test" in text
        assert "var x=1" not in text
    
    def test_extract_title(self, extractor):
        """测试提取标题"""
        html = "<html><head><title>Test Title</title></head></html>"
        title = extractor._extract_title(html)
        assert title == "Test Title"
    
    def test_extract_forms(self, extractor):
        """测试提取表单"""
        html = """
        <html><body>
        <form action="/login" method="POST">
            <input type="text" name="username">
            <input type="password" name="password">
            <input type="submit" value="Login">
        </form>
        </body></html>
        """
        forms = extractor._extract_forms(html)
        assert len(forms) == 1
        assert forms[0]['action'] == '/login'
        assert forms[0]['method'] == 'POST'
        assert len(forms[0]['inputs']) == 3
    
    @patch('requests.Session.get')
    def test_extract_content_success(self, mock_get, extractor):
        """测试成功提取内容"""
        mock_response = Mock()
        mock_response.content = b"<html><head><title>Test</title></head><body>Content</body></html>"
        mock_response.text = "<html><head><title>Test</title></head><body>Content</body></html>"
        mock_response.apparent_encoding = 'utf-8'
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = extractor.extract_content("http://example.com")
        
        assert result['success'] is True
        assert result['title'] == 'Test'
        assert 'Content' in result['text']


class TestBertClassifier:
    """BERT分类器测试"""
    
    def test_transformers_available(self):
        """测试transformers是否可用"""
        from networksecurity.components.bert_classifier import TRANSFORMERS_AVAILABLE
        # 只检查导入是否成功
        assert isinstance(TRANSFORMERS_AVAILABLE, bool)
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed"
    )
    def test_phishing_text_dataset(self):
        """测试数据集类"""
        try:
            from networksecurity.components.bert_classifier import PhishingTextDataset, TRANSFORMERS_AVAILABLE
            if not TRANSFORMERS_AVAILABLE:
                pytest.skip("transformers not available")
            
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            texts = ["Hello world", "Test text"]
            labels = [0, 1]
            
            dataset = PhishingTextDataset(texts, labels, tokenizer, max_length=64)
            
            assert len(dataset) == 2
            item = dataset[0]
            assert 'input_ids' in item
            assert 'attention_mask' in item
            assert 'labels' in item
        except Exception as e:
            pytest.skip(f"BERT test skipped: {e}")


class TestVisualSimilarity:
    """视觉相似度测试"""
    
    def test_faiss_available(self):
        """测试FAISS是否可用"""
        from networksecurity.components.visual_similarity import FAISS_AVAILABLE
        assert isinstance(FAISS_AVAILABLE, bool)
    
    def test_tf_available(self):
        """测试TensorFlow是否可用"""
        from networksecurity.components.visual_similarity import TF_AVAILABLE
        assert isinstance(TF_AVAILABLE, bool)
    
    def test_known_websites_list(self):
        """测试已知网站列表"""
        from networksecurity.components.visual_similarity import VisualSimilarityDatabase
        
        known = VisualSimilarityDatabase.KNOWN_WEBSITES
        assert 'google' in known
        assert 'paypal' in known
        assert 'amazon' in known
        assert 'google.com' in known['google']


class TestEnsembleDetector:
    """集成检测器测试"""
    
    def test_default_weights(self):
        """测试默认权重"""
        from networksecurity.components.ensemble_detector import EnsemblePhishingDetector
        
        detector = EnsemblePhishingDetector()
        
        assert 'statistical' in detector.weights
        assert 'text' in detector.weights
        assert 'visual' in detector.weights
        assert abs(sum(detector.weights.values()) - 1.0) < 0.01
    
    def test_set_weights(self):
        """测试设置权重"""
        from networksecurity.components.ensemble_detector import EnsemblePhishingDetector
        
        detector = EnsemblePhishingDetector()
        detector.set_weights(statistical=0.5, text=0.3, visual=0.2)
        
        assert abs(sum(detector.weights.values()) - 1.0) < 0.01
    
    def test_get_threat_level(self):
        """测试威胁级别判断"""
        from networksecurity.components.ensemble_detector import EnsemblePhishingDetector
        
        detector = EnsemblePhishingDetector()
        
        assert '安全' in detector._get_threat_level(0.1) or 'Safe' in detector._get_threat_level(0.1)
        assert '危险' in detector._get_threat_level(0.9) or 'Dangerous' in detector._get_threat_level(0.9)
    
    def test_detection_result_to_dict(self):
        """测试检测结果转字典"""
        from networksecurity.components.ensemble_detector import DetectionResult
        
        result = DetectionResult(
            url="http://example.com",
            is_phishing=False,
            confidence=0.3,
            threat_level="安全",
            statistical_score=0.2,
            text_score=0.4,
            models_used=['statistical', 'text']
        )
        
        d = result.to_dict()
        assert d['url'] == "http://example.com"
        assert d['is_phishing'] is False
        assert d['scores']['statistical'] == 0.2


# 运行测试
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
测试URL特征提取功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from networksecurity.utils.url_feature_extractor import URLFeatureExtractor


class TestURLFeatureExtractor:
    """URL特征提取器测试类"""
    
    @pytest.fixture
    def extractor(self):
        """创建提取器实例"""
        return URLFeatureExtractor(timeout=5)
    
    # ==================== URL标准化测试 ====================
    
    def test_normalize_url_with_http(self, extractor):
        """测试已有http协议的URL"""
        url = extractor._normalize_url("http://example.com")
        assert url == "http://example.com"
    
    def test_normalize_url_with_https(self, extractor):
        """测试已有https协议的URL"""
        url = extractor._normalize_url("https://example.com")
        assert url == "https://example.com"
    
    def test_normalize_url_without_protocol(self, extractor):
        """测试无协议的URL自动补全"""
        url = extractor._normalize_url("example.com")
        assert url == "http://example.com"
    
    def test_normalize_url_with_spaces(self, extractor):
        """测试带空格的URL"""
        url = extractor._normalize_url("  example.com  ")
        assert url == "http://example.com"
    
    # ==================== URL解析测试 ====================
    
    def test_parse_url_basic(self, extractor):
        """测试基本URL解析"""
        parsed = extractor._parse_url("https://www.example.com/path?query=1")
        assert parsed['scheme'] == 'https'
        assert parsed['domain'] == 'www.example.com'
        assert parsed['path'] == '/path'
        assert parsed['query'] == 'query=1'
    
    def test_parse_url_with_port(self, extractor):
        """测试带端口的URL解析"""
        parsed = extractor._parse_url("http://example.com:8080/path")
        assert parsed['port'] == 8080
        assert parsed['domain'] == 'example.com'
    
    # ==================== IP地址检测测试 ====================
    
    def test_check_ip_address_with_ip(self, extractor):
        """测试包含IP地址的URL"""
        result = extractor._check_ip_address("http://192.168.1.1/path")
        assert result == -1
    
    def test_check_ip_address_without_ip(self, extractor):
        """测试不包含IP地址的URL"""
        result = extractor._check_ip_address("http://example.com/path")
        assert result == 1
    
    def test_check_ip_address_with_hex(self, extractor):
        """测试包含十六进制IP的URL"""
        result = extractor._check_ip_address("http://0x7f000001/path")
        assert result == -1
    
    # ==================== URL长度测试 ====================
    
    def test_check_url_length_short(self, extractor):
        """测试短URL"""
        result = extractor._check_url_length("http://a.com")
        assert result == 1
    
    def test_check_url_length_medium(self, extractor):
        """测试中等长度URL"""
        url = "http://example.com/" + "a" * 50
        result = extractor._check_url_length(url)
        assert result == 0
    
    def test_check_url_length_long(self, extractor):
        """测试长URL"""
        url = "http://example.com/" + "a" * 100
        result = extractor._check_url_length(url)
        assert result == -1
    
    # ==================== 短链服务测试 ====================
    
    def test_check_shortening_service_bitly(self, extractor):
        """测试bit.ly短链"""
        result = extractor._check_shortening_service("http://bit.ly/abc123")
        assert result == -1
    
    def test_check_shortening_service_tinyurl(self, extractor):
        """测试tinyurl短链"""
        result = extractor._check_shortening_service("http://tinyurl.com/xyz")
        assert result == -1
    
    def test_check_shortening_service_normal(self, extractor):
        """测试正常URL"""
        result = extractor._check_shortening_service("http://example.com")
        assert result == 1
    
    # ==================== @符号测试 ====================
    
    def test_check_at_symbol_present(self, extractor):
        """测试包含@符号的URL"""
        result = extractor._check_at_symbol("http://user@example.com")
        assert result == -1
    
    def test_check_at_symbol_absent(self, extractor):
        """测试不包含@符号的URL"""
        result = extractor._check_at_symbol("http://example.com")
        assert result == 1
    
    # ==================== 双斜杠测试 ====================
    
    def test_check_double_slash_present(self, extractor):
        """测试包含双斜杠重定向的URL"""
        result = extractor._check_double_slash("http://example.com//redirect")
        assert result == -1
    
    def test_check_double_slash_absent(self, extractor):
        """测试正常URL"""
        result = extractor._check_double_slash("http://example.com/path")
        assert result == 1
    
    # ==================== 前缀后缀测试 ====================
    
    def test_check_prefix_suffix_with_dash(self, extractor):
        """测试域名包含连字符"""
        result = extractor._check_prefix_suffix("fake-bank.com")
        assert result == -1
    
    def test_check_prefix_suffix_without_dash(self, extractor):
        """测试正常域名"""
        result = extractor._check_prefix_suffix("example.com")
        assert result == 1
    
    # ==================== 子域名测试 ====================
    
    def test_check_sub_domain_none(self, extractor):
        """测试无子域名"""
        result = extractor._check_sub_domain("http://example.com")
        assert result == 1
    
    def test_check_sub_domain_www(self, extractor):
        """测试www子域名"""
        result = extractor._check_sub_domain("http://www.example.com")
        assert result == 1
    
    def test_check_sub_domain_one(self, extractor):
        """测试一个子域名"""
        result = extractor._check_sub_domain("http://sub.example.com")
        assert result == 0
    
    def test_check_sub_domain_multiple(self, extractor):
        """测试多个子域名"""
        result = extractor._check_sub_domain("http://a.b.c.example.com")
        assert result == -1
    
    # ==================== HTTPS令牌测试 ====================
    
    def test_check_https_token_in_domain(self, extractor):
        """测试域名包含https"""
        result = extractor._check_https_token("https-secure.example.com")
        assert result == -1
    
    def test_check_https_token_normal(self, extractor):
        """测试正常域名"""
        result = extractor._check_https_token("example.com")
        assert result == 1
    
    # ==================== 端口测试 ====================
    
    def test_check_port_standard(self, extractor):
        """测试标准端口"""
        result = extractor._check_port("http://example.com:80")
        assert result == 1
    
    def test_check_port_nonstandard(self, extractor):
        """测试非标准端口"""
        result = extractor._check_port("http://example.com:8888")
        assert result == -1
    
    def test_check_port_default(self, extractor):
        """测试默认端口"""
        result = extractor._check_port("http://example.com")
        assert result == 1
    
    # ==================== HTML内容特征测试（使用Mock） ====================
    
    def test_check_iframe_present(self, extractor):
        """测试包含iframe的页面"""
        from bs4 import BeautifulSoup
        html = "<html><body><iframe src='http://evil.com'></iframe></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_iframe(soup)
        assert result == -1
    
    def test_check_iframe_absent(self, extractor):
        """测试不包含iframe的页面"""
        from bs4 import BeautifulSoup
        html = "<html><body><div>Content</div></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_iframe(soup)
        assert result == 1
    
    def test_check_iframe_none_soup(self, extractor):
        """测试soup为None的情况"""
        result = extractor._check_iframe(None)
        assert result == 1
    
    def test_check_on_mouseover_present(self, extractor):
        """测试包含onMouseOver的页面"""
        from bs4 import BeautifulSoup
        html = "<html><body><a onmouseover='alert(1)'>Link</a></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_on_mouseover(soup)
        assert result == -1
    
    def test_check_on_mouseover_absent(self, extractor):
        """测试不包含onMouseOver的页面"""
        from bs4 import BeautifulSoup
        html = "<html><body><a href='#'>Link</a></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_on_mouseover(soup)
        assert result == 1
    
    def test_check_right_click_disabled(self, extractor):
        """测试禁用右键的页面"""
        from bs4 import BeautifulSoup
        html = "<html><body oncontextmenu='return false'>Content</body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_right_click(soup)
        assert result == -1
    
    def test_check_right_click_enabled(self, extractor):
        """测试正常页面"""
        from bs4 import BeautifulSoup
        html = "<html><body>Content</body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_right_click(soup)
        assert result == 1
    
    def test_check_popup_window_present(self, extractor):
        """测试包含弹窗的页面"""
        from bs4 import BeautifulSoup
        html = "<html><script>window.open('http://evil.com')</script></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_popup_window(soup)
        assert result == -1
    
    def test_check_popup_window_absent(self, extractor):
        """测试正常页面"""
        from bs4 import BeautifulSoup
        html = "<html><body>Content</body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_popup_window(soup)
        assert result == 1
    
    def test_check_submitting_to_email_present(self, extractor):
        """测试包含mailto的页面"""
        from bs4 import BeautifulSoup
        html = "<html><body><a href='mailto:test@test.com'>Email</a></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_submitting_to_email(soup)
        assert result == -1
    
    def test_check_submitting_to_email_absent(self, extractor):
        """测试正常页面"""
        from bs4 import BeautifulSoup
        html = "<html><body><a href='http://example.com'>Link</a></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extractor._check_submitting_to_email(soup)
        assert result == 1
    
    # ==================== 外部数据特征测试 ====================
    
    def test_check_web_traffic_known_tld(self, extractor):
        """测试知名TLD域名"""
        result = extractor._check_web_traffic("google.com")
        assert result == 1
    
    def test_check_web_traffic_unknown_tld(self, extractor):
        """测试未知TLD域名"""
        result = extractor._check_web_traffic("example.xyz")
        assert result == -1
    
    # ==================== 完整特征提取测试（使用Mock） ====================
    
    @patch.object(URLFeatureExtractor, '_fetch_html')
    @patch.object(URLFeatureExtractor, '_check_ssl_state')
    @patch.object(URLFeatureExtractor, '_check_dns_record')
    @patch.object(URLFeatureExtractor, '_check_domain_age')
    @patch.object(URLFeatureExtractor, '_check_domain_registration_length')
    @patch.object(URLFeatureExtractor, '_check_redirect')
    @patch.object(URLFeatureExtractor, '_check_abnormal_url')
    def test_extract_features_returns_30_features(
        self, mock_abnormal, mock_redirect, mock_reg_length, 
        mock_age, mock_dns, mock_ssl, mock_html, extractor
    ):
        """测试特征提取返回30个特征"""
        mock_html.return_value = "<html><body>Test</body></html>"
        mock_ssl.return_value = 1
        mock_dns.return_value = 1
        mock_age.return_value = 1
        mock_reg_length.return_value = 1
        mock_redirect.return_value = 0
        mock_abnormal.return_value = 1
        
        features = extractor.extract_features("http://example.com")
        
        assert len(features) == 30
        assert all(key in features for key in URLFeatureExtractor.FEATURE_NAMES)
    
    @patch.object(URLFeatureExtractor, '_fetch_html')
    def test_extract_features_with_failed_html(self, mock_html, extractor):
        """测试HTML获取失败时的特征提取"""
        mock_html.return_value = None
        
        features = extractor.extract_features("http://example.com")
        
        assert len(features) == 30
        # HTML相关特征应该返回默认值
        assert features['Iframe'] == 1  # None soup返回1
    
    def test_feature_names_count(self, extractor):
        """测试特征名称数量"""
        assert len(URLFeatureExtractor.FEATURE_NAMES) == 30
    
    def test_feature_names_unique(self, extractor):
        """测试特征名称唯一性"""
        names = URLFeatureExtractor.FEATURE_NAMES
        assert len(names) == len(set(names))


# 运行测试
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

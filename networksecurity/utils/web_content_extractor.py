"""
网页内容提取模块
用于爬取网页并提取可见文本内容，供NLP模型分析
"""

import re
import logging
import requests
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%H:%M:%S')


class WebContentExtractor:
    """网页内容提取器"""
    
    # 需要移除的标签
    REMOVE_TAGS = ['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer', 'nav', 'aside']
    
    # 常见的用户代理
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    def __init__(self, timeout: int = 15, max_content_length: int = 100000):
        """
        初始化提取器
        
        Args:
            timeout: 请求超时时间（秒）
            max_content_length: 最大内容长度（字节）
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.USER_AGENTS[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def _normalize_url(self, url: str) -> str:
        """标准化URL"""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        return url
    
    def _fetch_html(self, url: str) -> Optional[str]:
        """获取网页HTML内容"""
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                verify=False  # 忽略SSL证书验证
            )
            response.raise_for_status()
            
            # 检查内容长度
            if len(response.content) > self.max_content_length:
                logging.warning(f"内容过大，截断到 {self.max_content_length} 字节")
                return response.content[:self.max_content_length].decode('utf-8', errors='ignore')
            
            # 尝试检测编码
            response.encoding = response.apparent_encoding or 'utf-8'
            return response.text
            
        except requests.exceptions.Timeout:
            logging.warning(f"请求超时: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.warning(f"请求失败: {url}, 错误: {e}")
            return None
        except Exception as e:
            logging.error(f"获取HTML时出错: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    
    def _extract_visible_text(self, html: str) -> str:
        """从HTML中提取可见文本"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # 移除不需要的标签
            for tag in self.REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # 获取body内容
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            return self._clean_text(text)
            
        except Exception as e:
            logging.error(f"提取文本时出错: {e}")
            return ""
    
    def _extract_title(self, html: str) -> str:
        """提取网页标题"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                return self._clean_text(title_tag.get_text())
            return ""
        except Exception:
            return ""
    
    def _extract_meta_description(self, html: str) -> str:
        """提取meta描述"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and meta.get('content'):
                return self._clean_text(meta['content'])
            return ""
        except Exception:
            return ""
    
    def _extract_forms(self, html: str) -> list:
        """提取表单信息"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            forms = []
            for form in soup.find_all('form'):
                form_info = {
                    'action': form.get('action', ''),
                    'method': form.get('method', 'get').upper(),
                    'inputs': []
                }
                for inp in form.find_all('input'):
                    input_info = {
                        'type': inp.get('type', 'text'),
                        'name': inp.get('name', ''),
                        'placeholder': inp.get('placeholder', '')
                    }
                    form_info['inputs'].append(input_info)
                forms.append(form_info)
            return forms
        except Exception:
            return []
    
    def _extract_links(self, html: str, base_url: str) -> Dict[str, int]:
        """提取链接统计"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            parsed_base = urlparse(base_url)
            base_domain = parsed_base.netloc
            
            internal_links = 0
            external_links = 0
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith(('javascript:', 'mailto:', '#')):
                    continue
                    
                parsed_href = urlparse(href)
                if parsed_href.netloc == '' or parsed_href.netloc == base_domain:
                    internal_links += 1
                else:
                    external_links += 1
            
            return {
                'internal': internal_links,
                'external': external_links,
                'total': internal_links + external_links
            }
        except Exception:
            return {'internal': 0, 'external': 0, 'total': 0}
    
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        提取网页内容
        
        Args:
            url: 目标URL
            
        Returns:
            包含提取内容的字典
        """
        url = self._normalize_url(url)
        
        result = {
            'url': url,
            'success': False,
            'title': '',
            'description': '',
            'text': '',
            'text_length': 0,
            'forms': [],
            'links': {'internal': 0, 'external': 0, 'total': 0},
            'has_login_form': False,
            'has_password_field': False,
            'error': None
        }
        
        # 获取HTML
        html = self._fetch_html(url)
        if not html:
            result['error'] = '无法获取网页内容'
            return result
        
        # 提取各种内容
        result['title'] = self._extract_title(html)
        result['description'] = self._extract_meta_description(html)
        result['text'] = self._extract_visible_text(html)
        result['text_length'] = len(result['text'])
        result['forms'] = self._extract_forms(html)
        result['links'] = self._extract_links(html, url)
        
        # 检测登录表单
        for form in result['forms']:
            for inp in form['inputs']:
                if inp['type'] == 'password':
                    result['has_password_field'] = True
                    result['has_login_form'] = True
                    break
        
        result['success'] = True
        return result
    
    def extract_text_for_nlp(self, url: str, max_length: int = 512) -> str:
        """
        提取用于NLP分析的文本
        
        Args:
            url: 目标URL
            max_length: 最大文本长度（用于BERT等模型）
            
        Returns:
            处理后的文本
        """
        content = self.extract_content(url)
        
        if not content['success']:
            return ""
        
        # 组合标题、描述和正文
        parts = []
        if content['title']:
            parts.append(content['title'])
        if content['description']:
            parts.append(content['description'])
        if content['text']:
            parts.append(content['text'])
        
        combined_text = ' '.join(parts)
        
        # 截断到最大长度
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length]
        
        return combined_text


# 测试代码
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    extractor = WebContentExtractor(timeout=10)
    
    # 测试提取
    test_url = "https://www.google.com"
    print(f"测试URL: {test_url}")
    
    content = extractor.extract_content(test_url)
    print(f"成功: {content['success']}")
    print(f"标题: {content['title']}")
    print(f"文本长度: {content['text_length']}")
    print(f"链接统计: {content['links']}")
    print(f"有登录表单: {content['has_login_form']}")
    
    # 测试NLP文本提取
    nlp_text = extractor.extract_text_for_nlp(test_url)
    print(f"\nNLP文本 (前200字符): {nlp_text[:200]}...")

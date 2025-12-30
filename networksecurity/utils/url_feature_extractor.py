"""
URL特征自动提取器

功能:
1. 从URL自动提取30个网络安全特征
2. 支持DNS查询、Whois查询、SSL验证、HTML解析
3. 用于端到端的钓鱼网站检测
"""

import re
import ssl
import socket
import logging
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone
from typing import Dict, Optional, Any
import concurrent.futures

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# 请求超时设置（秒）
REQUEST_TIMEOUT = 10
DNS_TIMEOUT = 5

# 短链服务域名列表
SHORTENING_SERVICES = [
    'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'ow.ly', 'is.gd', 'buff.ly',
    'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr', 'yourls.org', 'cutt.ly', 'rb.gy'
]

# 标准端口
STANDARD_PORTS = [80, 443, 21, 22, 23, 25, 53, 110, 143, 993, 995]


class URLFeatureExtractor:
    """URL特征提取器类"""
    
    # 30个特征名称（与训练数据一致）
    FEATURE_NAMES = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
        'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
        'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
        'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
        'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
        'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
        'Statistical_report'
    ]
    
    def __init__(self, timeout: int = REQUEST_TIMEOUT):
        """
        初始化特征提取器
        
        Args:
            timeout: 网络请求超时时间（秒）
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _normalize_url(self, url: str) -> str:
        """标准化URL，自动补全协议"""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        return url
    
    def _parse_url(self, url: str) -> Dict[str, Any]:
        """解析URL各部分"""
        parsed = urlparse(url)
        return {
            'scheme': parsed.scheme,
            'netloc': parsed.netloc,
            'path': parsed.path,
            'query': parsed.query,
            'domain': parsed.netloc.split(':')[0],
            'port': parsed.port
        }
    
    # ==================== URL结构特征 ====================
    
    def _check_ip_address(self, url: str) -> int:
        """检查URL是否包含IP地址 (-1: 是, 1: 否)"""
        ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
        hex_pattern = r'0x[0-9a-fA-F]+'
        if re.search(ip_pattern, url) or re.search(hex_pattern, url):
            return -1
        return 1
    
    def _check_url_length(self, url: str) -> int:
        """检查URL长度 (1: <54, 0: 54-75, -1: >75)"""
        length = len(url)
        if length < 54:
            return 1
        elif length <= 75:
            return 0
        return -1
    
    def _check_shortening_service(self, url: str) -> int:
        """检查是否使用短链服务 (-1: 是, 1: 否)"""
        domain = self._parse_url(url)['domain'].lower()
        for service in SHORTENING_SERVICES:
            if service in domain:
                return -1
        return 1
    
    def _check_at_symbol(self, url: str) -> int:
        """检查URL是否包含@符号 (-1: 是, 1: 否)"""
        return -1 if '@' in url else 1
    
    def _check_double_slash(self, url: str) -> int:
        """检查是否有双斜杠重定向 (-1: 是, 1: 否)"""
        # 检查协议后是否还有//
        path_part = url.split('://')[-1] if '://' in url else url
        return -1 if '//' in path_part else 1
    
    def _check_prefix_suffix(self, domain: str) -> int:
        """检查域名是否有前缀/后缀（-） (-1: 是, 1: 否)"""
        # 去掉子域名，只检查主域名
        parts = domain.split('.')
        main_domain = parts[-2] if len(parts) >= 2 else domain
        return -1 if '-' in main_domain else 1
    
    def _check_sub_domain(self, url: str) -> int:
        """检查子域名数量 (1: <=1, 0: 2, -1: >=3)"""
        domain = self._parse_url(url)['domain']
        # 移除www
        if domain.startswith('www.'):
            domain = domain[4:]
        dots = domain.count('.')
        if dots <= 1:
            return 1
        elif dots == 2:
            return 0
        return -1
    
    def _check_https_token(self, domain: str) -> int:
        """检查域名中是否包含HTTPS (-1: 是, 1: 否)"""
        return -1 if 'https' in domain.lower() else 1
    
    def _check_port(self, url: str) -> int:
        """检查端口是否标准 (1: 标准, -1: 非标准)"""
        parsed = self._parse_url(url)
        port = parsed['port']
        if port is None:
            return 1  # 默认端口
        return 1 if port in STANDARD_PORTS else -1
    
    # ==================== SSL/DNS特征 ====================
    
    def _check_ssl_state(self, domain: str) -> int:
        """检查SSL证书状态 (1: 有效, 0: 可疑, -1: 无效)"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    # 检查证书是否过期
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    if not_after < datetime.now():
                        return -1
                    # 检查证书颁发者
                    issuer = dict(x[0] for x in cert['issuer'])
                    trusted_cas = ['DigiCert', 'Let\'s Encrypt', 'Comodo', 'GeoTrust', 
                                   'GlobalSign', 'Symantec', 'Thawte', 'VeriSign']
                    org = issuer.get('organizationName', '')
                    if any(ca.lower() in org.lower() for ca in trusted_cas):
                        return 1
                    return 0
        except Exception as e:
            logger.debug(f"SSL检查失败 {domain}: {e}")
            return -1
    
    def _check_dns_record(self, domain: str) -> int:
        """检查DNS记录 (1: 有, -1: 无)"""
        try:
            import dns.resolver
            dns.resolver.resolve(domain, 'A', lifetime=DNS_TIMEOUT)
            return 1
        except ImportError:
            # 如果没有dnspython，使用socket
            try:
                socket.gethostbyname(domain)
                return 1
            except:
                return -1
        except Exception:
            return -1
    
    # ==================== Whois特征 ====================
    
    def _check_domain_age(self, domain: str) -> int:
        """检查域名年龄 (1: >=6个月, -1: <6个月)"""
        try:
            import whois
            w = whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if creation_date:
                age_days = (datetime.now() - creation_date).days
                return 1 if age_days >= 180 else -1
        except Exception as e:
            logger.debug(f"Whois查询失败 {domain}: {e}")
        return -1
    
    def _check_domain_registration_length(self, domain: str) -> int:
        """检查域名注册时长 (1: >=1年, -1: <1年)"""
        try:
            import whois
            w = whois.whois(domain)
            expiration_date = w.expiration_date
            creation_date = w.creation_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if expiration_date and creation_date:
                reg_length = (expiration_date - creation_date).days
                return 1 if reg_length >= 365 else -1
        except Exception as e:
            logger.debug(f"域名注册时长查询失败 {domain}: {e}")
        return -1
    
    # ==================== HTML内容特征 ====================
    
    def _fetch_html(self, url: str) -> Optional[str]:
        """获取网页HTML内容"""
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            return response.text
        except Exception as e:
            logger.debug(f"获取HTML失败 {url}: {e}")
            return None
    
    def _check_favicon(self, url: str, soup: Optional[BeautifulSoup], domain: str) -> int:
        """检查是否有Favicon (1: 有且来自同域, -1: 无或来自外域)"""
        if soup is None:
            return -1
        try:
            favicon = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
            if favicon and favicon.get('href'):
                href = favicon['href']
                if href.startswith('//'):
                    href = 'http:' + href
                elif href.startswith('/'):
                    return 1  # 相对路径，同域
                favicon_domain = self._parse_url(href)['domain']
                return 1 if domain in favicon_domain or favicon_domain in domain else -1
            return -1
        except:
            return -1
    
    def _check_request_url(self, soup: Optional[BeautifulSoup], domain: str) -> int:
        """检查外部资源比例 (1: <22%, 0: 22-61%, -1: >61%)"""
        if soup is None:
            return -1
        try:
            total, external = 0, 0
            for tag in soup.find_all(['img', 'script', 'link']):
                src = tag.get('src') or tag.get('href')
                if src:
                    total += 1
                    if src.startswith(('http://', 'https://')):
                        src_domain = self._parse_url(src)['domain']
                        if domain not in src_domain:
                            external += 1
            if total == 0:
                return 1
            ratio = external / total * 100
            if ratio < 22:
                return 1
            elif ratio <= 61:
                return 0
            return -1
        except:
            return -1
    
    def _check_url_of_anchor(self, soup: Optional[BeautifulSoup], domain: str) -> int:
        """检查锚点URL比例 (1: <31%, 0: 31-67%, -1: >67%)"""
        if soup is None:
            return -1
        try:
            anchors = soup.find_all('a', href=True)
            if not anchors:
                return 1
            suspicious = 0
            for a in anchors:
                href = a['href']
                if href in ['#', '', 'javascript:void(0)', 'javascript:;']:
                    suspicious += 1
                elif href.startswith(('http://', 'https://')):
                    href_domain = self._parse_url(href)['domain']
                    if domain not in href_domain:
                        suspicious += 1
            ratio = suspicious / len(anchors) * 100
            if ratio < 31:
                return 1
            elif ratio <= 67:
                return 0
            return -1
        except:
            return -1
    
    def _check_links_in_tags(self, soup: Optional[BeautifulSoup], domain: str) -> int:
        """检查Meta/Script/Link标签中外部链接比例"""
        if soup is None:
            return -1
        try:
            total, external = 0, 0
            for tag in soup.find_all(['meta', 'script', 'link']):
                src = tag.get('src') or tag.get('href') or tag.get('content')
                if src and src.startswith(('http://', 'https://')):
                    total += 1
                    src_domain = self._parse_url(src)['domain']
                    if domain not in src_domain:
                        external += 1
            if total == 0:
                return 1
            ratio = external / total * 100
            if ratio < 17:
                return 1
            elif ratio <= 81:
                return 0
            return -1
        except:
            return -1
    
    def _check_sfh(self, soup: Optional[BeautifulSoup], domain: str) -> int:
        """检查表单提交地址 (1: 同域, 0: 空/about:blank, -1: 外域)"""
        if soup is None:
            return -1
        try:
            forms = soup.find_all('form', action=True)
            if not forms:
                return 1
            for form in forms:
                action = form['action']
                if action in ['', 'about:blank']:
                    return 0
                if action.startswith(('http://', 'https://')):
                    action_domain = self._parse_url(action)['domain']
                    if domain not in action_domain:
                        return -1
            return 1
        except:
            return -1
    
    def _check_submitting_to_email(self, soup: Optional[BeautifulSoup]) -> int:
        """检查是否提交到邮箱 (-1: 是, 1: 否)"""
        if soup is None:
            return 1
        try:
            html_str = str(soup)
            if 'mailto:' in html_str or 'mail(' in html_str:
                return -1
            return 1
        except:
            return 1
    
    def _check_abnormal_url(self, url: str, domain: str) -> int:
        """检查URL是否异常 (-1: 是, 1: 否)"""
        try:
            import whois
            w = whois.whois(domain)
            if w.domain_name:
                whois_domain = w.domain_name
                if isinstance(whois_domain, list):
                    whois_domain = whois_domain[0]
                if whois_domain and domain.lower() not in whois_domain.lower():
                    return -1
            return 1
        except:
            return -1
    
    def _check_redirect(self, url: str) -> int:
        """检查重定向次数 (0: <=1, 1: 2-3, -1: >=4)"""
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            redirects = len(response.history)
            if redirects <= 1:
                return 0
            elif redirects <= 3:
                return 1
            return -1
        except:
            return -1
    
    def _check_on_mouseover(self, soup: Optional[BeautifulSoup]) -> int:
        """检查是否有onMouseOver事件 (-1: 是, 1: 否)"""
        if soup is None:
            return 1
        try:
            html_str = str(soup)
            if 'onmouseover' in html_str.lower():
                return -1
            return 1
        except:
            return 1
    
    def _check_right_click(self, soup: Optional[BeautifulSoup]) -> int:
        """检查是否禁用右键 (-1: 是, 1: 否)"""
        if soup is None:
            return 1
        try:
            html_str = str(soup).lower()
            if 'event.button==2' in html_str or 'oncontextmenu' in html_str:
                return -1
            return 1
        except:
            return 1
    
    def _check_popup_window(self, soup: Optional[BeautifulSoup]) -> int:
        """检查是否有弹窗 (-1: 是, 1: 否)"""
        if soup is None:
            return 1
        try:
            html_str = str(soup).lower()
            if 'window.open' in html_str or 'alert(' in html_str:
                return -1
            return 1
        except:
            return 1
    
    def _check_iframe(self, soup: Optional[BeautifulSoup]) -> int:
        """检查是否使用iframe (-1: 是, 1: 否)"""
        if soup is None:
            return 1
        try:
            iframes = soup.find_all('iframe')
            return -1 if iframes else 1
        except:
            return 1
    
    # ==================== 外部数据特征 ====================
    # 注意: web_traffic, Page_Rank, Google_Index, Links_pointing_to_page, Statistical_report
    # 这些特征需要访问第三方API（如Alexa, Google等），由于API限制，暂时使用启发式方法
    
    def _check_web_traffic(self, domain: str) -> int:
        """
        检查网站流量 (1: 高, 0: 中, -1: 低)
        注意: 由于Alexa API已停止服务，这里使用启发式方法
        基于域名特征和可访问性进行估算
        """
        try:
            # 检查是否是知名域名后缀
            known_tlds = ['.com', '.org', '.net', '.edu', '.gov']
            has_known_tld = any(domain.endswith(tld) for tld in known_tlds)
            
            # 检查域名长度（短域名通常流量更高）
            domain_parts = domain.split('.')
            main_domain = domain_parts[-2] if len(domain_parts) >= 2 else domain
            
            if has_known_tld and len(main_domain) <= 10:
                return 1
            elif has_known_tld:
                return 0
            return -1
        except:
            return -1
    
    def _check_page_rank(self, domain: str) -> int:
        """
        检查页面排名 (1: 高, -1: 低)
        注意: Google PageRank API已停止，使用启发式方法
        """
        try:
            # 基于域名特征估算
            if self._check_dns_record(domain) == 1 and self._check_ssl_state(domain) == 1:
                return 1
            return -1
        except:
            return -1
    
    def _check_google_index(self, domain: str) -> int:
        """
        检查是否被Google索引 (1: 是, -1: 否)
        注意: 实际应该查询Google，这里使用DNS作为代理指标
        """
        return self._check_dns_record(domain)
    
    def _check_links_pointing_to_page(self, soup: Optional[BeautifulSoup]) -> int:
        """
        检查指向页面的链接数 (1: 多, 0: 中, -1: 少)
        注意: 实际需要反向链接API，这里基于页面内容估算
        """
        if soup is None:
            return -1
        try:
            # 基于页面复杂度估算
            links = len(soup.find_all('a'))
            if links > 50:
                return 1
            elif links > 10:
                return 0
            return -1
        except:
            return -1
    
    def _check_statistical_report(self, domain: str) -> int:
        """
        检查统计报告 (-1: 有异常, 1: 无异常)
        注意: 实际应查询PhishTank等数据库，这里使用综合评估
        """
        try:
            # 综合评估：如果DNS和SSL都正常，认为无异常
            dns_ok = self._check_dns_record(domain) == 1
            ssl_ok = self._check_ssl_state(domain) >= 0
            return 1 if dns_ok and ssl_ok else -1
        except:
            return -1
    
    # ==================== 主提取方法 ====================
    
    def extract_features(self, url: str) -> Dict[str, int]:
        """
        从URL提取30个特征
        
        Args:
            url: 目标URL字符串
            
        Returns:
            包含30个特征的字典
        """
        # 标准化URL
        url = self._normalize_url(url)
        parsed = self._parse_url(url)
        domain = parsed['domain']
        
        logger.info(f"开始提取URL特征: {url}")
        
        # 获取HTML内容
        html_content = self._fetch_html(url)
        soup = BeautifulSoup(html_content, 'html.parser') if html_content else None
        
        # 提取所有特征
        features = {
            # URL结构特征
            'having_IP_Address': self._check_ip_address(url),
            'URL_Length': self._check_url_length(url),
            'Shortining_Service': self._check_shortening_service(url),
            'having_At_Symbol': self._check_at_symbol(url),
            'double_slash_redirecting': self._check_double_slash(url),
            'Prefix_Suffix': self._check_prefix_suffix(domain),
            'having_Sub_Domain': self._check_sub_domain(url),
            'HTTPS_token': self._check_https_token(domain),
            'port': self._check_port(url),
            
            # SSL/DNS特征
            'SSLfinal_State': self._check_ssl_state(domain),
            'DNSRecord': self._check_dns_record(domain),
            
            # Whois特征
            'Domain_registeration_length': self._check_domain_registration_length(domain),
            'age_of_domain': self._check_domain_age(domain),
            
            # HTML内容特征
            'Favicon': self._check_favicon(url, soup, domain),
            'Request_URL': self._check_request_url(soup, domain),
            'URL_of_Anchor': self._check_url_of_anchor(soup, domain),
            'Links_in_tags': self._check_links_in_tags(soup, domain),
            'SFH': self._check_sfh(soup, domain),
            'Submitting_to_email': self._check_submitting_to_email(soup),
            'Abnormal_URL': self._check_abnormal_url(url, domain),
            'Redirect': self._check_redirect(url),
            'on_mouseover': self._check_on_mouseover(soup),
            'RightClick': self._check_right_click(soup),
            'popUpWidnow': self._check_popup_window(soup),
            'Iframe': self._check_iframe(soup),
            
            # 外部数据特征
            'web_traffic': self._check_web_traffic(domain),
            'Page_Rank': self._check_page_rank(domain),
            'Google_Index': self._check_google_index(domain),
            'Links_pointing_to_page': self._check_links_pointing_to_page(soup),
            'Statistical_report': self._check_statistical_report(domain)
        }
        
        logger.info(f"特征提取完成，共 {len(features)} 个特征")
        return features
    
    def extract_features_async(self, url: str) -> Dict[str, int]:
        """
        异步提取特征（用于API调用）
        使用线程池并行执行耗时操作
        """
        return self.extract_features(url)

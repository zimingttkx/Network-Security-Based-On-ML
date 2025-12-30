# 网络安全威胁检测系统

基于机器学习的网络安全威胁检测项目 - 适合初学者学习的完整机器学习项目示例

## 项目简介

这是一个**麻雀虽小，五脏俱全**的机器学习项目，专注于网络安全威胁检测。项目包含了一个完整的机器学习工作流程：数据处理、模型训练、预测服务和Web界面。

**适合人群：** 想要学习如何构建完整机器学习项目的初学者

## 主要功能

- ✅ **威胁预测**：输入网络流量特征，预测是否存在安全威胁
- ✅ **模型训练**：使用自己的数据训练模型
- ✅ **Web界面**：简单易用的网页操作界面
- ✅ **API接口**：提供RESTful API供其他程序调用

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.12 | 编程语言 |
| FastAPI | Web框架 |
| Scikit-learn | 机器学习库 |
| XGBoost | 梯度提升算法 |
| Pandas | 数据处理 |
| MongoDB | 数据存储（可选） |

## 快速开始

### 1. 安装Python环境

确保你的电脑已安装 Python 3.12 或更高版本。

```bash
python --version  # 检查Python版本
```

### 2. 克隆项目

```bash
git clone https://github.com/zimingttkx/Network-Security-Based-On-ML.git
cd Network-Security-Based-On-ML
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动服务

```bash
python app.py
```

### 5. 访问应用

打开浏览器访问：http://localhost:8000

- **首页**：http://localhost:8000
- **威胁预测**：http://localhost:8000/predict
- **模型训练**：http://localhost:8000/train
- **API文档**：http://localhost:8000/docs

## 使用教程

### 威胁预测

1. 访问 http://localhost:8000/predict
2. 输入30个网络流量特征（或点击"填充示例数据"）
3. 点击"预测"按钮
4. 查看预测结果：安全 或 威胁

### 模型训练

#### 使用默认数据训练

1. 访问 http://localhost:8000/train
2. 选择"使用项目默认数据"
3. 点击"开始训练"
4. 等待训练完成（约5-10分钟）

#### 使用自定义数据训练

1. 准备CSV格式的数据文件（需包含30个特征）
2. 访问 http://localhost:8000/train
3. 选择"上传自定义数据"
4. 上传CSV文件
5. 点击"验证数据"检查数据质量
6. 如有缺失特征，选择补全策略
7. 使用补全后的数据进行训练

### API调用示例

```python
import requests

# 预测威胁
data = {
    "having_IP_Address": 1,
    "URL_Length": 1,
    "Shortining_Service": 1,
    # ... 其他27个特征
}

response = requests.post('http://localhost:8000/predict_live', json=data)
result = response.json()
print(f"预测结果: {result['prediction']}")
```

## 项目结构

```
Network-Security-Based-On-ML/
├── networksecurity/          # 核心代码
│   ├── components/          # 数据处理和模型训练组件
│   ├── pipeline/            # 训练和预测流程
│   ├── config/              # 配置管理
│   └── logging/             # 日志记录
├── templates/               # HTML模板
├── static/                  # 静态资源（CSS、JS）
├── data/                    # 训练数据
├── models/                  # 训练好的模型
├── artifacts/               # 训练产物
├── docs/                    # 项目文档
├── tests/                   # 测试文件
├── app.py                   # 主程序入口
├── requirements.txt         # 依赖包列表
└── README.md                # 项目说明
```

## 数据特征说明

模型需要30个网络流量特征，每个特征的值通常为 -1、0 或 1：

<details>
<summary>点击查看完整特征列表</summary>

1. having_IP_Address - URL中是否包含IP地址
2. URL_Length - URL长度
3. Shortining_Service - 是否使用短链服务
4. having_At_Symbol - URL中是否包含@符号
5. double_slash_redirecting - 是否有双斜杠重定向
6. Prefix_Suffix - 前缀后缀
7. having_Sub_Domain - 是否有子域名
8. SSLfinal_State - SSL状态
9. Domain_registeration_length - 域名注册时长
10. Favicon - 网站图标
11. port - 端口
12. HTTPS_token - HTTPS令牌
13. Request_URL - 请求URL
14. URL_of_Anchor - 锚点URL
15. Links_in_tags - 标签中的链接
16. SFH - 服务器表单处理
17. Submitting_to_email - 提交到邮箱
18. Abnormal_URL - 异常URL
19. Redirect - 重定向
20. on_mouseover - 鼠标悬停事件
21. RightClick - 右键点击
22. popUpWidnow - 弹窗
23. Iframe - 内嵌框架
24. age_of_domain - 域名年龄
25. DNSRecord - DNS记录
26. web_traffic - 网络流量
27. Page_Rank - 页面排名
28. Google_Index - 谷歌索引
29. Links_pointing_to_page - 指向页面的链接
30. Statistical_report - 统计报告

</details>

## 常见问题

### 如何更换训练数据？

将你的CSV文件放到 `data/` 目录下，或通过Web界面上传。

### 训练需要多长时间？

使用默认数据约5-10分钟，具体取决于你的电脑性能。

### 是否需要MongoDB？

不是必须的。如果不配置MongoDB，项目会使用本地CSV文件作为数据源。

### 如何提高模型准确率？

1. 使用更多高质量的训练数据
2. 调整模型参数（在 `config/config.yaml` 中）
3. 尝试不同的机器学习算法

## 学习建议

如果你是初学者，建议按以下顺序学习项目：

1. **先运行起来**：按照"快速开始"步骤启动项目
2. **体验功能**：使用Web界面进行预测和训练
3. **阅读代码**：从 `app.py` 开始，了解项目入口
4. **理解流程**：查看 `networksecurity/pipeline/` 了解训练流程
5. **修改尝试**：尝试修改参数，观察效果变化

## 技术支持

- 📧 Email: 2147514473@qq.com
- 🐛 Issues: [GitHub Issues](https://github.com/zimingttkx/Network-Security-Based-On-ML/issues)

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**作者：** 梓铭

**提示：** 这是一个学习项目，适合了解机器学习项目的完整流程。如果对你有帮助，请给个Star！⭐

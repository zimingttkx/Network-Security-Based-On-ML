# 安全政策

## 支持的版本

目前正在接收安全更新的项目版本：

| 版本 | 支持状态 |
| ------- | ------------------ |
| 3.0.x   | :white_check_mark: |
| 2.0.x   | :white_check_mark: |
| 1.0.x   | :x: |
| < 1.0   | :x: |

## 报告漏洞

我们非常重视安全问题。如果您发现了安全漏洞，请**不要**公开提交 Issue。

### 报告流程

1. **通过私密渠道报告**
   - 发送邮件至：security@example.com
   - 或创建私密的 Security Advisory

2. **包含以下信息**
   - 漏洞类型（如 SQL 注入、XSS、权限提升等）
   - 受影响的文件或组件
   - 漏洞位置（文件路径、行号）
   - 复现步骤
   - 潜在的影响
   - 建议的修复方案（如果有）

3. **响应时间**
   - 我们会在 48 小时内确认收到您的报告
   - 在 7 天内提供初步评估
   - 在 30 天内发布修复补丁（视严重程度而定）

### 安全漏洞示例

以下是我们特别关注的安全问题类型：

#### 高危漏洞
- SQL 注入
- 远程代码执行（RCE）
- 身份认证绕过
- 权限提升
- 敏感数据泄露

#### 中危漏洞
- 跨站脚本（XSS）
- 跨站请求伪造（CSRF）
- 不安全的反序列化
- 路径遍历
- 不当的输入验证

#### 低危漏洞
- 信息泄露
- 拒绝服务（DoS）
- 不安全的配置
- 缺少安全响应头

## 安全最佳实践

### 对于用户

1. **保持软件更新**
   ```bash
   pip install --upgrade networksecurity
   ```

2. **使用强密码和密钥**
   - MongoDB 连接使用强密码
   - API keys 保持机密
   - 不要在代码中硬编码凭证

3. **配置环境变量**
   ```bash
   # .env 文件示例（不要提交到 Git）
   MONGO_DB_URL=mongodb://username:strong_password@localhost:27017/
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

4. **限制网络访问**
   ```bash
   # 仅在本地运行
   uvicorn app:app --host 127.0.0.1 --port 8000

   # 生产环境使用反向代理
   # 参见 nginx 配置示例
   ```

5. **定期备份数据**
   ```bash
   # 备份 MongoDB 数据
   mongodump --uri="mongodb://localhost:27017/networksecurity" --out=/backup/
   ```

### 对于开发者

1. **输入验证**
   ```python
   from pydantic import BaseModel, validator

   class PredictionInput(BaseModel):
       features: list

       @validator('features')
       def validate_features(cls, v):
           if len(v) != 30:
               raise ValueError('必须包含30个特征')
           return v
   ```

2. **SQL 注入防护**
   ```python
   # 使用参数化查询
   from pymongo import MongoClient

   # 好的做法
   db.collection.find({"_id": ObjectId(user_id)})

   # 避免字符串拼接
   # db.collection.find(f"{{_id: '{user_id}'}}")  # 危险！
   ```

3. **敏感数据处理**
   ```python
   # 不要记录敏感信息
   logger.info(f"User login: {username}")  # OK
   # logger.info(f"Password: {password}")  # 危险！

   # 使用环境变量
   import os
   api_key = os.getenv('API_KEY')  # 好
   # api_key = "hardcoded_key"  # 危险！
   ```

4. **依赖项安全**
   ```bash
   # 检查已知漏洞
   pip install safety
   safety check

   # 更新依赖
   pip list --outdated
   pip install --upgrade package_name
   ```

5. **代码审查清单**
   - [ ] 所有用户输入都经过验证
   - [ ] 没有硬编码的凭证
   - [ ] 敏感操作需要身份验证
   - [ ] 错误消息不泄露敏感信息
   - [ ] 使用 HTTPS 进行通信
   - [ ] 实施了速率限制
   - [ ] 日志不包含敏感数据

## 已知安全考虑

### 数据隐私
- 上传的训练数据仅在服务器临时存储
- 不会永久保存用户的预测数据
- 建议使用私有部署处理敏感数据

### API 安全
- 生产环境应启用 API 密钥认证
- 建议使用 HTTPS
- 实施速率限制防止滥用

### 依赖项
- 定期更新依赖包以修复已知漏洞
- 使用 `safety` 工具检查依赖安全性

## 安全更新通知

- GitHub Security Advisories: [订阅通知](../../security/advisories)
- Release Notes: 每次发布都包含安全修复说明

## 致谢

我们感谢以下安全研究人员的负责任披露：

<!-- 在此列出报告安全问题的研究人员 -->
- 暂无

---

**最后更新**: 2025-11-23

如有任何安全问题，请联系：security@example.com

# 贡献指南

感谢你对本项目的关注！欢迎提交贡献。

## 如何贡献

### 报告Bug
1. 在Issues中搜索是否已有相同问题
2. 如果没有，创建新Issue并使用Bug报告模板
3. 提供详细的复现步骤和环境信息

### 提出新功能
1. 在Issues中创建功能请求
2. 描述功能的使用场景和价值
3. 等待讨论和反馈

### 提交代码
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 代码规范

### Python代码
- 遵循PEP 8规范
- 使用有意义的变量名
- 添加必要的注释
- 保持函数简洁

### 提交信息
- 使用清晰的提交信息
- 格式：`类型: 简短描述`
- 类型：feat, fix, docs, style, refactor, test, chore

示例：
```
feat: 添加数据验证功能
fix: 修复模型训练时的内存泄漏
docs: 更新README安装说明
```

## 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/zimingttkx/Network-Security-Based-On-ML.git
cd Network-Security-Based-On-ML

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 运行测试
python test_functionality.py
```

## 测试

在提交PR前，请确保：
- 所有现有测试通过
- 新功能有相应的测试
- 代码无明显错误

## 问题反馈

如有任何问题，欢迎通过以下方式联系：
- 创建Issue
- 发送邮件至：2147514473@qq.com

再次感谢你的贡献！

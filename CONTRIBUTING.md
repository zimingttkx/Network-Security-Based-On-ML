# 贡献指南

感谢您对网络安全威胁检测系统的关注！我们欢迎任何形式的贡献。

## 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议，请：

1. 先检查 [Issues](../../issues) 中是否已有相关问题
2. 如果没有，创建新的 Issue，并提供：
   - 清晰的标题和描述
   - 复现步骤（对于 bug）
   - 期望的行为
   - 实际的行为
   - 系统环境信息（Python 版本、操作系统等）
   - 相关的日志或截图

### 提交代码

1. **Fork 项目**
   ```bash
   git clone https://github.com/your-username/PythonProject4.git
   cd PythonProject4
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **进行开发**
   - 遵循现有的代码风格
   - 添加适当的注释
   - 确保代码通过所有测试
   - 为新功能编写测试

4. **运行测试**
   ```bash
   # 运行所有测试
   python -m pytest tests/ -v

   # 检查代码覆盖率
   python -m pytest tests/ --cov=networksecurity --cov-report=html
   ```

5. **提交更改**
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   # 或
   git commit -m "fix: 修复问题描述"
   ```

   **提交信息规范**：
   - `feat`: 新功能
   - `fix`: 修复 bug
   - `docs`: 文档更新
   - `style`: 代码格式调整（不影响功能）
   - `refactor`: 代码重构
   - `test`: 测试相关
   - `chore`: 构建过程或辅助工具的变动

6. **推送到 GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建 Pull Request**
   - 访问原项目的 GitHub 页面
   - 点击 "New Pull Request"
   - 选择您的分支
   - 填写 PR 描述，说明您的更改
   - 等待代码审查

## 代码规范

### Python 代码风格

遵循 PEP 8 规范：

```python
# 好的示例
def validate_features(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    验证数据特征完整性

    Args:
        df: 输入数据框

    Returns:
        Tuple[bool, Dict]: 验证结果和报告
    """
    missing_features = [f for f in self.REQUIRED_FEATURES if f not in df.columns]
    return len(missing_features) == 0, {"missing_features": missing_features}
```

### 文档字符串

所有公共函数和类都应该有文档字符串：

```python
def impute_missing_features(
    self,
    df: pd.DataFrame,
    strategy: str = 'constant',
    fill_value: int = 0
) -> Tuple[pd.DataFrame, Dict]:
    """
    补全缺失的特征

    Args:
        df: 输入数据框
        strategy: 补全策略 ('mean', 'median', 'most_frequent', 'constant', 'knn')
        fill_value: 常数填充值（仅当 strategy='constant' 时使用）

    Returns:
        Tuple[pd.DataFrame, Dict]: 补全后的数据框和补全报告

    Raises:
        ValueError: 当策略不支持时
    """
```

### 测试要求

- 所有新功能必须有相应的测试
- 测试覆盖率应保持在 80% 以上
- 使用有意义的测试名称

```python
def test_validate_complete_data(self, validator, complete_data):
    """测试验证完整数据"""
    is_valid, report = validator.validate_features(complete_data)
    assert is_valid is True
    assert len(report['missing_features']) == 0
```

## 开发环境设置

1. **克隆项目**
   ```bash
   git clone https://github.com/username/PythonProject4.git
   cd PythonProject4
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # 开发依赖
   ```

4. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，填入必要的配置
   ```

5. **运行开发服务器**
   ```bash
   python test_app.py
   ```

## 审查流程

1. 提交 PR 后，维护者会进行代码审查
2. 可能会要求修改或补充
3. 审查通过后，代码会被合并到主分支
4. 您的贡献会被记录在 Contributors 列表中

## 问题讨论

对于复杂的功能或重大更改，建议先创建 Issue 进行讨论：

1. 描述您的想法
2. 等待社区反馈
3. 达成共识后再开始开发

## 行为准则

请阅读我们的 [行为准则](CODE_OF_CONDUCT.md)，并在参与项目时遵守。

## 联系方式

- GitHub Issues: [项目 Issues 页面](../../issues)
- Email: your-email@example.com

## 许可证

通过贡献代码，您同意您的贡献将按照 [MIT License](LICENSE) 进行许可。

---

再次感谢您的贡献！🎉

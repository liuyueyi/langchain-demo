# LangChain 实战项目

这是一个专注于 LangChain 框架实际应用的实战项目，通过具体案例演示如何构建智能对话系统。项目涵盖了从基础配置到高级功能的完整开发流程，适合想要深入学习 LangChain 的开发者参考和实践。

## 项目结构

```[config](.git%2Fconfig)
langchain-demo/
├── basic00/
│   └── SimpleChat.py      # 主要的聊天示例代码
├── config.env             # 配置文件（API密钥等）
├── pyproject.toml         # 项目配置文件
└── README.md              # [config](.git%2Fconfig)项目说明文档
```

## 环境要求

- Python >= 3.10
- uv 包管理器

## 安装依赖

```bash
uv sync
```

## 配置说明

将 `config.env_demo` 拷贝为 `config.env`，并填入你的 API 密钥。

在 `config.env` 文件中配置以下信息：

```env
# OpenAI API密钥
API_KEY=your_api_key_here

# API基础URL
BASE_URL=https://api.siliconflow.cn/v1

# 模型名称
MODEL=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

## 运行示例

```bash
uv run python basic00/HelloLangChain.py
```

## 学习目标

通过本项目你将掌握：

- 🔧 **LangChain 基础配置**：环境变量管理、模型集成
- 💬 **智能对话实现**：系统提示词设计、对话流程控制
- 🌊 **流式响应处理**：实时输出、用户体验优化
- ⚙️ **项目工程化**：现代包管理、配置分离、代码组织

## 核心功能

- 🤖 基于 DeepSeek 模型的智能对话系统
- 🧠 完整的 Agents 功能实现（工具调用、记忆、流式等）
- 🌊 实时流式输出，提升交互体验
- ⚙️ 安全的环境变量配置管理
- 🎨 美化的终端输出格式
- 📦 使用 uv 进行现代化包管理

## 项目特色

### 🎯 实战导向
- 从零开始构建完整的对话应用
- 涵盖开发、测试、部署全流程
- 代码注释详细，便于理解和学习

### 🛠️ 工程化实践
- 使用 uv 进行依赖管理
- 配置文件与代码分离
- 标准化的项目结构

### 📚 学习价值
- LangChain 核心概念的实际应用
- 现代 Python 开发最佳实践
- AI 应用开发的完整流程演示

## 开发工具

项目预配置了以下开发工具：
- **pytest** - 单元测试框架
- **black** - 代码格式化工具
- **flake8** - 代码质量检查工具

## 学习建议

1. **循序渐进**：按照文件顺序理解每个组件的作用
2. **动手实践**：修改配置参数观察不同效果
3. **扩展思考**：尝试集成其他模型或添加新功能
4. **关注更新**：LangChain 框架持续演进，建议关注官方文档

## 贡献指南

欢迎提出 Issue 和 Pull Request 来改进这个项目！

## 许可证

MIT License
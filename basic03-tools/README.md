# LangChain Tools 学习实战示例

本目录包含 LangChain Tools 的完整学习路径，从基础使用到高级特性，每个文件代表不同的学习阶段。

## 文件说明

### 1. BasicToolDefinition.py
- **主题**: 基础工具定义
- **内容**:
  - 使用 `@tool` 装饰器定义工具
  - 经典方式创建工具 (`Tool` 类)
  - 工具注册和基本信息查看
  - 简单参数工具的实现

### 2. AdvancedSchemaTools.py
- **主题**: 高级 Schema 定义
- **内容**:
  - 使用 Pydantic 模型定义复杂参数
  - 参数验证和约束设置
  - `StructuredTool` 手动创建
  - Schema 验证错误处理

### 3. ContextAccessTools.py
- **主题**: 上下文访问
- **内容**:
  - 自定义 `BaseTool` 子类
  - 访问会话和运行时上下文
  - 全局状态管理和模拟
  - 上下文感知工具实现

### 4. ComprehensiveToolsDemo.py
- **主题**: 综合实战示例
- **内容**:
  - 整合前三部分概念
  - 实际应用场景（任务管理）
  - 错误处理和验证
  - 特性对比总结

## 学习路径

按以下顺序学习以获得最佳效果：

1. **基础概念** → `BasicToolDefinition.py`
   - 了解如何定义和注册工具
   - 学习工具的基本结构

2. **高级特性** → `AdvancedSchemaTools.py`
   - 掌握复杂参数定义
   - 学习参数验证机制

3. **上下文操作** → `ContextAccessTools.py`
   - 理解上下文访问模式
   - 学习状态管理

4. **综合应用** → `ComprehensiveToolsDemo.py`
   - 实际项目应用示例
   - 最佳实践参考

## 运行示例

每个文件都可以独立运行：

```bash
cd basic03-tools
python BasicToolDefinition.py
python AdvancedSchemaTools.py
python ContextAccessTools.py
python ComprehensiveToolsDemo.py
```

## 核心概念

### 工具定义方式
- **装饰器方式**: 使用 `@tool` 装饰器快速定义
- **类方式**: 继承 `BaseTool` 或使用 `StructuredTool`
- **函数方式**: 通过 `Tool` 类包装普通函数

### Schema 验证
- 使用 Pydantic `BaseModel` 定义参数结构
- 支持类型验证、范围限制、枚举值等
- 自动生成工具描述和参数文档

### 上下文访问
- 通过 `BaseTool` 子类访问运行时上下文
- 支持会话状态、历史记录等上下文信息
- 可与外部系统集成

## 实践要点

1. **工具命名**: 使用有意义的名称，便于理解和调试
2. **参数验证**: 使用 Schema 验证确保输入质量
3. **错误处理**: 适当处理异常情况并提供清晰反馈
4. **文档说明**: 为工具提供清晰的描述和参数说明
5. **性能考虑**: 避免在工具中执行过于耗时的操作

这些示例旨在帮助您掌握 LangChain Tools 的核心概念和实用技巧，从基础到高级逐步深入。
# LangChain实战开发教程（十一）：工具Schema定义与参数验证进阶

> **解决工具参数验证的3种技术方案对比**：掌握Pydantic Schema和JSON Schema定义工具输入参数的高级技巧

## 🎯 本文目标

深入解析LangChain工具Schema定义的高级用法，掌握基于Pydantic模型的参数验证机制，学会通过结构化Schema确保工具调用的准确性和可靠性。

### 📚 核心知识点概览

通过本文你将掌握：
- **Pydantic Schema定义**：使用BaseModel定义复杂参数结构
- **参数验证机制**：字段约束、类型检查、数据验证
- **JSON Schema方式**：直接使用JSON Schema定义工具参数
- **保留参数处理**：config和runtime参数的正确处理方式
- **错误处理策略**：参数验证失败时的优雅处理

### 🎯 使用场景判断

✅ **推荐使用Schema定义**：
- 需要复杂参数结构验证的工具
- 参数类型和格式要求严格的场景
- 需要提供详细参数说明给AI的工具
- 企业级应用中对数据安全要求高的场景
- 多参数、嵌套参数的复杂工具

❌ **不建议使用**：
- 简单的单参数工具
- 参数验证要求不严格的场景
- 快速原型开发阶段
- 参数结构经常变动的工具

## 💡 Schema核心使用

**Schema定义**：通过结构化的方式定义工具参数的类型、约束和验证规则，确保AI在调用工具时提供符合要求的参数格式。

### 🔧 前置知识点

**前置条件**：
- 理解Pydantic BaseModel的基本用法
- 掌握字段验证和约束定义
- 了解type hints和Optional类型
- 理解Pydantic与type hints的结合使用
- 掌握常用Field约束参数（min_length, max_length, ge, le等）
- 了解嵌套模型和复杂数据结构定义

**Pydantic和Type Hints核心知识点**：

Pydantic是一个用于数据解析和验证的Python库，它使用类型提示进行数据验证和设置管理。

```python
# Pydantic BaseModel基础示例
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class UserInput(BaseModel):
    # 基础字段定义
    name: str = Field(description="用户姓名")
    age: int = Field(description="用户年龄", ge=0, le=150)  # ge: greater or equal, le: less or equal
    email: Optional[str] = Field(default=None, description="邮箱地址")
    
    # 列表字段定义
    tags: List[str] = Field(description="用户标签", min_items=1, max_items=10)
    
    # 字典字段定义
    metadata: Dict[str, Any] = Field(description="元数据", default_factory=dict)
    
    # 使用正则表达式验证
    phone: str = Field(description="手机号码", pattern=r"^1[3-9]\d{9}$")
```

**常用Field约束参数**：
- `description`：字段描述，用于生成文档和提示
- `default`：默认值
- `default_factory`：默认值工厂函数
- `ge`：大于等于（greater or equal）
- `gt`：大于（greater than）
- `le`：小于等于（less or equal）
- `lt`：小于（less than）
- `min_length`：最小长度
- `max_length`：最大长度
- `pattern`：正则表达式验证
- `min_items`：列表最小项目数
- `max_items`：列表最大项目数
- `unique_items`：列表项目唯一性

Type Hints（类型提示/注解）是自 Python 3.5 版本引入的一种语法特性（PEP 484），允许开发者在代码中显式标注变量、函数参数和返回值的预期数据类型。这是一种非强制性的类型声明，旨在通过提高代码可读性、辅助静态分析工具检查错误以及增强 IDE 的自动补全功能，从而提升大型项目的开发效率与维护性，不影响代码实际执行速度

**Type Hints常用类型**：
- `str`, `int`, `float`, `bool`：基础类型
- `List[type]`：列表类型
- `Dict[key_type, value_type]`：字典类型
- `Optional[type]`：可选类型（可以为None）
- `Union[type1, type2]`：联合类型
- `Any`：任意类型
- `Tuple[type1, type2]`：元组类型

**嵌套模型定义**：

```python
from pydantic import BaseModel, Field
from typing import Optional

class Address(BaseModel):
    street: str = Field(description="街道地址")
    city: str = Field(description="城市")
    zip_code: str = Field(description="邮编", pattern=r"^\d{6}$")

class UserProfile(BaseModel):
    name: str = Field(description="用户姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    address: Address = Field(description="地址信息")  # 嵌套模型
    emergency_contact: Optional[Address] = Field(default=None, description="紧急联系人地址")
```

这些知识点是定义复杂工具Schema的基础，能够确保参数验证的准确性和可靠性。

### 步骤1：Pydantic Schema基础定义 ⚙️

```python
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 环境配置
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.env')
load_dotenv(config_path)

os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model_name = os.getenv('MODEL')

def init_model(model=model_name):
    """初始化LLM模型"""
    return init_chat_model(
        model=model,
        model_provider="openai",
        temperature=0.7,
        timeout=30,
        max_tokens=1000,
        max_retries=3,
    )

# 基础Pydantic Schema定义
class UserSearchInput(BaseModel):
    """用户搜索的输入参数"""
    username: str = Field(description="用户名", min_length=1, max_length=50)
    department: Optional[str] = Field(default=None, description="部门名称")
    active_only: bool = Field(default=True, description="是否只返回活跃用户")

class DatabaseQueryInput(BaseModel):
    """数据库查询的输入参数"""
    table_name: str = Field(description="表名", min_length=1)
    columns: List[str] = Field(description="要查询的列名列表", default_factory=list)
    conditions: Dict[str, Any] = Field(description="查询条件", default_factory=dict)
    limit: Optional[int] = Field(default=100, description="查询结果数量限制", ge=1, le=1000)

def pretty_print_schema_info(tool_obj):
    """美化的Schema信息输出"""
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"🔧 工具名称: {tool_obj.name}")
    print(f"📝 工具描述: {tool_obj.description}")
    print(f"📋 工具Schema:")
    if hasattr(tool_obj, 'args_schema'):
        print(f"   Schema: {tool_obj.args_schema.model_json_schema()}")
    elif hasattr(tool_obj, 'args'):
        print(f"   Args: {tool_obj.args}")
    print(separator)
```

---

除了上面推荐的 Pydantic 的Schema定义方式之外，同样支持 JSON Schema，如下

```python
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}

@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

### 步骤2：Schema工具定义实现 🚀

通过注解 `@tool` 中的 `args_schema` 来表明工具的input schema定义，谁用方式比较简单

```python
# 使用Pydantic Schema的工具定义
@tool(args_schema=UserSearchInput)
def search_user(username: str, department: Optional[str] = None, active_only: bool = True) -> Dict[str, Any]:
    """
    根据用户名搜索用户信息的工具
    
    Args:
        username: 用户名
        department: 部门名称（可选）
        active_only: 是否只返回活跃用户（默认为True）
        
    Returns:
        Dict: 包含用户信息的字典
    """
    print(f"搜索用户: {username}, 部门: {department}, 仅活跃用户: {active_only}")
    
    # 模拟用户数据库
    mock_users = {
        "张三": {"id": 1, "name": "张三", "department": "技术部", "active": True},
        "李四": {"id": 2, "name": "李四", "department": "销售部", "active": False},
        "王五": {"id": 3, "name": "王五", "department": "技术部", "active": True},
        "赵六": {"id": 4, "name": "赵六", "department": "人事部", "active": True},
    }
    
    user = mock_users.get(username)
    if user:
        # 应用过滤条件
        if department and user["department"] != department:
            return {"error": f"用户 {username} 不在 {department} 部门"}
        if active_only and not user["active"]:
            return {"error": f"用户 {username} 不活跃"}
        return {"user_found": True, "user_info": user}
    else:
        return {"user_found": False, "message": f"未找到用户 {username}"}

@tool(args_schema=DatabaseQueryInput)
def query_database(table_name: str, columns: List[str], conditions: Dict[str, Any], limit: Optional[int] = 100) -> Dict[str, Any]:
    """
    查询数据库的工具
    
    Args:
        table_name: 表名
        columns: 要查询的列名列表
        conditions: 查询条件
        limit: 查询结果数量限制
        
    Returns:
        Dict: 查询结果
    """
    print(f"查询表: {table_name}")
    print(f"列: {columns}")
    print(f"条件: {conditions}")
    print(f"限制: {limit}")
    
    # 模拟数据库查询
    mock_data = {
        "employees": [
            {"id": 1, "name": "张三", "department": "技术部", "salary": 15000},
            {"id": 2, "name": "李四", "department": "销售部", "salary": 12000},
            {"id": 3, "name": "王五", "department": "技术部", "salary": 18000},
        ],
        "departments": [
            {"id": 1, "name": "技术部", "head": "张主任"},
            {"id": 2, "name": "销售部", "head": "李主任"},
            {"id": 3, "name": "人事部", "head": "王主任"},
        ]
    }
    
    table_data = mock_data.get(table_name, [])
    
    # 应用筛选条件
    filtered_data = []
    for row in table_data:
        match = True
        for key, value in conditions.items():
            if key not in row or row[key] != value:
                match = False
                break
        if match:
            filtered_data.append(row)
    
    # 应用列筛选
    if columns:
        filtered_data = [{k: v for k, v in row.items() if k in columns} for row in filtered_data]
    
    # 应用数量限制
    if limit:
        filtered_data = filtered_data[:limit]
    
    return {
        "table": table_name,
        "total_rows": len(filtered_data),
        "data": filtered_data
    }
```

### 步骤3：高级Schema工具与验证 ⚡

上一篇介绍了除了`@tool`的工具声明方式之外，还可以手动通过`StructuredTool`来创建，此时在传参中通过 `args_schema` 来指定即可

```python
def create_structured_tool_with_schema():
    """创建带复杂Schema的结构化工具"""
    def advanced_calculator(numbers: List[float], operation: str) -> Dict[str, Any]:
        """高级计算器，支持对数字列表执行操作"""
        print(f"对数字列表 {numbers} 执行 {operation} 操作")
        
        if operation == "sum":
            result = sum(numbers)
        elif operation == "average":
            result = sum(numbers) / len(numbers) if numbers else 0
        elif operation == "max":
            result = max(numbers) if numbers else 0
        elif operation == "min":
            result = min(numbers) if numbers else 0
        else:
            raise ValueError(f"不支持的操作: {operation}")
        
        return {
            "operation": operation,
            "numbers": numbers,
            "result": result,
            "count": len(numbers)
        }
    
    # 定义输入Schema
    class CalculatorInput(BaseModel):
        numbers: List[float] = Field(description="要计算的数字列表", min_items=1)
        operation: str = Field(
            description="计算操作 (sum, average, max, min)",
            json_schema_extra={"enum": ["sum", "average", "max", "min"]}
        )
    
    structured_tool = StructuredTool(
        name="AdvancedCalculator",
        description="高级计算器，支持对数字列表执行多种数学操作",
        func=advanced_calculator,
        args_schema=CalculatorInput
    )
    
    return structured_tool
```


**重要提醒：保留参数处理** 

⚠️ 重要注意事项：

在定义工具时，有两个保留参数不应该作为tool的参数出现：

1. config参数：这是LangChain内部使用的配置参数
2. runtime参数：这是运行时环境参数

正确的做法：
- 不要在工具函数签名中包含这些参数
- 如果需要配置，通过其他方式传递
- 让LangChain框架自动处理这些保留参数

错误示例：
```python
# ❌ 错误 - 不要这样做
@tool
def my_tool(param1: str, config: dict, runtime: dict) -> str:
    pass
```

正确示例：
```python
# ✅ 正确 - 只定义业务参数
@tool
def my_tool(param1: str) -> str:
    pass
```

### 步骤4：使用示例

使用方式和上一篇教程的并没有太大的差别，下面是前面几个工具的简单使用示例

```python
def advanced_schema_demo():
    """高级Schema工具演示"""
    print("🚀 开始 LangChain Tools 高级Schema示例演示")

    model = init_model()
    
    # 1. 使用Pydantic Schema的用户搜索工具
    print("\n1️⃣ 用户搜索工具 (带Schema):")
    pretty_print_schema_info(search_user)
    
    # 使用模型触发工具调用
    tools = [search_user, query_database]
    model_with_tools = model.bind_tools(tools)
    
    user_request = "请帮我查找技术部的张三用户信息，只返回活跃用户"
    print(f"   用户请求: {user_request}")
    
    response = model_with_tools.invoke([HumanMessage(content=user_request)])
    
    if response.tool_calls:
        print(f"   模型决定调用工具: {response.tool_calls[0]['name']}")
        print(f"   工具参数: {response.tool_calls[0]['args']}")
        
        # 执行工具调用
        for tool_call in response.tool_calls:
            if tool_call['name'] == search_user.name:
                result = search_user.invoke(tool_call['args'])
                print(f"   工具调用结果: {result}")
    else:
        print("   模型决定不需要调用工具")
    
    # 2. Schema验证示例
    print("\n2️⃣ Schema验证示例:")
    validation_request = "请帮我查找空用户名的用户信息"
    
    response2 = model_with_tools.invoke([HumanMessage(content=validation_request)])
    
    if response2.tool_calls:
        print(f"   模型决定调用工具: {response2.tool_calls[0]['name']}")
        print(f"   工具参数: {response2.tool_calls[0]['args']}")
        
        # 尝试执行工具调用，可能会因验证失败而抛出异常
        try:
            for tool_call in response2.tool_calls:
                if tool_call['name'] == search_user.name:
                    result = search_user.invoke(tool_call['args'])
                    print(f"   工具调用结果: {result}")
        except Exception as e:
            print(f"   ✅ 正确捕获验证错误: {e}")
    else:
        print("   模型决定不需要调用工具")
```

## ❓ 常见问题解答

**Q1**: Pydantic Schema和直接使用type hints有什么区别？
**A1**: 

| 方式 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| Pydantic Schema | 强大的验证功能、详细约束、JSON Schema输出 | 代码稍复杂 | 复杂参数验证 |
| Type Hints | 简单直观、IDE支持好 | 验证能力有限 | 简单参数验证 |

```python
# Type Hints方式（简单）
@tool
def simple_tool(name: str, age: int) -> str:
    """简单工具"""
    return f"Hello {name}, you are {age} years old"

# Pydantic Schema方式（复杂验证）
class ComplexInput(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    age: int = Field(ge=0, le=150)
    email: str = Field(pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

@tool(args_schema=ComplexInput)
def complex_tool(name: str, age: int, email: str) -> str:
    """复杂验证工具"""
    return f"Validated user: {name}, {age}, {email}"
```

**Q2**: 如何处理config和runtime保留参数？
**A2**: 这两个参数是LangChain框架内部使用的保留参数，不应该在工具定义中出现：

```python
# ❌ 错误做法 - 不要包含保留参数
@tool
def wrong_tool(param1: str, config: dict, runtime: dict) -> str:
    """错误的工具定义"""
    pass

# ✅ 正确做法 - 只定义业务参数
@tool
def correct_tool(param1: str, param2: Optional[str] = None) -> str:
    """正确的工具定义"""
    # LangChain会自动处理config和runtime参数
    return f"Processing {param1} and {param2}"
```

**Q3**: Schema验证失败时如何优雅处理？
**A3**: 实现完善的异常处理机制：

```python
def robust_schema_tool_call(tool_func, tool_args):
    """健壮的Schema工具调用"""
    try:
        # Pydantic会自动进行参数验证
        result = tool_func.invoke(tool_args)
        return result, None
    except Exception as e:
        # 捕获验证错误和其他异常
        if "validation" in str(e).lower():
            error_msg = f"参数验证失败: {str(e)}"
        else:
            error_msg = f"工具执行失败: {str(e)}"
        print(f"⚠️ {error_msg}")
        return None, error_msg

# 使用示例
def handle_validation_errors():
    """处理验证错误示例"""
    model = init_model()
    tools = [search_user]
    model_with_tools = model.bind_tools(tools)
    
    # 故意发送无效参数
    bad_request = "查找用户名为空的用户"
    response = model_with_tools.invoke([HumanMessage(content=bad_request)])
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result, error = robust_schema_tool_call(search_user, tool_call['args'])
            if error:
                print(f"处理错误: {error}")
                # 可以返回友好的错误信息给用户
```

## 🏆 最佳实践总结

✅ **正确做法**：
- 为每个字段提供清晰的描述信息
- 使用合理的约束条件（min_length, max_length, ge, le等）
- 对于枚举类型使用json_schema_extra定义可选值
- 实现完善的错误处理和验证失败反馈
- 避免在工具参数中包含config和runtime保留参数

❌ **避免做法**：
- 忽略参数描述和验证约束
- 定义过于宽松或过于严格的验证规则
- 在工具签名中包含保留参数
- 不处理验证异常直接让程序崩溃
- 缺乏对边界情况的考虑

⚖️ **技术选型对比**

| 定义方式 | 验证能力 | 实现复杂度 | 适用场景 | 推荐指数 |
|----------|----------|------------|----------|----------|
| Pydantic Schema | 强大 | 中等 | 复杂参数验证 | ⭐⭐⭐⭐⭐ |
| Type Hints | 基础 | 简单 | 简单参数验证 | ⭐⭐⭐ |
| JSON Schema | 灵活 | 复杂 | 跨语言工具定义 | ⭐⭐⭐⭐ |

**选型建议**：
- 复杂业务逻辑：优先选择Pydantic Schema
- 简单工具：使用Type Hints
- 跨平台工具：考虑JSON Schema

## 📝 总结

Schema定义是确保工具参数准确性的关键机制：

✅ **Pydantic Schema**：提供强大的参数验证和约束  
✅ **结构化定义**：确保参数类型和格式正确  
✅ **保留参数处理**：正确处理config和runtime参数  
✅ **验证机制**：提前发现和处理参数错误  
✅ **错误处理**：优雅处理验证失败情况  

## 🔗 相关资源

- [Pydantic官方文档](https://docs.pydantic.dev/)
- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/tools/)
- [JSON Schema规范](https://json-schema.org/)

---
*本教程深入解析了工具Schema定义的高级应用。下一期我们将探索工具调用的性能优化技巧。*
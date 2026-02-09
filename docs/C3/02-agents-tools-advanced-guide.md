# LangChain实战开发教程（十三）：智能体工具高级应用与动态注册实战

> **解决智能体工具应用的3种技术方案对比**：掌握工具注册、ReAct模式和动态工具注册的高级技巧

## 🎯 本文目标

深入理解LangChain Agent工具系统的高级特性，掌握工具注册与使用、ReAct推理模式和动态工具注册的实现方法，学会构建具备动态扩展能力的智能体系统。

### 📚 核心知识点概览

通过本文你将掌握：
- **工具注册与管理**：高级工具注册和错误处理机制
- **ReAct推理模式**：推理与行动交替的智能决策模式
- **动态工具注册**：运行时动态注册和权限控制机制
- **中间件机制**：工具调用中间件和错误处理策略
- **权限控制系统**：基于用户角色的工具访问控制

### 🎯 使用场景判断

✅ **推荐使用高级工具管理**：
- 需要复杂错误处理和异常恢复的场景
- 要求动态扩展工具功能的应用
- 需要权限控制的多角色系统
- 智能决策和推理分析系统
- 复杂任务分解和协调执行

❌ **不建议使用**：
- 简单工具调用场景
- 静态工具配置已满足需求
- 对性能要求极高的实时系统
- 工具功能固定且不需要扩展

### 💡 核心概念解释

**ReAct模式**：一种"推理+行动"的智能体工作模式，智能体在简短的推理步骤和有针对性的工具调用之间交替进行，并将观察结果反馈到后续决策中，直到能够给出最终答案。

**动态工具注册**：允许在运行时根据需要动态注册、注销或过滤工具的能力，支持基于用户权限或上下文的工具访问控制。

## 🔧 实施三步走

### 一：工具注册与错误处理机制 ⚙️

**前置条件**：
- 掌握基础工具定义方法
- 理解中间件概念和使用
- 熟悉异常处理机制

**Agent使用工具的核心知识点**：

Agent与工具的交互是LangChain智能体系统的核心能力，主要包括以下几个关键方面：

1. **工具注册机制**：
   - 使用@tool装饰器定义工具
   - 将工具列表传递给Agent
   - 支持多种工具类型（函数工具、类工具等）

2. **工具调用流程**：
   - Agent分析用户请求并确定所需工具
   - 解析工具参数并执行调用
   - 处理工具返回结果
   - 根据结果决定下一步行动

3. **错误处理策略**：
   - 中间件模式进行统一错误处理
   - 自定义错误消息避免敏感信息泄露
   - 异常捕获和恢复机制

4. **中间件应用**：
   - wrap_tool_call装饰器处理工具调用
   - ModelResponse包装器处理模型响应
   - 支持多种中间件组合使用

对于统一的异常处理策略，关键实现就是利用`@wrap_tool_call`包装工具执行，在这里补货工具的执行异常，然后返回统一的异常调用工具消息

```python
import logging
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
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
        max_tokens=1500,
        max_retries=3,
    )

# 定义基础工具
@tool
def calculator(num1: float, operation: str, num2: float) -> float:
    """
    执行基本数学运算的计算器工具
    
    Args:
        num1: 第一个数字
        operation: 运算符 (+, -, *, /)
        num2: 第二个数字
        
    Returns:
        float: 计算结果
    """
    print(f"🧮 执行计算: {num1} {operation} {num2}")
    
    if operation == "+":
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        if num2 == 0:
            raise ValueError("除数不能为零")
        return num1 / num2
    else:
        raise ValueError(f"不支持的运算符: {operation}")

@tool
def weather_checker(city: str) -> str:
    """
    查询城市天气信息的工具
    
    Args:
        city: 城市名称
        
    Returns:
        str: 天气信息
    """
    print(f"🌤️ 查询 {city} 的天气")
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 15°C",
        "上海": "多云，温度 18°C",
        "广州": "雨天，温度 22°C",
        "深圳": "阴天，温度 20°C",
        "杭州": "晴天，温度 16°C"
    }
    return weather_data.get(city, f"暂无 {city} 的天气信息")

# 自定义工具错误处理中间件
@wrap_tool_call
def handle_tool_errors(request, handler):
    """
    工具执行错误处理中间件
    - 捕获工具执行异常
    - 返回自定义错误消息
    - 避免暴露敏感错误信息
    """
    try:
        return handler(request)
    except Exception as e:
        # 返回自定义错误消息给模型，不暴露敏感信息
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

def advanced_tool_demo():
    """高级工具使用演示"""
    print("🚀 开始 LangChain Agents 高级工具示例演示")
    
    # 初始化模型和工具
    llm = init_model()
    tools = [calculator, weather_checker]
    
    # 创建带有错误处理中间件的Agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是一个智能助手，可以根据用户的问题选择合适的工具来帮助解决问题。",
        middleware=[handle_tool_errors]  # 添加错误处理中间件
    )
    
    # 测试用例 - 包含可能导致错误的情况
    test_queries = [
        "今天北京的天气怎么样？",           # 正常查询
        "计算 25 乘以 4 等于多少？",      # 正常计算
        "请计算半径为5的圆周长",           # 需要π值的计算（会失败）
        "先计算 100 除以 5，然后告诉我上海的天气"  # 多步骤任务
    ]
    
    print(f"\n🎯 开始测试 {len(test_queries)} 个问题:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 20} 测试 {i}/{len(test_queries)} {'=' * 20}")
        print(f"❓ 问题: {query}")
        
        try:
            inputs = {"messages": [{"role": "user", "content": query}]}
            response = agent.invoke(inputs)
            
            print("💬 对话历史:")
            for msg in response["messages"]:
                if hasattr(msg, 'content'):
                    print(f"   {msg.type}: {msg.content}")
                elif isinstance(msg, dict):
                    print(f"   {msg.get('role', 'unknown')}: {msg.get('content', '')}")
                    
        except Exception as e:
            print(f"❌ 执行出错: {e}")
            continue
```

### 二：ReAct推理模式实现 🧠

```python
def react_loop_demo():
    """
    ReAct（推理+行动）模式演示
    智能体在推理步骤和工具调用之间交替进行
    """
    print("\n🎯 ReAct模式演示:")
    
    llm = init_model()
    tools = [calculator, weather_checker]
    
    agent = create_agent(
        llm, 
        tools, 
        system_prompt="""你是一个专业的智能助手，具有以下能力：
        1. 天气查询 - 可以查询中国主要城市的天气
        
        请根据用户的问题选择最合适的工具来解决问题。
        如果问题涉及多个步骤，请依次执行相应的工具调用。
        回答时要清晰、准确，并给出完整的解决方案。"""
    )
    
    queries = [
        "比较一下北京和上海今天的天气哪个更好"
    ]
    
    for query in queries:
        print(f"\n❓ 问题: {query}")
        try:
            inputs = {"messages": [HumanMessage(query)]}
            
            print("💬 对话历史 (ReAct过程):")
            last_type = None
            
            # 流式调用观察ReAct过程
            for step in agent.stream(inputs, stream_mode="values"):
                msg = step['messages'][-1]
                
                # 识别不同类型的步骤
                if last_type != msg.type:
                    print(f"\n   {msg.type}: ", end='')
                    last_type = msg.type
                print(msg.content, end='', flush=True)
            
            print("\n\n")
            
        except Exception as e:
            print(f"❌ 执行出错: {e}")
```

**ReAct模式核心机制**

ReAct (Reasoning + Acting) 模式的工作原理：

1. Reasoning（推理）：Agent分析问题并制定执行计划
   - 理解用户请求
   - 分析需要哪些工具
   - 制定执行步骤

2. Acting（行动）：执行选定的工具
   - 调用相应的工具函数
   - 获取执行结果

3. Observation（观察）：观察工具执行结果
   - 分析工具返回的数据
   - 评估是否达到目标

4. Repeat（重复）：根据结果决定下一步行动
   - 继续调用其他工具
   - 或返回最终答案

这种模式让Agent能够进行多步骤的智能决策。

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   用户请求   │───▶│   推理计划   │───▶│   工具调用   │
│             │    │             │    │             │
│ "比较天气"   │    │ "需要查询两  │    │ "查询北京和  │
│             │    │ 个城市天气"  │    │  上海天气"   │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   结果整合   │◀───│   分析结果   │◀───│   获取数据   │
│             │    │             │    │             │
│ "生成比较报  │    │ "分析天气数  │    │ "北京：晴天  │
│  告"        │    │  据差异"    │    │  上海：多云" │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 三：基于权限的动态工具过滤方案 🔐

第一种动态工具注册方案是基于用户权限的工具过滤机制。这种方式预先注册所有可能的工具，然后根据用户的角色或权限动态筛选可用的工具集。

> 核心实现：同样是利用`@wrap_model_call`来包装工具的执行，在这里做统一的工具过滤，然后覆盖默认注册的全量工具集

```python
from dataclasses import dataclass
from typing import Callable
from langchain.agents.middleware import ModelRequest, ModelResponse
from langgraph.prebuilt.tool_node import ToolCallRequest

@dataclass
class UserContext:
    """用户上下文信息"""
    user_role: str

@wrap_tool_call
def handle_dynamic_tool_call(request, handler):
    """处理动态工具调用"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Dynamic tool error: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )

@wrap_model_call
def filter_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    基于用户权限的工具过滤中间件
    根据用户角色动态筛选可用工具
    """
    # 从请求上下文中获取用户角色
    user_role = getattr(request.runtime, 'context', {}).get('user_role', 'guest')
    
    if user_role == "math":
        # 只允许访问计算器工具
        tools = [t for t in request.tools if t.name == "calculator"]
    elif user_role == 'search':
        # 只允许访问搜索工具（如果存在）
        tools = [t for t in request.tools if t.name in ["web_search", "calculator"]]
    elif user_role == 'admin':
        # 管理员可以访问所有工具
        tools = request.tools
    else:
        # 普通用户只能访问部分工具
        tools = [t for t in request.tools if t.name in ["weather_checker"]]
    
    return handler(request.override(tools=tools))

def dynamic_tool_filtering_demo():
    """
    动态工具过滤演示
    根据用户权限动态控制可用工具
    """
    print("🚀 动态工具过滤示例演示")
    
    llm = init_model()
    # 预先注册所有可能的工具
    tools = [calculator, weather_checker]
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是一个智能助手，可以根据用户的问题选择合适的工具来帮助解决问题。",
        middleware=[filter_tools]  # 添加工具过滤中间件
    )
    
    # 测试不同角色的用户
    test_cases = [
        ("math", "计算 25 乘以 4 等于多少？"),
        ("search", "计算 4 乘以 4 等于多少？"),  # search角色无法使用calculator
        ("guest", "今天北京的天气怎么样？")
    ]
    
    for role, query in test_cases:
        print(f"\n👤 用户角色: {role}")
        print(f"❓ 问题: {query}")
        
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                context=UserContext(user_role=role)  # 传递用户上下文
            )
            
            print("💬 对话历史:")
            for msg in response["messages"]:
                if hasattr(msg, 'content'):
                    print(f"   {msg.type}: {msg.content}")
                elif isinstance(msg, dict):
                    print(f"   {msg.get('role', 'unknown')}: {msg.get('content', '')}")
                    
        except Exception as e:
            print(f"❌ 执行出错: {e}")
```

### 四：运行时动态工具注册方案 🔄

第二种动态工具注册方案是在运行时动态添加工具。这种方式允许在Agent运行期间根据需要注册新的工具，提供了更大的灵活性。

> 这种适合从外部动态加载工具的场景，比如动态注册mcp tools

```python
# 动态工具注册中间件
class DynamicToolMiddleware:
    """动态工具注册中间件"""
    
    def __init__(self, dynamic_tools=None):
        self.dynamic_tools = dynamic_tools or []
    
    def wrap_model_call(self, request, handler):
        """在模型调用前添加动态工具"""
        # 将动态工具添加到请求中的工具列表
        updated = request.override(tools=[*request.tools, *self.dynamic_tools])
        return handler(updated)
    
    def wrap_tool_call(self, request, handler):
        """处理动态工具调用"""
        # 检查是否是动态工具
        for tool in self.dynamic_tools:
            if request.tool_call["name"] == tool.name:
                return handler(request.override(tool=tool))
        return handler(request)

def runtime_tool_registration_demo():
    """
    运行时工具注册演示
    在运行时动态注册和处理工具
    """
    print("\n🚀 运行时工具注册示例")
    
    llm = init_model()
    
    # 只注册静态工具
    agent = create_agent(
        model=llm,
        tools=[weather_checker],  # 只注册静态工具
        middleware=[DynamicToolMiddleware([calculator])]  # 动态注册计算器
    )
    
    query = "先计算 100 除以 5，然后告诉我上海的天气"
    print(f"\n❓ 问题: {query}")
    
    try:
        inputs = {"messages": [HumanMessage(query)]}
        print("💬 对话历史:")
        
        last_type = None
        for step in agent.stream(inputs, stream_mode="values"):
            msg = step['messages'][-1]
            if last_type != msg.type:
                print(f"\n   {msg.type}: ", end='')
                last_type = msg.type
            print(msg.content, end='', flush=True)
        
        print("\n\n")
    except Exception as e:
        print(f"❌ 执行出错: {e}")
```

## ❓ 常见问题解答

**Q1**: ReAct模式和普通工具调用有什么区别？
**A1**: 

| 特性 | 普通工具调用 | ReAct模式 |
|------|-------------|-----------|
| 决策过程 | 单步决策 | 推理+行动交替 |
| 执行流程 | 直接调用 | 循环推理执行 |
| 智能程度 | 固定逻辑 | 自适应推理 |
| 适用场景 | 简单任务 | 复杂推理任务 |
| 调试难度 | 简单 | 较复杂 |

```python
# 普通工具调用 - 直接执行
def simple_tool_call():
    # 直接调用工具，没有推理过程
    result = calculator.invoke({"num1": 10, "operation": "+", "num2": 5})
    return result

# ReAct模式 - 推理+行动
def react_approach():
    # Agent会先推理需要什么工具，然后执行
    # 可能需要多次推理和行动才能完成任务
    agent = create_agent(model, [calculator, weather_checker])
    response = agent.invoke({
        "messages": [{"role": "user", "content": "先计算10+5，再查询天气，最后比较结果"]}
    })
    # 这里Agent会自动推理执行步骤
    return response
```

**Q2**: 如何实现基于用户权限的工具访问控制？
**A2**: 通过中间件实现动态工具过滤：

```python
# 权限控制实现示例
def permission_control_example():
    """权限控制实现示例"""
    
    # 定义权限映射
    PERMISSION_MAP = {
        "admin": ["calculator", "weather_checker", "web_search"],
        "user": ["weather_checker"],
        "guest": ["weather_checker"],
        "math_expert": ["calculator"]
    }
    
    @wrap_model_call
    def permission_based_filter(request: ModelRequest, handler):
        """基于权限的工具过滤"""
        user_role = getattr(request.runtime, 'context', {}).get('user_role', 'guest')
        allowed_tools = PERMISSION_MAP.get(user_role, [])
        
        # 过滤工具列表
        filtered_tools = [
            tool for tool in request.tools 
            if tool.name in allowed_tools
        ]
        
        return handler(request.override(tools=filtered_tools))
    
    return permission_based_filter

# 使用示例
def secure_agent_with_permissions():
    """带权限控制的安全Agent"""
    llm = init_model()
    tools = [calculator, weather_checker]
    
    agent = create_agent(
        model=llm,
        tools=tools,
        middleware=[permission_control_example()]
    )
    
    # 不同角色的用户调用
    admin_context = UserContext(user_role="admin")
    guest_context = UserContext(user_role="guest")
    
    return agent, admin_context, guest_context
```

**Q3**: 动态工具注册的最佳时机是什么？
**A3**: 动态工具注册的最佳时机包括：

```
# 1. 应用启动时根据配置注册
def register_tools_at_startup():
    """启动时根据配置注册工具"""
    import json
    
    # 从配置文件加载工具配置
    with open("tools_config.json", "r") as f:
        config = json.load(f)
    
    dynamic_tools = []
    for tool_config in config.get("dynamic_tools", []):
        if tool_config.get("enabled", True):
            # 根据配置动态创建工具
            dynamic_tools.append(create_tool_from_config(tool_config))
    
    return dynamic_tools

# 2. 用户会话开始时根据权限注册
def register_tools_for_session(user_info):
    """根据用户信息注册个性化工具"""
    user_tools = []
    
    if user_info.get("department") == "finance":
        user_tools.extend([financial_calculator, budget_tracker])
    elif user_info.get("department") == "hr":
        user_tools.extend([employee_directory, schedule_manager])
    
    return user_tools

# 3. 运行时根据任务需求注册
def register_tools_for_task(task_type):
    """根据任务类型动态注册工具"""
    task_tools_map = {
        "research": [web_search, document_analyzer],
        "calculation": [advanced_calculator, unit_converter],
        "analysis": [data_analyzer, chart_generator]
    }
    
    return task_tools_map.get(task_type, [])

# 4. 事件驱动的工具注册
class EventDrivenToolRegistry:
    """事件驱动的工具注册器"""
    
    def __init__(self):
        self.tools = {}
        self.listeners = {}
    
    def register_event_listener(self, event_type, callback):
        """注册事件监听器"""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
    
    def trigger_event(self, event_type, data):
        """触发事件"""
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                callback(data)
    
    def add_tool_on_demand(self, tool_name, tool_func):
        """按需添加工具"""
        self.tools[tool_name] = tool_func
        self.trigger_event("tool_added", {"name": tool_name, "func": tool_func})

# 使用示例
registry = EventDrivenToolRegistry()

def on_tool_added(data):
    print(f"工具 {data['name']} 已添加")

registry.register_event_listener("tool_added", on_tool_added)
registry.add_tool_on_demand("dynamic_tool", lambda x: x * 2)
```

## 🏆 最佳实践总结

✅ **正确做法**：
- 实现完善的工具错误处理和异常恢复机制
- 使用中间件模式进行工具调用拦截和处理
- 根据用户权限动态控制工具访问
- 采用ReAct模式处理复杂推理任务
- 实现工具使用监控和日志记录

❌ **避免做法**：
- 忽略工具调用的错误处理
- 将敏感信息暴露在错误消息中
- 不验证动态工具的安全性
- 缺乏工具使用权限控制
- 不监控工具执行性能

## ⚖️ 技术选型对比

| 方案 | 静态注册 | 动态过滤 | 运行时注册 | 推荐指数 |
|------|---------|---------|-----------|----------|
| 灵活性 | 低 | 中等 | 高 | ⭐⭐⭐ |
| 安全性 | 高 | 高 | 中等 | ⭐⭐⭐⭐ |
| 性能 | 最优 | 良好 | 中等 | ⭐⭐⭐⭐⭐ |
| 复杂度 | 低 | 中等 | 高 | ⭐⭐ |
| 适用场景 | 固定功能 | 权限控制 | 动态扩展 | ⭐⭐⭐⭐ |

**选型建议**：
- 功能固定的系统：使用静态注册
- 需要权限控制：使用动态过滤
- 需要动态扩展：使用运行时注册

## 📝 总结

LangChain Agent工具系统提供了强大的功能扩展能力：

✅ **错误处理**：通过中间件实现优雅的错误处理和恢复  
✅ **ReAct模式**：支持推理与行动交替的智能决策  
✅ **动态注册**：运行时动态注册和权限控制  
✅ **权限控制**：基于用户角色的细粒度权限管理  
✅ **监控扩展**：完整的工具使用监控和日志记录  

## 🔗 相关资源

- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---
*本教程深入解析了Agent工具高级应用。下一期我们将探索智能体编排和工作流管理。*
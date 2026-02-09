"""
LangChain Agents 动态工具进阶使用示例
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config-zhipu.env')
load_dotenv(config_path)

# 初始化环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model_name = os.getenv('MODEL')


def init_model(model=model_name):
    """初始化 LLM Model"""
    return init_chat_model(
        model=model,
        model_provider="openai",
        temperature=0.7,
        timeout=30,
        max_tokens=1500,
        max_retries=3,
    )


# 定义一些基础工具
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


@tool
def web_search(query: str) -> str:
    """
    模拟网络搜索工具

    Args:
        query: 搜索关键词

    Returns:
        str: 搜索结果摘要
    """
    print(f"🔍 搜索: {query}")
    # 模拟搜索结果
    search_results = {
        "人工智能发展": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器...",
        "Python编程": "Python是一种高级编程语言，以其简洁易读的语法和强大的功能库而闻名...",
        "机器学习": "机器学习是人工智能的一个子领域，使计算机能够在不被明确编程的情况下从数据中学习..."
    }
    return search_results.get(query, f"关于'{query}'的搜索结果显示：这是相关的知识内容...")


@dataclass
class UserContext:
    user_role: str


@wrap_model_call
def filter_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on user permissions."""
    # 如果在创建代理时已知所有可能的工具，则可以预先注册它们，并根据状态、权限或上下文动态筛选哪些工具可以公开给模型。
    user_role = request.runtime.context.user_role

    if user_role == "math":
        # Admins get all tools
        tools = [t for t in request.tools if t.name == "calculator"]
    elif user_role == 'search':
        tools = [t for t in request.tools if t.name == "web_search"]
    elif user_role == 'admin':
        tools = request.tools
    else:
        # Regular users get read-only tools
        tools = []

    return handler(request.override(tools=tools))


def filter_pre_registered_tools():
    # 如果在创建代理时已知所有可能的工具，则可以预先注册它们，并根据状态、权限或上下文动态筛选哪些工具可以公开给模型。
    print("🚀 开始 LangChain Agents Tools示例演示")

    # 创建Agent
    # 初始化模型
    llm = init_model()

    # 预先注册所有的工具列表
    tools = [calculator, weather_checker, web_search]

    # 创建Agent，主要是基于model， tools，系统提示词来构建agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是一个智能助手，可以根据用户的问题选择合适的工具来帮助解决问题。",
        middleware=[filter_tools]
    )

    # math 角色的用户，用于调用数据计算
    response = agent.invoke({"messages": [{"role": "user", "content": "计算 25 乘以 4 等于多少？"}]},
                            context=UserContext(user_role="math"))
    for msg in response["messages"]:
        print(f"{msg.type}: {msg.content}")

    # search 角色的用户，用于访问math的工具，预期是无法正常调用工具
    response = agent.invoke({"messages": [{"role": "user", "content": "计算 4 乘以 4 等于多少？"}]},
                            context=UserContext(user_role="search"))
    for msg in response["messages"]:
        print(f"{msg.type}: {msg.content}")

    # 给一个guest角色，预期是无法正常调用工具
    response = agent.invoke({"messages": [{"role": "user", "content": "今天北京的天气怎么样？"}]},
                            context=UserContext(user_role="guest"))
    for msg in response["messages"]:
        print(f"{msg.type}: {msg.content}")


class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that registers and handles dynamic tools."""

    def wrap_model_call(self, request: ModelRequest, handler):
        # Add dynamic tool to the request
        # This could be loaded from an MCP server, database, etc.
        updated = request.override(tools=[*request.tools, calculator])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # Handle execution of the dynamic tool
        if request.tool_call["name"] == "calculator":
            return handler(request.override(tool=calculator))
        return handler(request)


def runtime_tool_registration():
    """
    当在运行时发现或创建工具时（例如，从 MCP 服务器加载、根据用户数据生成或从远程注册表中获取），您需要注册这些工具并动态处理它们的执行。
    :return:
    """
    print("🚀 运行时工具注册示例")
    # 初始化模型
    llm = init_model()

    agent = create_agent(
        model=llm,
        tools=[weather_checker],  # Only static tools registered here
        middleware=[DynamicToolMiddleware()],  # 动态工具注册
    )

    query = "先计算 100 除以 5，然后告诉我上海的天气"
    print(f"\n❓ 问题: {query}")
    try:
        inputs = {"messages": [HumanMessage(query)]}
        # 显示完整对话历史
        print("💬 对话历史:")
        # 流式调用：该模式会在智能体的每个执行步骤完成后传输中间数据，让开发者能够观察到完整的决策过程，如上例所示，值流模式会分四次更新数据：HumanMessage（用户输入）、AIMessage（模型初始响应）、ToolMessage（工具调用结果）和最终的AIMessage（总结回答）
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


if __name__ == "__main__":
    print("🔧 LangChain Agents 工具进阶使用示例")
    print("=" * 60)

    # 高级工具演示
    filter_pre_registered_tools()

    # 错误处理演示
    runtime_tool_registration()

    print("\n✅ 工具进阶示例演示完成！")

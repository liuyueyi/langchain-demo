"""
LangChain Agents 工具进阶使用示例
展示工具定义与基础使用，高阶使用（如异常处理策略、ReAct实现、Dynamic tools等）
"""

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


# 要自定义工具错误的处理方式，请使用 @wrap_tool_call 装饰器创建中间件
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


def basic_agent_demo():
    """基础Agent使用演示"""
    print("🚀 开始 LangChain Agents 基础示例演示")

    # 创建Agent
    # 初始化模型
    llm = init_model()

    # 定义工具列表
    tools = [calculator, weather_checker]

    # 创建Agent，主要是基于model， tools，系统提示词来构建agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是一个智能助手，可以根据用户的问题选择合适的工具来帮助解决问题。",
        middleware=[handle_tool_errors]  # 处理工具调用失败
    )

    # 测试用例
    test_queries = [
        "今天北京的天气怎么样？",
        "计算 25 乘以 4 等于多少？",
        "请计算半径为5的圆周长",
        "先计算 100 除以 5，然后告诉我上海的天气"
    ]

    print(f"\n🎯 开始测试 {len(test_queries)} 个问题:")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 20} 测试 {i}/{len(test_queries)} {'=' * 20}")
        print(f"❓ 问题: {query}")

        try:
            # 调用Agent，传入的是一个输入字典, 其中 messages 是一个消息列表, 除了使用字典方式表示消息之外，还可以通过 HumanMessage 的方式传入
            inputs = {"messages": [{"role": "user", "content": query}]}
            # 同步调用，直接获取返回结果
            response = agent.invoke(inputs)

            # 显示完整对话历史，对于agent，调用工具时，不需要像model一样，由我们来维护工具的执行；工具的完整执行链路都是由Agent来驱动的
            print("💬 对话历史:")
            for msg in response["messages"]:
                if hasattr(msg, 'content'):
                    print(f"   {msg.type}: {msg.content}")
                elif isinstance(msg, dict):
                    print(f"   {msg.get('role', 'unknown')}: {msg.get('content', '')}")

        except Exception as e:
            print(f"❌ 执行出错: {e}")
            continue


def react_loop():
    """
    智能体遵循 ReAct（“推理+行动”）模式，在简短的推理步骤和有针对性的工具调用之间交替进行，并将由此产生的观察结果反馈到后续决策中，直到能够给出最终答案。
    :return:
    """
    print("\n🎯 Stream访问演示:")

    llm = init_model()
    tools = [calculator, weather_checker]

    agent = create_agent(llm, tools, system_prompt="""你是一个专业的智能助手，具有以下能力：
            1. 天气查询 - 可以查询中国主要城市的天气

            请根据用户的问题选择最合适的工具来解决问题。
            如果问题涉及多个步骤，请依次执行相应的工具调用。
            回答时要清晰、准确，并给出完整的解决方案。""")

    queries = [
        "比较一下北京和上海今天的天气哪个更好"
    ]

    for query in queries:
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
    basic_agent_demo()

    # 错误处理演示
    react_loop()

    print("\n✅ 工具进阶示例演示完成！")

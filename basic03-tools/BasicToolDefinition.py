"""
LangChain Tools 基础定义示例
展示如何定义基本工具、注册工具以及使用装饰器创建工具
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# 加载环境变量
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
load_dotenv(config_path)

# 初始化环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model_name = os.getenv('MODEL')


def init_model(model=model_name):
    # 初始化 LLM Model
    return init_chat_model(model=model,
                           model_provider="openai",  # 指定模型厂商
                           temperature=0.7,  # 温度，控制返回更稳定还是更有创造力的结果
                           timeout=30,  # 设置超时时间，单位秒
                           max_tokens=1000,  # 限制响应中的令牌总数，从而有效地控制输出的长度。
                           max_retries=3,  # 最大失败重试次数
                           )


def pretty_print_ai_response(response):
    """
    美化的 AI 响应输出
    :param response: 大模型的返回
    :return:
    """
    separator = "=" * 60

    print(f"\n{separator}")
    print("🤖 AI 智能回复")
    print(separator)

    # 主要内容显示
    print(f"\n💬 回复内容:")
    if hasattr(response, 'content'):
        print(response.content)
    else:
        print(str(response))

        # 技术信息
    print(f"\n{separator}")
    print("📊 技术详情:")
    print(f"  📁 类型: {type(response).__name__}")

    # Token 使用情况
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        print(f"  💰 Token: {response.usage_metadata}")
    elif hasattr(response, 'usage') and response.usage:
        print(f"  💰 Token: {response.usage}")
    else:
        print("  💰 Token: 未提供")

    # 对象属性统计
    attr_count = len([attr for attr in dir(response) if not attr.startswith('_')])
    print(f"  🔍 属性数: {attr_count} 个")
    print(separator)


# 方式1： 创建工具最简单的方法是使用 @tool 装饰器。默认情况下，函数的文档字符串会成为工具的描述，帮助模型理解何时使用该工具：
@tool
def calculator(num1: float, operation: str, num2: float) -> float:
    """
    执行基本数学运算的计算器工具，当需要进行数学计算时，请调用这个工具
    
    Args:
        num1: 第一个数字
        operation: 运算符 (+, -, *, /)
        num2: 第二个数字
        
    Returns:
        float: 计算结果
    """
    print(f"执行计算: {num1} {operation} {num2}")

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


# 默认情况下，工具名称来源于函数名称。如果需要更具描述性的名称，可以进行覆盖；通过description来提供多大模型更友好的工具描述说明
@tool("weather_search", description="根据传入的城市返回对应的天气信息，当你需要查询天气时，调用这个工具!")
def weather_checker(city: str) -> str:
    """
    模拟查询天气的工具
    
    Args:
        city: 城市名称
        
    Returns:
        str: 天气信息
    """
    print(f"查询 {city} 的天气")
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 15°C",
        "上海": "多云，温度 18°C",
        "广州": "雨天，温度 22°C",
        "深圳": "阴天，温度 20°C",
        "杭州": "晴天，温度 16°C"
    }
    return weather_data.get(city, f"暂无 {city} 的天气信息")


# 方式2：直接通过StructuredTool()来创建工具
def define_tool_classically():
    """使用经典方式定义工具"""
    def multiply(a: float, b: float) -> float:
        """乘法运算"""
        print(f"执行乘法: {a} * {b}")
        return a * b

    from langchain_core.tools import StructuredTool
    
    class MultiplyInput(BaseModel):
        a: float = Field(description="第一个数字")
        b: float = Field(description="第二个数字")

    # 若工具只接收一个参数，使用Tool() 来创建；若工具接收多个参数，使用StructuredTool() 来创建
    # 通过 MultiplyInput 来定义传参说明
    multiplication_tool = StructuredTool(
        name="MultiplicationTool",
        description="执行两个数字的乘法运算，接收两个数字类型的参数",
        func=multiply,
        args_schema=MultiplyInput
    )
    return multiplication_tool


def basic_tool_demo():
    """基础工具使用演示"""
    print("🚀 开始 LangChain Tools 基础示例演示")

    model = init_model()

    # 1. 使用 @tool 装饰器定义的工具
    print("\n1️⃣ 使用 @tool 装饰器定义的计算器工具:")
    # step1: 创建模型并绑定工具
    cal_model = model.bind_tools([calculator], tool_choice="any")
    msg_list = [HumanMessage("计算 10 + 5 的结果")]
    # step2: 调用模型
    response = cal_model.invoke(msg_list)

    # step3: 处理工具调用
    for tool_call in response.tool_calls:
        print(f"工具调用: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")

        # step4: 处理工具调用结果
        if tool_call['name'] == 'calculator':
            tool_result = calculator.invoke(tool_call)
            print(f"工具调用结果: {tool_result}")
            msg_list.append(tool_result)

    # step5: 将返回结果回传给大模型
    res = model.invoke(msg_list)
    pretty_print_ai_response(res)

    # 2. 使用经典方式定义的工具
    print("\n2️⃣ 使用经典方式定义的乘法工具:")
    classic_tool = define_tool_classically()
    cal_model = model.bind_tools([classic_tool], tool_choice="any")
    msg_list = [HumanMessage("计算 11 * 5 的结果")]
    response = cal_model.invoke(msg_list)
    for tool_call in response.tool_calls:
        print(f"工具调用: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")

        # step4: 处理工具调用结果
        if tool_call['name'] == classic_tool.name:
            # 使用args中的参数字典调用工具
            tool_result = classic_tool.invoke(tool_call)
            print(f"工具调用结果: {tool_result}")
            msg_list.append(tool_result)
    res = model.invoke(msg_list)
    pretty_print_ai_response(res)

    # 3. 查看工具的基本信息
    print("\n3️⃣ 工具基本信息:")
    print(f"计算器工具名称: {calculator.name}")
    print(f"计算器工具描述: {calculator.description}")
    print(f"天气工具名称: {weather_checker.name}")
    print(f"天气工具描述: {weather_checker.description}")
    print(f"经典工具名称: {classic_tool.name}")
    print(f"经典工具描述: {classic_tool.description}")


if __name__ == "__main__":
    basic_tool_demo()

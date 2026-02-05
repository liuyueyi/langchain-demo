"""
工具回调
"""
import datetime
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
load_dotenv(config_path)

# 初始化环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model = os.getenv('MODEL')


def pretty_print_ai_response_prefix(response_type="sync"):
    separator = "=" * 60

    print(f"\n{separator}")
    if response_type == "stream":
        print("🤖 AI 流式回复中...")
    else:
        print("🤖 AI 智能回复")
    print(separator)


def pretty_print_ai_response_suffix(response):
    separator = "=" * 60
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
        print(f"  💰 Token: 未返回")

    # 对象属性统计
    attr_count = len([attr for attr in dir(response) if not attr.startswith('_')])
    print(f"  🔍 属性数: {attr_count} 个")
    print(separator)


def pretty_print_ai_response(response):
    """
    美化的 AI 响应输出
    :param response: 大模型的返回
    :return:
    """
    pretty_print_ai_response_prefix("sync")

    # 主要内容显示
    print(f"\n💬 回复内容:")
    if hasattr(response, 'content'):
        print(response.content)
    else:
        print(str(response))

    pretty_print_ai_response_suffix(response)


def init_model(model):
    # 初始化 LLM Model
    return init_chat_model(model=model,
                           model_provider="openai",  # 指定模型厂商
                           temperature=0.7,  # 温度，控制返回更稳定还是更有创造力的结果
                           timeout=30,  # 设置超时时间，单位秒
                           max_tokens=1000,  # 限制响应中的令牌总数，从而有效地控制输出的长度。
                           max_retries=3,  # 最大失败重试次数
                           )


@tool
def now_time(area):
    """
    根据地区，获取对应地区的当前时间
    :param area: 地区名称，如 'Asia/Shanghai', 'America/New_York'
    :return: 格式化的当前时间字符串
    """
    try:
        print(f"进入工具调用 {area}")
        # 获取指定时区的当前时间
        tz = datetime.timezone.utc if area.lower() == 'utc' else datetime.datetime.now(
            datetime.timezone.utc).astimezone().tzinfo
        current_time = datetime.datetime.now(tz)
        ans = f"{area} 当前时间是 {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        print(f"工具调用，返回：{ans}")
        return ans
    except Exception as e:
        print(f"工具调用失败：{str(e)}")
        return f"无法获取 {area} 的当前时间: {str(e)}"


def tool_calling(model):
    model = init_model(model)

    msg_list = []

    # 默认情况下，模型可以根据用户输入自由选择要使用的绑定工具。但是，当希望强制选择一个工具，可以添加参数的tool_choice = 'any'：
    model_with_tools = model.bind_tools([now_time])
    msg_list.append(HumanMessage("现在纽约几点了？"))

    # Step1: 调用大模型，回调工具获取当前时间
    response = model_with_tools.invoke(msg_list)

    # step2: 执行工具并收集结果
    for tool_call in response.tool_calls:
        print(f"工具调用: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")

        # 执行工具函数
        if tool_call['name'] == 'now_time':
            tool_result = now_time.invoke(tool_call)
            print(f"工具调用结果: {tool_result}")
            msg_list.append(tool_result)

    # step3: 将返回结果回传给大模型
    res = model.invoke(msg_list)
    pretty_print_ai_response(res)


print("--" * 30 + " 工具调用 " + "--" * 30)
# 请注意，要选择一个支持tool的模型
tool_calling("Qwen/Qwen3-8B")

# 一个演示的示例
'''
工具调用: now_time
参数: {'area': 'America/New_York'}
工具调用结果: {'name': 'now_time', 'args': {'area': 'America/New_York'}, 'id': '019c2d19fa16d64121353d6843a85413', 'type': 'tool_call'}

============================================================
🤖 AI 智能回复
============================================================

💬 回复内容:


============================================================
📊 技术详情:
  📁 类型: AIMessage
  💰 Token: {'input_tokens': 187, 'output_tokens': 22, 'total_tokens': 209, 'input_token_details': {}, 'output_token_details': {'reasoning': 0}}
  🔍 属性数: 48 个
============================================================
'''

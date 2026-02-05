import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
load_dotenv(config_path)

# 初始化环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model = os.getenv('MODEL')


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


def simple_invoke(model):
    """
    基础的基于Model的大模型同步访问，设置超时时间、温度、最大token限制
    :param model:
    :return:
    """
    # 初始化 LLM Model
    model = init_chat_model(model=model,
                            model_provider="openai",  # 指定模型厂商
                            temperature=0.7,  # 温度，控制返回更稳定还是更有创造力的结果
                            timeout=30,  # 设置超时时间，单位秒
                            max_tokens=1000,  # 限制响应中的令牌总数，从而有效地控制输出的长度。
                            max_retries=3,  # 最大失败重试次数
                            )

    # 直接使用model进行大模型的交互
    response = model.invoke("请写一首关于颜色的五言绝句")
    pretty_print_ai_response(response)


print("--" * 30 + " model直接同步访问 " + "--" * 30)
simple_invoke(model)

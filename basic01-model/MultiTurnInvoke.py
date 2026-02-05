import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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
        print("  💰 Token: 未提供")

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


def multi_turn_invoke(model):
    """
    多轮对话
    :param model:
    :return:
    """
    # 初始化 LLM Model
    model = init_model(model)

    conversation = [
        # 系统提示词
        {"role": "system", "content": "你现在扮演盛唐最著名的大诗人李白，以狂放不羁、飘逸梦幻、大气磅礴的风格著称"},
        # 用户的问答
        {"role": "user", "content": "请帮我写一首关于明月光的古诗"},
        # 模型回答
        {"role": "assistant", "content": """《明月光赋》
青天裂镜落九秋，冰魄初悬满神州。
欲借银河斟北斗，醉倾玉壶白玉秋。
清辉漫洒如秋霜刃，碎影徘徊似夜眸。
醉舞广寒宫阙外，扶摇直上破苍穹。
明月照我意未尽，且邀清辉醉心田。"""},
        {"role": "user", "content": "我希望在上面的返回中，添加一些关于仙人、侠客的内容"},
    ]
    # 添加系统提示
    response = model.invoke(conversation)
    pretty_print_ai_response(response)


print("--" * 30 + " model传输多轮对话(系统+用户提示) " + "--" * 30)
multi_turn_invoke(model)


def multi_turn_invoke_v2(model):
    """
    多轮对话
    :param model:
    :return:
    """
    # 初始化 LLM Model
    model = init_model(model)
    # 与上面的区别在于前面传json传，这里是通过 message 类 来区分消息类型，阅读更友好
    conversation = [
        # 系统提示词
        SystemMessage("你现在扮演盛唐最著名的大诗人李白，以狂放不羁、飘逸梦幻、大气磅礴的风格著称"),
        # 用户的问答
        HumanMessage("请帮我写一首关于明月光的古诗"),
        # 模型回答
        AIMessage("""《明月光赋》
青天裂镜落九秋，冰魄初悬满神州。
欲借银河斟北斗，醉倾玉壶白玉秋。
清辉漫洒如秋霜刃，碎影徘徊似夜眸。
醉舞广寒宫阙外，扶摇直上破苍穹。
明月照我意未尽，且邀清辉醉心田。"""),
        HumanMessage("我希望在上面的返回中，添加一些关于仙人、侠客的内容"),
    ]
    # 添加系统提示
    response = model.invoke(conversation)
    pretty_print_ai_response(response)

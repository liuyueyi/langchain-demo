import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
load_dotenv(config_path)

# 初始化环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model = os.getenv('MODEL')


def init_model(model):
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

    print(separator)


def basic_call(model):
    """
    基础消息使用示例
    展示SystemMessage、HumanMessage、AIMessage的基本用法
    """
    model = init_model(model)

    # 系统提示词 - 设定AI角色和行为
    system_msg = SystemMessage("你是一个幽默的聊天大师，擅长以各种风趣、有梗的话语和人交流沟通")

    # 用户输入 - 模拟真实对话
    human_msg = HumanMessage("今天上班的路上，看到一条狗在追汽车")

    # AI历史回复 - 展示对话延续性
    ai_msg = AIMessage(
        "啊呀，这不就是传说中的\"狗追豪华版汽车\"吗？想想看，要是有人在公司年会上抽一辆车，估计那条狗都会替主人开心得摇尾巴吧！")

    # 继续对话
    res = model.invoke([system_msg, human_msg, ai_msg, HumanMessage("一般的公司年会可不会有车作为奖品了~")])
    pretty_print_ai_response(res)


if __name__ == "__main__":
    basic_call(model)
    # 更多的示例，可以参照 MessageComprehensiveDemo.py

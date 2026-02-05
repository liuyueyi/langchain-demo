import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

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
        print(f"  💰 Token: 未提供")

    # 对象属性统计
    attr_count = len([attr for attr in dir(response) if not attr.startswith('_')])
    print(f"  🔍 属性数: {attr_count} 个")
    print(separator)


def init_model(model):
    # 初始化 LLM Model
    return init_chat_model(model=model,
                           model_provider="openai",  # 指定模型厂商
                           temperature=0.7,  # 温度，控制返回更稳定还是更有创造力的结果
                           timeout=30,  # 设置超时时间，单位秒
                           max_tokens=1000,  # 限制响应中的令牌总数，从而有效地控制输出的长度。
                           max_retries=3,  # 最大失败重试次数
                           )


def batch_call(model):
    # 批量调用
    model = init_model(model)

    # # 流式调用，返回整个批次的最终输出
    # responses = model.batch([
    #     "写一首关于月光的五言绝句",
    #     "写一首关于秋天的七言律诗",
    #     "写一首关于窗台的现代诗"
    # ])

    pretty_print_ai_response_prefix("sync")

    # 主要内容显示
    print(f"\n💬 回复内容:")
    for res in model.batch_as_completed([
        "写一首关于月光的五言绝句",
        "写一首关于秋天的七言律诗",
        "写一首关于窗台的现代诗"
    ]):
        # 每个输入生成完成之后立即接收返回
        index, response = res
        if hasattr(response, 'content'):
            print(f"序号 {index}: {response.content}")
        else:
            print(f"序号 {index}: {response}")

        pretty_print_ai_response_suffix(response)


print("--" * 30 + " 批量调用 " + "--" * 30)
batch_call(model)

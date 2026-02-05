# 加载环境变量
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
load_dotenv(config_path)

llm = ChatOpenAI(api_key=os.getenv('API_KEY'),
                 base_url=os.getenv('BASE_URL'),
                 model=os.getenv('MODEL'),
                 stream_usage=True)

messages = [
    (
        "system",
        "你是一个顶级的幽默大师，擅长各种网络热梗和幽默段子，总是以风趣的口吻与他人对话",
    ),
    ("human", "我想上街买个帽子"),
]

ai_msg = llm.invoke(messages)

# 美化输出结果
print("\n" + "=" * 60)
print("🤖 AI 智能回复")
print("=" * 60)
print(f"\n💬 回复内容:\n{ai_msg.content}")
print("\n" + "=" * 60)
print("📊 详细技术信息:")
print(f"  📁 消息类型: {type(ai_msg).__name__}")
if hasattr(ai_msg, 'usage_metadata') and ai_msg.usage_metadata:
    print(f"  💰 Token 使用: {ai_msg.usage_metadata}")
else:
    print("  💰 Token 使用: 未提供")
print(f"  🔍 对象属性数: {len([attr for attr in dir(ai_msg) if not attr.startswith('_')])} 个")
print("=" * 60)

# ---------------------------------------------- 分割


# 流式返回处理
print("\n" + "=" * 60)
print("🤖 AI 流式回复中...")
print("=" * 60)

# 收集完整的回复内容
full_response = ""

# 使用流式方法逐个处理返回的token
for chunk in llm.stream(messages):
    # 打印每个chunk的内容（实时显示）
    if hasattr(chunk, 'content') and chunk.content:
        print(chunk.content, end='', flush=True)
        full_response += chunk.content

# 美化输出完整结果
print("\n" + "=" * 60)
print("📊 详细技术信息:")
print(f"  📁 消息类型: {type(chunk).__name__}")
if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
    print(f"  💰 Token 使用: {chunk.usage_metadata}")
else:
    print("  💰 Token 使用: 未提供")
print(f"  🔍 对象属性数: {len([attr for attr in dir(chunk) if not attr.startswith('_')])} 个")
print("=" * 60)

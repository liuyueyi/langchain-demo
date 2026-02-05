"""
流式调用的场景演示
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

config_path = os.path.join(Path(__file__).resolve().parents[2], 'config.env')
load_dotenv(config_path)

# 初始化环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model = os.getenv('MODEL')


def enhanced_stream_output(prompt, model):
    """增强版流式输出，支持更多控制选项"""
    full_response = ""
    token_stats = {"input": 0, "output": 0}

    print("🤖 AI正在思考中...")
    print("-" * 50)

    try:
        for chunk in model.stream(prompt):
            # 实时输出内容
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end='', flush=True)
                full_response += chunk.content

            # 收集token统计
            if hasattr(chunk, 'usage_metadata'):
                usage = chunk.usage_metadata
                if usage:
                    token_stats["input"] = usage.get("input_tokens", 0)
                    token_stats["output"] = usage.get("output_tokens", 0)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断了流式输出")
    except Exception as e:
        print(f"\n\n❌ 流式调用出错: {e}")

    return full_response, token_stats


def init_model(model):
    # 初始化 LLM Model
    return init_chat_model(model=model,
                           model_provider="openai",  # 指定模型厂商
                           temperature=0.7,  # 温度，控制返回更稳定还是更有创造力的结果
                           timeout=30,  # 设置超时时间，单位秒
                           max_tokens=1000,  # 限制响应中的令牌总数，从而有效地控制输出的长度。
                           max_retries=3,  # 最大失败重试次数
                           )


def long_text_streaming(model_name):
    """长文本流式生成示例"""

    model = init_model(model_name)

    long_prompt = """请写一篇关于人工智能未来发展的详细文章，
    包括技术趋势、社会影响、伦理考量等方面，至少1000字。"""

    print("📝 开始生成长篇文章...")
    full_text, stats = enhanced_stream_output(long_prompt, model)

    print(f"\n\n📋 文章统计:")
    print(f"字数: {len(full_text)} 字符")
    print(f"Token消耗: 输入{stats['input']}, 输出{stats['output']}")
    print(f"生成效率: {stats['output'] / len(full_text):.2f} tokens/字符")


long_text_streaming(model)


def code_generation_stream(model_name):
    """代码生成的流式预览"""
    code_prompt = "用Python写一个快速排序算法，并添加详细注释"

    print("💻 正在生成代码...")
    print("=" * 60)

    model = init_model(model_name)
    for chunk in model.stream(code_prompt):
        # 代码高亮效果模拟
        if chunk.content:
            if 'def' in chunk.content or 'class' in chunk.content:
                print(f"\033[94m{chunk.content}\033[0m", end='')  # 蓝色
            elif 'import' in chunk.content:
                print(f"\033[92m{chunk.content}\033[0m", end='')  # 绿色
            elif '#' in chunk.content:
                print(f"\033[93m{chunk.content}\033[0m", end='')  # 黄色
            else:
                print(chunk.content, end='')

    print("\n" + "=" * 60)
    print("✅ 代码生成完成!")

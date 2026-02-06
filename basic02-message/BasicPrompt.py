"""
提示词
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

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


def simple_text_prompt(model):
    """
    基础的基于Model的大模型同步访问，设置超时时间、温度、最大token限制
    :param model:
    :return:
    """
    # 创建 LLM Model
    model = init_model(model)

    # 直接使用字符串作为提示词，直接在invoke中传入一个字符串
    res = model.invoke("写一首关于池塘的古风歌曲")
    pretty_print_ai_response(res)

    # 使用 Message 对象来作为传送提示词，此时invoke接收的是一个列表，通常是 SystemMessage -> HumanMessage -> AIMessage 这种顺序循环
    user_msg = HumanMessage("写一首关于友情的古风歌曲")
    res = model.invoke([user_msg])
    pretty_print_ai_response(res)

    # 也可以使用字典格式传输提示词
    msg_list = [
        {"role": "system", "content": "你是一个古风编词作曲大师，擅长写各种类型的故事歌曲"},
        {"role": "user", "content": "写一首关于事业的古风歌曲"},
    ]
    res = model.invoke(msg_list)
    pretty_print_ai_response(res)


# simple_text_prompt("Qwen/Qwen3-8B")


def prompt_template(model):
    """
    提示词模板, 基于python语法中的 f-string 方式进行变量替换
    这种方式，有以下限制:
    - 不支持嵌套访问方式：如 {user.name} 这是非法的
    - 不支持格式化：如 {price:.2f} 这种保留两位小数的方式也不行
    - 不支持表达式：如 {x+y} 这种也非法
    - 不支持函数调用: 如 {str.upper()} 这种不行
    - 循环or条件判断: 不支持
    - 数组选择：不支持 {items[0]} 这种方式
    :param model:
    :return:
    """
    template = """
你是一个起名大师，擅长结合古诗词、五行八字给人取出好听、寓意好、五行圆满的名字，你应该返回五个名字，并解释每个名字的寓意。
下面是需要取名的信息: 
{info}
"""
    prompt = PromptTemplate.format_prompt(template, info="26年2月6日 10:01分出生的小女孩，姓:钱")
    # 转换为 HumanMessage
    # user_message = prompt.to_messages()
    # 转换为文本
    user_txt = prompt.to_string()
    print(user_txt)

    # res = init_model(model).invoke(user_txt)
    # pretty_print_ai_response(res)

    # 如果提示词中本身就有 {}，比如提示词中有json的数据，此时针对不需要做关键词替换的地方，使用双层括号 {{ }}
    print(PromptTemplate.format_prompt("不需要转换的 ={{not_var}}= 需要替换的 ={info}=", info="哈哈⌚️",
                                       not_var="不被替换"))


def prompt_template_by_mustache(model = model):
    """
    提示词模板, 基于mustache语法进行变量替换
    :param model:
    :return:
    """
    # 基础mustache模板示例
    template = """
你是一个起名大师，擅长结合古诗词、五行八字给人取出好听、寓意好、五行圆满的名字，你应该返回五个名字，并解释每个名字的寓意。
下面是需要取名的信息: 
{{info}}
"""
    
    # 正确的mustache模板创建方式
    prompt_template = PromptTemplate(
        template=template,
        template_format="mustache",
        input_variables=["info"]
    )
    
    # 正确的格式化方式
    prompt = prompt_template.format(info="26年2月6日 10:01分出生的小女孩，姓:钱")
    print("=== 基础mustache模板 ===")
    print(prompt)

    res = init_model(model).invoke(prompt)
    pretty_print_ai_response(res)
    
    # 更复杂的mustache模板示例，直接参考 MustacheTemplateDemo.py


# 只在需要时调用
prompt_template_by_mustache(model)

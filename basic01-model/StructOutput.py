"""
结构化输出
"""

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


def struct_output(model):
    """
    Pydantic 模型提供最丰富的功能集，包括字段验证、描述和嵌套结构。
    :param model:
    :return:
    """
    model = init_model(model)

    from pydantic import BaseModel, Field
    class Movie(BaseModel):
        """A movie with details."""
        title: str = Field(..., description="The title of the movie")
        year: int = Field(..., description="The year the movie was released")
        director: str = Field(..., description="The director of the movie")
        rating: float = Field(..., description="The movie's rating out of 10")

    model_with_structure = model.with_structured_output(Movie)
    response = model_with_structure.invoke("请提供周星驰的电影《功夫》的详细信息")
    print(response)


def struct_output_v2(model):
    """
    Python 的 TypedDict 为 Pydantic 模型提供了一个更简单的替代方案，非常适合不需要运行时验证的情况。
    :param model:
    :return:
    """
    model = init_model(model)

    from typing_extensions import TypedDict, Annotated
    class MovieDict(TypedDict):
        """A movie with details."""
        title: Annotated[str, ..., "The title of the movie"]
        year: Annotated[int, ..., "The year the movie was released"]
        director: Annotated[str, ..., "The director of the movie"]
        rating: Annotated[float, ..., "The movie's rating out of 10"]

    model_with_structure = model.with_structured_output(MovieDict)
    response = model_with_structure.invoke("请提供周星驰的电影《功夫》的详细信息")
    print(response)


def struct_output_v3(model):
    """
    json schema 的方式定义结构化返回
    :param model:
    :return:
    """
    model = init_model(model)
    json_schema = {
        "title": "Movie",
        "description": "A movie with details",
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the movie"
            },
            "year": {
                "type": "integer",
                "description": "The year the movie was released"
            },
            "director": {
                "type": "string",
                "description": "The director of the movie"
            },
            "rating": {
                "type": "number",
                "description": "The movie's rating out of 10"
            }
        },
        "required": ["title", "year", "director", "rating"]
    }

    model_with_structure = model.with_structured_output(
        json_schema,
        method="json_schema",
    )
    response = model_with_structure.invoke("请提供周星驰的电影《功夫》的详细信息")
    print(response)


print("--" * 30 + " 结构化输出 " + "--" * 30)
struct_output_v3(model)

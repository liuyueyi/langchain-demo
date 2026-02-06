"""
Message全面使用示例
展示SystemMessage、HumanMessage、AIMessage、ToolMessage的各种使用场景
"""

import os
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage,
    FunctionMessage
)
from langchain_core.tools import tool

# 加载配置
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config-zhipu.env')
load_dotenv(config_path)

# 初始化环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model_name = os.getenv('MODEL')


def init_model(model):
    """初始化模型"""
    return init_chat_model(
        model=model,
        model_provider="openai",
        temperature=0.7,
        timeout=30,
        max_tokens=1000,
        max_retries=3,
    )


def pretty_print_ai_response(response):
    """美化AI响应输出"""
    separator = "=" * 60
    print(f"\n{separator}")
    print("🤖 AI 智能回复")
    print(separator)
    
    print(f"\n💬 回复内容:")
    if hasattr(response, 'content'):
        print(response.content)
    else:
        print(str(response))
        
    print(f"\n{separator}")
    print("📊 技术详情:")
    print(f"  📁 类型: {type(response).__name__}")
    
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        print(f"  💰 Token: {response.usage_metadata}")
    elif hasattr(response, 'usage') and response.usage:
        print(f"  💰 Token: {response.usage}")
    else:
        print("  💰 Token: 未提供")
    print(separator)


def system_message_examples():
    """SystemMessage使用示例"""
    print("=== SystemMessage使用示例 ===")
    
    model = init_model(model_name)
    
    # 示例1：角色设定
    print("1. 角色设定示例：")
    system_role = SystemMessage("你是一位资深的Python开发专家，擅长代码审查和最佳实践指导")
    human_code = HumanMessage("请帮我审查这段代码：\n```python\ndef calculate_sum(numbers):\n    result = 0\n    for num in numbers:\n        result += num\n    return result\n```")
    
    response1 = model.invoke([system_role, human_code])
    pretty_print_ai_response(response1)
    
    # 示例2：行为约束
    print("\n2. 行为约束示例：")
    system_constraint = SystemMessage("""
    你是一个严格的语言老师，请用以下规则回复：
    1. 必须使用正式、礼貌的语言
    2. 回答要简洁明了，不超过100字
    3. 如果不知道答案，直接说"我不知道"
    """)
    human_question = HumanMessage("世界上最高的山是什么？")
    
    response2 = model.invoke([system_constraint, human_question])
    pretty_print_ai_response(response2)
    
    # 示例3：多角色切换
    print("\n3. 多角色切换示例：")
    system_poet = SystemMessage("你现在是一位古代诗人，说话要有古风韵味")
    human_topic = HumanMessage("请写一首关于春天的诗")
    
    response3 = model.invoke([system_poet, human_topic])
    pretty_print_ai_response(response3)


def human_message_examples():
    """HumanMessage使用示例"""
    print("\n=== HumanMessage使用示例 ===")
    
    model = init_model(model_name)
    
    # 示例1：基础问答
    print("1. 基础问答：")
    question = HumanMessage("Python中列表和元组有什么区别？")
    response = model.invoke([question])
    pretty_print_ai_response(response)
    
    # 示例2：带上下文的提问
    print("\n2. 带上下文的提问：")
    context = SystemMessage("你正在帮助用户学习数据结构")
    question_with_context = HumanMessage("能详细解释一下二叉树的遍历方式吗？")
    response = model.invoke([context, question_with_context])
    pretty_print_ai_response(response)
    
    # 示例3：多轮对话延续
    print("\n3. 多轮对话延续：")
    conversation = [
        HumanMessage("我想学习机器学习"),
        AIMessage("很好的选择！机器学习是人工智能的重要分支。你对哪个方面特别感兴趣？"),
        HumanMessage("我想先了解监督学习和无监督学习的区别")
    ]
    response = model.invoke(conversation)
    pretty_print_ai_response(response)


def ai_message_examples():
    """AIMessage使用示例"""
    print("\n=== AIMessage使用示例 ===")
    
    model = init_model(model_name)
    
    # 示例1：模拟历史对话
    print("1. 模拟历史对话：")
    conversation_history = [
        SystemMessage("你是一个旅游咨询助手"),
        HumanMessage("我想去北京旅游，有什么推荐吗？"),
        AIMessage("北京有很多值得游览的地方！推荐您参观故宫、天坛、颐和园等历史文化景点，还有798艺术区等现代文化场所。"),
        HumanMessage("这些地方的门票价格怎么样？")
    ]
    response = model.invoke(conversation_history)
    pretty_print_ai_response(response)
    
    # 示例2：带有思考过程的AI回复
    print("\n2. 带有思考过程的AI回复：")
    complex_conversation = [
        SystemMessage("你是一个逻辑推理专家，请展示你的思考过程"),
        HumanMessage("如果所有的猫都是动物，汤姆是猫，那么汤姆是什么？"),
        AIMessage("""
        让我来分析这个逻辑推理题：
        
        已知条件：
        1. 所有的猫都是动物（前提）
        2. 汤姆是猫（事实）
        
        推理过程：
        根据三段论推理：
        - 大前提：所有猫都是动物
        - 小前提：汤姆是猫
        - 结论：汤姆是动物
        
        因此，汤姆是动物。
        """),
        HumanMessage("那如果我还告诉你汤姆会飞呢？")
    ]
    response = model.invoke(complex_conversation)
    pretty_print_ai_response(response)


# 定义测试工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，气温-2°C到8°C，西北风3-4级",
        "上海": "多云，气温3°C到12°C，东南风2-3级",
        "广州": "小雨，气温15°C到22°C，微风"
    }
    return weather_data.get(city, f"暂无{city}的天气信息")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 简单的安全计算（实际项目中应使用更安全的方式）
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except:
        return f"无法计算表达式: {expression}"


def tool_message_examples():
    """ToolMessage使用示例"""
    print("\n=== ToolMessage使用示例 ===")
    
    model = init_model(model_name)
    
    # 示例1：工具调用基础示例
    print("1. 工具调用基础示例：")
    
    # 绑定工具到模型
    model_with_tools = model.bind_tools([get_weather, calculate])
    
    # 用户询问天气
    user_query = HumanMessage("北京今天天气怎么样？")
    
    # 第一步：AI决定是否需要调用工具
    print("第一步：AI分析是否需要工具")
    response = model_with_tools.invoke([user_query])
    
    if response.tool_calls:
        print("发现工具调用需求：")
        for tool_call in response.tool_calls:
            print(f"  工具名称: {tool_call['name']}")
            print(f"  参数: {tool_call['args']}")
            
            # 执行工具调用
            if tool_call['name'] == 'get_weather':
                tool_result = get_weather.invoke(tool_call)
                print(f"  执行结果: {tool_result}")
                
                # 创建ToolMessage
                tool_message = ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id'] if 'id' in tool_call else 'tool_1'
                )
                
                # 第二步：AI基于工具结果生成最终回答
                print("\n第二步：AI生成最终回答")
                final_response = model.invoke([user_query, response, tool_message])
                pretty_print_ai_response(final_response)
    
    # 示例2：多工具调用
    print("\n2. 多工具调用示例：")
    
    complex_query = HumanMessage("帮我计算一下(25+15)*2，然后查询上海的天气")
    
    print("AI分析复杂请求...")
    response = model_with_tools.invoke([complex_query])
    
    if response.tool_calls:
        tool_messages = []
        
        for tool_call in response.tool_calls:
            print(f"调用工具: {tool_call['name']}")
            
            if tool_call['name'] == 'calculate':
                result = calculate.invoke(tool_call)
                tool_msg = ToolMessage(content=result, tool_call_id=tool_call.get('id', 'calc_1'))
                tool_messages.append(tool_msg)
                print(f"计算结果: {result}")
                
            elif tool_call['name'] == 'get_weather':
                result = get_weather.invoke(tool_call)
                tool_msg = ToolMessage(content=result, tool_call_id=tool_call.get('id', 'weather_1'))
                tool_messages.append(tool_msg)
                print(f"天气信息: {result}")
        
        # 基于所有工具结果生成最终回答
        conversation = [complex_query, response] + tool_messages
        final_response = model.invoke(conversation)
        pretty_print_ai_response(final_response)


def mixed_message_scenarios():
    """混合消息类型场景示例"""
    print("\n=== 混合消息类型场景示例 ===")
    
    model = init_model(model_name)
    
    # 场景1：角色扮演游戏
    print("场景1：角色扮演游戏")
    rpg_scenario = [
        SystemMessage("""
        游戏设定：你是一位中世纪的骑士导师
        规则：
        1. 使用中世纪风格的语言
        2. 回答要体现骑士精神
        3. 可以适当加入历史典故
        """),
        HumanMessage("尊敬的导师，我想学习剑术"),
        AIMessage("年轻的勇士，剑术乃骑士之魂！让我先考验你的决心..."),
        HumanMessage("我愿意接受任何考验！"),
    ]
    
    response = model.invoke(rpg_scenario)
    pretty_print_ai_response(response)
    
    # 场景2：技术支持对话
    print("\n场景2：技术支持对话")
    support_scenario = [
        SystemMessage("你是一位专业的Python技术支持工程师，善于解决各种编程问题"),
        HumanMessage("我的Python程序运行时报错：ImportError: No module named 'requests'"),
        AIMessage("这个错误表明您的环境中缺少requests库。您可以通过以下方式解决：\n1. 运行命令：pip install requests\n2. 如果使用conda：conda install requests"),
        HumanMessage("安装后还是报同样的错误怎么办？"),
    ]
    
    response = model.invoke(support_scenario)
    pretty_print_ai_response(response)
    
    # 场景3：创意写作助手
    print("\n场景3：创意写作助手")
    writing_scenario = [
        SystemMessage("你是一位经验丰富的科幻小说作家，善于构建复杂的世界观"),
        HumanMessage("我想写一个关于时间旅行的故事，能给我一些灵感吗？"),
        AIMessage("""
        时间旅行题材的经典元素：
        
        🌟 核心冲突：
        - 因果悖论：改变过去影响现在
        - 时间线分裂：创造平行宇宙
        - 记忆错位：穿越者保留记忆
        
        🎭 角色设定：
        - 时间守护者：维护时间秩序
        - 历史观察者：只能观看不能干预
        - 时间罪犯：恶意篡改历史
        
        🌌 世界观构建：
        - 时间管理局：监管所有时间活动
        - 时间风暴：时空不稳定现象
        - 记忆锚点：固定历史关键节点
        """),
        HumanMessage("听起来很棒！那主角应该是什么身份比较好？")
    ]
    
    response = model.invoke(writing_scenario)
    pretty_print_ai_response(response)


def advanced_message_patterns():
    """高级消息模式示例"""
    print("\n=== 高级消息模式示例 ===")
    
    model = init_model(model_name)
    
    # 模式1：思维链推理 (Chain-of-Thought)
    print("模式1：思维链推理")
    cot_pattern = [
        SystemMessage("你是一个逻辑推理专家，请分步骤思考问题"),
        HumanMessage("如果A>B，B>C，C>D，那么A和D的关系是什么？"),
        AIMessage("""
        让我逐步分析这个逻辑关系：
        
        第一步：分析已知条件
        - A > B （A大于B）
        - B > C （B大于C）
        - C > D （C大于D）
        
        第二步：建立传递关系
        由于大于关系具有传递性：
        - 从A>B和B>C可以推出A>C
        - 从A>C和C>D可以推出A>D
        
        第三步：得出结论
        因此，A > D
        
        答案：A大于D
        """),
        HumanMessage("很好！那如果是A≥B，B≥C，C≥D呢？")
    ]
    
    response = model.invoke(cot_pattern)
    pretty_print_ai_response(response)
    
    # 模式2：自我反思机制
    print("\n模式2：自我反思机制")
    reflection_pattern = [
        SystemMessage("你是一个善于自我反思的AI助手，请在回答后评估自己的回答质量"),
        HumanMessage("请解释什么是递归函数"),
        AIMessage("""
        递归函数是在函数定义中调用函数自身的编程技术。
        
        特点：
        1. 必须有基础情况（终止条件）
        2. 必须有递归情况（调用自身）
        3. 每次递归都应该向基础情况靠近
        
        示例：
        ```python
        def factorial(n):
            if n <= 1:  # 基础情况
                return 1
            else:       # 递归情况
                return n * factorial(n-1)
        ```
        
        自我评估：
        ✓ 准确解释了递归的概念
        ✓ 提供了关键特点
        ✓ 给出了具体代码示例
        ⚠ 可以补充更多实际应用场景
        """),
        HumanMessage("你的解释还可以更详细一些")
    ]
    
    response = model.invoke(reflection_pattern)
    pretty_print_ai_response(response)


if __name__ == "__main__":
    print("📚 Message全面使用示例")
    print("=" * 50)
    
    try:
        # 运行各个示例
        system_message_examples()
        human_message_examples()
        ai_message_examples()
        tool_message_examples()
        mixed_message_scenarios()
        advanced_message_patterns()
        
        print("\n✅ 所有Message使用示例演示完毕！")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查配置文件和网络连接")
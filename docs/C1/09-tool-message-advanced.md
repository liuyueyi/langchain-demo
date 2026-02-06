# LangChain实战开发教程（九）：ToolMessage与工具调用高级实战

> **解决{具体问题}的{数字}种技术方案对比**：掌握LangChain工具调用机制，让AI具备真实世界操作能力

## 🎯 本文目标

深入解析ToolMessage和工具调用的高级应用，掌握AI如何识别、调用外部工具并处理执行结果的完整流程，构建具备真实操作能力的智能应用。

## 📚 核心知识点概览

通过本文你将掌握：
- **ToolMessage核心机制**：工具执行结果的标准化处理
- **工具注册与绑定**：如何将自定义函数注册为AI可用工具
- **参数解析与验证**：AI如何理解和调用工具参数
- **多工具协调使用**：复杂场景下多个工具的协同工作
- **错误处理与重试**：工具调用失败时的优雅处理

## 🔧 ToolMessage核心技术解析

### 什么是ToolMessage？

ToolMessage是LangChain中专门用于处理工具执行结果的消息类型。当AI决定调用工具后，工具的执行结果需要通过ToolMessage传递回AI，以便生成最终的用户响应。

### 核心工作流程

```
用户提问 → AI分析 → 识别工具需求 → 调用工具 → ToolMessage封装结果 → AI整合回答
```

### 关键组件说明

1. **tool_calls属性**：AI识别出需要调用的工具列表
2. **ToolMessage对象**：封装工具执行结果
3. **tool_call_id**：关联工具调用和结果的唯一标识
4. **结果整合**：AI基于工具结果生成最终回答

## 🚀 工具调用完整实现

### 1. 基础工具定义与注册

```python
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 定义基础工具
@tool
def get_weather(city: str) -> str:
    """
    获取城市天气信息
    :param city: 城市名称
    :return: 天气信息字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，气温-2°C到8°C，西北风3-4级",
        "上海": "多云，气温3°C到12°C，东南风2-3级", 
        "广州": "小雨，气温15°C到22°C，微风",
        "深圳": "晴天，气温18°C到25°C，南风3级"
    }
    return weather_data.get(city, f"暂无{city}的天气信息")

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式
    :param expression: 数学表达式字符串
    :return: 计算结果
    """
    try:
        # 安全计算（生产环境应使用更安全的方式）
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {expression} - {str(e)}"

@tool
def get_stock_price(symbol: str) -> str:
    """
    获取股票价格信息
    :param symbol: 股票代码
    :return: 股票价格信息
    """
    # 模拟股票数据
    stock_data = {
        "AAPL": "苹果公司(AAPL)当前价格: $185.23 (+1.2%)",
        "GOOGL": "谷歌(GOOGL)当前价格: $2,847.50 (-0.8%)",
        "TSLA": "特斯拉(TSLA)当前价格: $248.75 (+3.5%)",
        "MSFT": "微软(MSFT)当前价格: $378.85 (+0.5%)"
    }
    return stock_data.get(symbol.upper(), f"未找到股票代码: {symbol}")

def basic_tool_calling():
    """基础工具调用示例"""
    
    model = init_chat_model(model="Qwen/Qwen3-8B")
    
    # 绑定工具到模型
    model_with_tools = model.bind_tools([get_weather, calculate, get_stock_price])
    
    # 用户询问天气
    user_query = HumanMessage("北京今天天气怎么样？")
    
    print("=== 基础工具调用流程 ===")
    
    # 第一步：AI分析是否需要调用工具
    print("第一步：AI分析工具需求")
    response = model_with_tools.invoke([user_query])
    
    # 检查是否有工具调用需求
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print("发现工具调用需求：")
        for i, tool_call in enumerate(response.tool_calls):
            print(f"  工具{i+1}: {tool_call['name']}")
            print(f"  参数: {tool_call['args']}")
            
            # 执行工具调用
            tool_result = None
            if tool_call['name'] == 'get_weather':
                tool_result = get_weather.invoke(tool_call['args'])
            elif tool_call['name'] == 'calculate':
                tool_result = calculate.invoke(tool_call['args'])
            elif tool_call['name'] == 'get_stock_price':
                tool_result = get_stock_price.invoke(tool_call['args'])
            
            if tool_result:
                print(f"  执行结果: {tool_result}")
                
                # 创建ToolMessage
                tool_message = ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call.get('id', f'tool_{i+1}')
                )
                
                # 第二步：AI基于工具结果生成最终回答
                print("\n第二步：AI生成最终回答")
                final_response = model.invoke([user_query, response, tool_message])
                print("最终回答：", final_response.content)
    else:
        print("无需工具调用，直接回答：", response.content)
```

### 2. 多工具协调调用

```python
def multi_tool_coordination():
    """多工具协调调用示例"""
    
    model = init_chat_model(model="Qwen/Qwen3-8B")
    model_with_tools = model.bind_tools([get_weather, calculate, get_stock_price])
    
    # 复杂查询：需要多个工具
    complex_query = HumanMessage("帮我计算一下(25+15)*2，然后查询上海的天气，最后看看AAPL的股价")
    
    print("=== 多工具协调调用 ===")
    print("用户查询：", complex_query.content)
    
    # AI分析复杂请求
    print("\nAI分析工具需求...")
    response = model_with_tools.invoke([complex_query])
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_messages = []
        
        print("识别到的工具调用：")
        for i, tool_call in enumerate(response.tool_calls):
            print(f"  {i+1}. {tool_call['name']} - {tool_call['args']}")
            
            # 执行对应工具
            tool_result = None
            if tool_call['name'] == 'calculate':
                tool_result = calculate.invoke(tool_call['args'])
            elif tool_call['name'] == 'get_weather':
                tool_result = get_weather.invoke(tool_call['args'])
            elif tool_call['name'] == 'get_stock_price':
                tool_result = get_stock_price.invoke(tool_call['args'])
            
            if tool_result:
                print(f"     执行结果: {tool_result}")
                
                # 创建ToolMessage
                tool_msg = ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call.get('id', f'tool_{i+1}')
                )
                tool_messages.append(tool_msg)
        
        # 基于所有工具结果生成最终回答
        print("\nAI整合所有工具结果生成最终回答...")
        conversation = [complex_query, response] + tool_messages
        final_response = model.invoke(conversation)
        print("最终整合回答：", final_response.content)
```

### 3. 工具调用错误处理

```python
def tool_error_handling():
    """工具调用错误处理示例"""
    
    @tool
    def risky_operation(operation: str, value: str) -> str:
        """
        模拟可能出错的操作
        :param operation: 操作类型
        :param value: 操作值
        :return: 操作结果
        """
        if operation == "divide" and value == "0":
            raise ValueError("除零错误")
        elif operation == "sqrt" and float(value) < 0:
            raise ValueError("负数不能开平方根")
        elif operation == "unknown":
            raise NotImplementedError("未知操作")
        else:
            return f"操作 {operation}({value}) 执行成功"
    
    model = init_chat_model(model="Qwen/Qwen3-8B")
    model_with_tools = model.bind_tools([risky_operation, calculate])
    
    # 测试各种错误情况
    test_cases = [
        HumanMessage("计算 10/0"),
        HumanMessage("计算 sqrt(-4)"),
        HumanMessage("执行 unknown 操作")
    ]
    
    print("=== 工具调用错误处理测试 ===")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_case.content}")
        
        try:
            # AI分析
            response = model_with_tools.invoke([test_case])
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                print(f"识别工具: {tool_call['name']}")
                
                try:
                    # 执行工具（可能出错）
                    if tool_call['name'] == 'risky_operation':
                        tool_result = risky_operation.invoke(tool_call['args'])
                    else:
                        tool_result = calculate.invoke(tool_call['args'])
                    
                    # 成功执行
                    tool_message = ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call.get('id', 'tool_1')
                    )
                    final_response = model.invoke([test_case, response, tool_message])
                    print("✅ 执行成功：", final_response.content)
                    
                except Exception as tool_error:
                    # 工具执行失败
                    error_message = ToolMessage(
                        content=f"工具执行失败: {str(tool_error)}",
                        tool_call_id=tool_call.get('id', 'tool_1')
                    )
                    print(f"❌ 工具执行失败: {tool_error}")
                    
                    # 让AI处理错误并生成用户友好的回复
                    error_response = model.invoke([test_case, response, error_message])
                    print("🔄 错误处理回复：", error_response.content)
                    
        except Exception as e:
            print(f"❌ AI分析阶段出错: {e}")
```

## 🎯 高级应用场景

### 场景1：智能助手综合服务

```python
def intelligent_assistant_scenario():
    """智能助手综合服务场景"""
    
    # 扩展工具集
    @tool
    def get_news(category: str) -> str:
        """获取新闻资讯"""
        news_data = {
            "科技": "【科技新闻】苹果发布新款iPhone 15，搭载A17芯片...",
            "财经": "【财经新闻】美联储维持利率不变，美股收高...",
            "体育": "【体育新闻】湖人队战胜勇士队，詹姆斯表现抢眼..."
        }
        return news_data.get(category, f"暂无{category}类新闻")
    
    @tool
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        """货币转换"""
        # 简化汇率（实际应调用实时汇率API）
        rates = {
            "USD": 1.0, "CNY": 7.2, "EUR": 0.93, "JPY": 149.0
        }
        if from_currency in rates and to_currency in rates:
            converted = amount * rates[to_currency] / rates[from_currency]
            return f"{amount} {from_currency} = {converted:.2f} {to_currency}"
        return "不支持的货币转换"

    @tool
    def search_web(query: str) -> str:
        """网络搜索"""
        return f"搜索结果：关于'{query}'的最新信息显示..."

    # 绑定所有工具
    model = init_chat_model(model="Qwen/Qwen3-8B")
    all_tools = [get_weather, calculate, get_stock_price, get_news, convert_currency, search_web]
    model_with_tools = model.bind_tools(all_tools)
    
    print("=== 智能助手综合服务 ===")
    
    # 复杂多步骤请求
    user_request = HumanMessage("""
    我的计划：
    1. 查看今天北京的天气
    2. 计算100美元兑换成人民币
    3. 查看AAPL的当前股价
    4. 了解最新的科技新闻
    """)
    
    print("用户请求：", user_request.content)
    
    # 处理复杂请求
    response = model_with_tools.invoke([user_request])
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\n识别到 {len(response.tool_calls)} 个工具调用需求")
        
        tool_messages = []
        for i, tool_call in enumerate(response.tool_calls):
            print(f"\n执行工具 {i+1}: {tool_call['name']}")
            print(f"参数: {tool_call['args']}")
            
            # 执行对应工具
            try:
                tool_func = next((t for t in all_tools if t.name == tool_call['name']), None)
                if tool_func:
                    result = tool_func.invoke(tool_call['args'])
                    print(f"执行结果: {result}")
                    
                    tool_msg = ToolMessage(
                        content=result,
                        tool_call_id=tool_call.get('id', f'tool_{i+1}')
                    )
                    tool_messages.append(tool_msg)
            except Exception as e:
                print(f"执行失败: {e}")
                error_msg = ToolMessage(
                    content=f"执行失败: {str(e)}",
                    tool_call_id=tool_call.get('id', f'tool_{i+1}')
                )
                tool_messages.append(error_msg)
        
        # 生成综合回答
        print("\nAI生成综合服务报告...")
        final_conversation = [user_request, response] + tool_messages
        final_response = model.invoke(final_conversation)
        print("智能助手回复：", final_response.content)
```

### 场景2：数据分析助手

```python
def data_analysis_scenario():
    """数据分析助手场景"""
    
    import json
    from datetime import datetime
    
    @tool
    def analyze_data(data_json: str) -> str:
        """数据分析工具"""
        try:
            data = json.loads(data_json)
            if isinstance(data, list):
                total = len(data)
                if total > 0 and isinstance(data[0], dict):
                    # 简单统计分析
                    keys = list(data[0].keys())
                    analysis = f"数据分析结果：\n"
                    analysis += f"- 数据条目数: {total}\n"
                    analysis += f"- 数据字段: {', '.join(keys)}\n"
                    analysis += f"- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    return analysis
                else:
                    return f"数据格式不正确，共{total}条记录"
            else:
                return "请提供JSON数组格式的数据"
        except json.JSONDecodeError as e:
            return f"JSON解析错误: {str(e)}"

    @tool
    def generate_report(title: str, data_summary: str) -> str:
        """生成分析报告"""
        report = f"""
报告标题: {title}
生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}

{data_summary}

总结:
本报告基于提供的数据分析生成，涵盖了主要的数据特征和趋势信息。
建议结合具体业务场景进行深入分析。

报告生成完毕。
        """
        return report.strip()

    # 示例数据
    sample_data = [
        {"name": "张三", "age": 25, "city": "北京", "salary": 8000},
        {"name": "李四", "age": 30, "city": "上海", "salary": 12000},
        {"name": "王五", "age": 28, "city": "广州", "salary": 9500}
    ]
    
    model = init_chat_model(model="Qwen/Qwen3-8B")
    analysis_tools = [analyze_data, generate_report, calculate]
    model_with_tools = model.bind_tools(analysis_tools)
    
    print("=== 数据分析助手 ===")
    
    # 数据分析请求
    analysis_request = HumanMessage(f"""
    请帮我分析以下员工数据：
    {json.dumps(sample_data, ensure_ascii=False)}
    
    要求：
    1. 先对数据进行统计分析
    2. 计算平均薪资
    3. 生成分析报告
    """)
    
    print("分析请求：", analysis_request.content)
    
    # 处理分析请求
    response = model_with_tools.invoke([analysis_request])
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_messages = []
        
        print(f"\n需要执行 {len(response.tool_calls)} 个分析步骤：")
        for i, tool_call in enumerate(response.tool_calls):
            print(f"  步骤 {i+1}: {tool_call['name']}")
            
            try:
                if tool_call['name'] == 'analyze_data':
                    result = analyze_data.invoke(tool_call['args'])
                elif tool_call['name'] == 'calculate':
                    # 计算平均薪资
                    total_salary = sum(emp['salary'] for emp in sample_data)
                    avg_salary = total_salary / len(sample_data)
                    result = f"平均薪资本月为: {avg_salary:.2f} 元"
                elif tool_call['name'] == 'generate_report':
                    data_summary = tool_messages[0].content if tool_messages else "数据分析完成"
                    result = generate_report.invoke({
                        "title": "员工数据分析报告",
                        "data_summary": data_summary + f"\n- 平均薪资本月：{sum(emp['salary'] for emp in sample_data) / len(sample_data):.2f} 元"
                    })
                else:
                    continue
                    
                print(f"  执行结果: {result}")
                tool_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call.get('id', f'tool_{i+1}')
                ))
                
            except Exception as e:
                print(f"  执行失败: {e}")
                tool_messages.append(ToolMessage(
                    content=f"执行失败: {str(e)}",
                    tool_call_id=tool_call.get('id', f'tool_{i+1}')
                ))
        
        # 生成最终分析报告
        print("\n生成最终分析报告...")
        final_conversation = [analysis_request, response] + tool_messages
        final_response = model.invoke(final_conversation)
        print("数据分析报告：", final_response.content)
```

## ⚡ 性能优化策略

### 1. 工具调用并行处理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def parallel_tool_execution():
    """并行工具执行优化"""
    
    def execute_tool_parallel(tool_call):
        """并行执行单个工具"""
        try:
            if tool_call['name'] == 'get_weather':
                return get_weather.invoke(tool_call['args'])
            elif tool_call['name'] == 'calculate':
                return calculate.invoke(tool_call['args'])
            elif tool_call['name'] == 'get_stock_price':
                return get_stock_price.invoke(tool_call['args'])
        except Exception as e:
            return f"执行失败: {str(e)}"
    
    model = init_chat_model(model="Qwen/Qwen3-8B")
    model_with_tools = model.bind_tools([get_weather, calculate, get_stock_price])
    
    # 多个独立查询
    queries = [
        HumanMessage("北京天气如何？"),
        HumanMessage("计算 25*4"),
        HumanMessage("AAPL股价多少？")
    ]
    
    print("=== 并行工具执行 ===")
    
    # 并行处理多个查询
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for query in queries:
            future = executor.submit(process_single_query, model_with_tools, query)
            futures.append(future)
        
        # 收集结果
        results = [future.result() for future in futures]
        
    for i, (query, result) in enumerate(zip(queries, results)):
        print(f"\n查询 {i+1}: {query.content}")
        print(f"结果: {result}")

def process_single_query(model_with_tools, query):
    """处理单个查询"""
    try:
        response = model_with_tools.invoke([query])
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_result = execute_tool_parallel(tool_call)
            tool_message = ToolMessage(content=tool_result, tool_call_id=tool_call.get('id', 'tool_1'))
            final_response = model_with_tools.invoke([query, response, tool_message])
            return final_response.content
        else:
            return response.content
    except Exception as e:
        return f"处理失败: {str(e)}"
```

### 2. 工具缓存机制

```python
class ToolCache:
    """工具结果缓存"""
    
    def __init__(self, cache_duration=300):  # 5分钟缓存
        self.cache = {}
        self.cache_duration = cache_duration
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def get_cached_result(self, tool_name, args):
        """获取缓存结果"""
        import time
        cache_key = f"{tool_name}_{str(sorted(args.items()))}"
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                print(f"🔧 使用{tool_name}缓存结果")
                return result
            else:
                # 缓存过期
                del self.cache[cache_key]
        
        return None
    
    def cache_result(self, tool_name, args, result):
        """缓存工具结果"""
        import time
        cache_key = f"{tool_name}_{str(sorted(args.items()))}"
        self.cache[cache_key] = (result, time.time())
    
    def execute_with_cache(self, tool_func, args):
        """带缓存的工具执行"""
        # 检查缓存
        cached_result = self.get_cached_result(tool_func.name, args)
        if cached_result is not None:
            return cached_result
        
        # 执行工具
        try:
            result = tool_func.invoke(args)
            # 缓存结果
            self.cache_result(tool_func.name, args, result)
            return result
        except Exception as e:
            return f"执行失败: {str(e)}"

# 使用示例
def cached_tool_demo():
    cache = ToolCache()
    
    # 多次查询相同内容
    queries = [
        HumanMessage("北京天气如何？"),
        HumanMessage("再查一下北京天气"),  # 应该使用缓存
        HumanMessage("北京天气怎么样？"),   # 应该使用缓存
    ]
    
    model = init_chat_model(model="Qwen/Qwen3-8B")
    model_with_tools = model.bind_tools([get_weather])
    
    print("=== 工具缓存演示 ===")
    for i, query in enumerate(queries, 1):
        print(f"\n查询 {i}: {query.content}")
        response = model_with_tools.invoke([query])
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            # 使用缓存执行
            result = cache.execute_with_cache(get_weather, tool_call['args'])
            print(f"结果: {result}")
```

## 🛡️ 安全与最佳实践

### 1. 工具权限控制

```python
class ToolPermissionManager:
    """工具权限管理器"""
    
    def __init__(self):
        self.permissions = {
            'get_weather': ['public'],
            'calculate': ['public'],
            'get_stock_price': ['finance_user'],
            'risky_operation': ['admin']
        }
    
    def check_permission(self, tool_name, user_role):
        """检查工具调用权限"""
        if tool_name not in self.permissions:
            return False
        
        required_roles = self.permissions[tool_name]
        return user_role in required_roles or 'public' in required_roles

def permission_controlled_tool_calling():
    """权限控制的工具调用"""
    
    permission_manager = ToolPermissionManager()
    
    # 模拟不同用户角色
    users = [
        {'name': '普通用户', 'role': 'public'},
        {'name': '金融用户', 'role': 'finance_user'},
        {'name': '管理员', 'role': 'admin'}
    ]
    
    model = init_chat_model(model="Qwen/Qwen3-8B")
    
    for user in users:
        print(f"\n=== {user['name']} ({user['role']}) 的工具调用测试 ===")
        
        # 根据权限绑定工具
        available_tools = []
        for tool in [get_weather, calculate, get_stock_price, risky_operation]:
            if permission_manager.check_permission(tool.name, user['role']):
                available_tools.append(tool)
        
        if available_tools:
            model_with_tools = model.bind_tools(available_tools)
            test_query = HumanMessage("请执行各种操作")
            response = model_with_tools.invoke([test_query])
            
            print(f"可用工具: {[t.name for t in available_tools]}")
            if hasattr(response, 'tool_calls'):
                print(f"可调用工具数: {len(response.tool_calls)}")
                for tool_call in response.tool_calls:
                    print(f"  - {tool_call['name']}")
        else:
            print("无可用工具")
```

### 2. 输入验证与清理

```python
def validate_tool_inputs():
    """工具输入验证"""
    
    import re
    from typing import Dict, Any
    
    def sanitize_input(args: Dict[str, Any]) -> Dict[str, Any]:
        """输入清理和验证"""
        sanitized = {}
        
        for key, value in args.items():
            if key == 'city':
                # 城市名验证
                if isinstance(value, str) and re.match(r'^[\u4e00-\u9fffA-Za-z\s]+$', value):
                    sanitized[key] = value.strip()
                else:
                    raise ValueError("城市名格式不正确")
            
            elif key == 'expression':
                # 数学表达式验证
                if isinstance(value, str) and re.match(r'^[0-9+\-*/().\s]+$', value):
                    sanitized[key] = value
                else:
                    raise ValueError("表达式包含非法字符")
            
            elif key == 'symbol':
                # 股票代码验证
                if isinstance(value, str) and re.match(r'^[A-Z]{1,5}$', value.upper()):
                    sanitized[key] = value.upper()
                else:
                    raise ValueError("股票代码格式不正确")
            
            else:
                sanitized[key] = value
        
        return sanitized
    
    @tool
    def safe_weather_query(city: str) -> str:
        """安全的天气查询工具"""
        try:
            # 输入验证
            validated_args = sanitize_input({'city': city})
            return get_weather.invoke(validated_args)
        except ValueError as e:
            return f"输入验证失败: {str(e)}"
    
    # 测试验证功能
    test_cases = [
        "北京",           # 正常
        "<script>alert('xss')</script>北京",  # 包含恶意代码
        "New York",      # 英文城市名
        "123invalid"     # 无效输入
    ]
    
    print("=== 输入验证测试 ===")
    for city in test_cases:
        try:
            result = safe_weather_query.invoke({'city': city})
            print(f"查询 '{city}': {result}")
        except Exception as e:
            print(f"查询 '{city}' 失败: {e}")
```

## 📝 总结

ToolMessage和工具调用是LangChain实现智能应用的核心能力：

✅ **工具识别**：AI能够智能识别何时需要调用外部工具  
✅ **参数解析**：自动提取和验证工具调用参数  
✅ **结果处理**：ToolMessage标准化处理工具执行结果  
✅ **多工具协调**：复杂场景下多个工具的协同工作  
✅ **错误处理**：完善的异常处理和用户友好反馈  

## 🔗 相关资源

- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/tools/)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Tool Calling Best Practices](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)

---
*本教程深入解析了工具调用的高级应用。结合前几期的消息系统和提示词工程，您已经掌握了LangChain的核心开发技能。*
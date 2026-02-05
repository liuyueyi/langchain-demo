# LangChain实战开发教程（五）：工具调用赋能AI

> **效率暴涨10倍**：掌握LangChain工具调用，让AI具备真实世界操作能力

## 🎯 本文目标

深入解析LangChain工具调用机制，学会为AI赋予外部工具使用能力，实现真正的智能助手功能。

## 📚 核心知识点概览

通过本文你将掌握：
- **工具注册与绑定**：如何将自定义函数注册为AI可用工具
- **参数解析机制**：AI如何理解和调用工具参数
- **结果回传处理**：工具执行结果如何反馈给AI
- **多工具协调**：复杂场景下多个工具的协同使用

## 🔧 工具调用核心技术解析

### 什么是工具调用？

工具调用是指让AI模型能够识别何时需要使用外部工具，并正确调用这些工具获取所需信息或执行操作的能力。

### 核心工作流程

```
用户提问 → AI分析 → 识别需要工具 → 调用工具 → 获取结果 → 整合回答
```

### 关键组件说明

1. **@tool装饰器**：标记可被AI调用的函数
2. **参数解析**：AI自动提取和验证函数参数
3. **执行引擎**：实际调用工具函数
4. **结果整合**：将工具结果融入最终回答

## 🚀 核心实现详解

### 1. 基础工具定义

```python
from langchain_core.tools import tool
import datetime

@tool
def get_current_time(location: str):
    """
    获取指定地区的当前时间
    :param location: 地区名称，如 'Asia/Shanghai', 'America/New_York'
    :return: 格式化的当前时间字符串
    """
    try:
        # 处理不同的时区输入
        timezone_map = {
            '北京': 'Asia/Shanghai',
            '上海': 'Asia/Shanghai', 
            '纽约': 'America/New_York',
            '伦敦': 'Europe/London',
            '东京': 'Asia/Tokyo'
        }
        
        tz_name = timezone_map.get(location, location)
        tz = datetime.timezone.utc if tz_name.lower() == 'utc' else datetime.datetime.now(
            datetime.timezone.utc).astimezone().tzinfo
            
        current_time = datetime.datetime.now(tz)
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        return f"{location}当前时间是 {formatted_time}"
        
    except Exception as e:
        return f"无法获取{location}的时间: {str(e)}"
```

### 2. 工具绑定与调用

```python
def tool_calling_workflow(model_name):
    """完整的工具调用工作流程"""
    
    # 1. 初始化模型
    model = init_model(model_name)
    
    # 2. 绑定工具到模型
    model_with_tools = model.bind_tools([get_current_time])
    
    # 3. 构造对话历史
    messages = [HumanMessage("现在北京几点了？")]
    
    # 4. 第一阶段：AI决定是否需要调用工具
    print("🤖 AI正在分析是否需要调用工具...")
    response = model_with_tools.invoke(messages)
    
    # 5. 检查是否有工具调用
    if response.tool_calls:
        print("🔧 发现工具调用需求:")
        for tool_call in response.tool_calls:
            print(f"   工具名称: {tool_call['name']}")
            print(f"   参数: {tool_call['args']}")
            
            # 6. 执行工具调用
            if tool_call['name'] == 'get_current_time':
                tool_result = get_current_time.invoke(tool_call)
                print(f"   执行结果: {tool_result}")
                
                # 7. 将结果添加到对话历史
                messages.append(tool_result)
    
    # 8. 第二阶段：AI基于工具结果生成最终回答
    print("🤖 AI正在生成最终回答...")
    final_response = model.invoke(messages)
    pretty_print_ai_response(final_response)
```

## 💡 高级工具设计模式

### 1. 参数验证与默认值

```python
@tool
def search_weather(city: str, days: int = 1):
    """
    查询城市天气预报
    :param city: 城市名称
    :param days: 查询天数，默认1天
    :return: 天气预报信息
    """
    # 参数验证
    if not city:
        return "错误：城市名称不能为空"
    
    if not isinstance(days, int) or days < 1 or days > 7:
        return "错误：天数必须是1-7之间的整数"
    
    # 模拟天气查询
    weather_data = {
        "北京": ["晴天 25°C", "多云 22°C", "小雨 18°C"],
        "上海": ["阴天 20°C", "晴天 23°C", "雷阵雨 19°C"],
        "广州": ["炎热 32°C", "多云 30°C", "台风预警"]
    }
    
    forecasts = weather_data.get(city, ["暂无该城市天气数据"])
    result = f"{city}未来{min(days, len(forecasts))}天天气预报：\n"
    
    for i in range(min(days, len(forecasts))):
        result += f"第{i+1}天: {forecasts[i]}\n"
    
    return result
```

### 2. 异步工具支持

```python
import asyncio
import aiohttp

@tool
async def async_web_search(query: str, max_results: int = 5):
    """
    异步网络搜索工具
    :param query: 搜索关键词
    :param max_results: 最大结果数
    :return: 搜索结果列表
    """
    try:
        # 模拟异步API调用
        async with aiohttp.ClientSession() as session:
            # 这里应该是真实的搜索引擎API调用
            await asyncio.sleep(1)  # 模拟网络延迟
            
            # 模拟搜索结果
            mock_results = [
                f"结果{i+1}: 关于'{query}'的相关信息...",
                f"结果{i+2}: '{query}'的详细解释...",
                f"结果{i+3}: '{query}'的应用场景..."
            ][:max_results]
            
            return "\n".join(mock_results)
            
    except Exception as e:
        return f"搜索失败: {str(e)}"

# 同步包装器
def web_search(query: str, max_results: int = 5):
    """同步版本的网络搜索工具"""
    return asyncio.run(async_web_search(query, max_results))
```

### 3. 工具组合与链式调用

```python
class ToolOrchestrator:
    def __init__(self, model):
        self.model = init_model(model)
        self.available_tools = {}
        self.execution_history = []
    
    def register_tool(self, tool_func, name=None):
        """注册工具"""
        tool_name = name or tool_func.__name__
        self.available_tools[tool_name] = tool_func
        print(f"✅ 工具 '{tool_name}' 注册成功")
    
    def bind_all_tools(self):
        """绑定所有已注册的工具"""
        return self.model.bind_tools(list(self.available_tools.values()))
    
    def execute_tool_chain(self, user_query):
        """执行工具链式调用"""
        print(f"🎯 处理用户请求: {user_query}")
        
        # 绑定工具
        model_with_tools = self.bind_all_tools()
        messages = [HumanMessage(user_query)]
        
        max_iterations = 3  # 防止无限循环
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"🔄 第{iteration}轮处理...")
            
            # AI决策
            response = model_with_tools.invoke(messages)
            
            # 检查工具调用
            if response.tool_calls:
                print(f"🔧 发现 {len(response.tool_calls)} 个工具调用")
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    print(f"   执行工具: {tool_name}({tool_args})")
                    
                    # 执行工具
                    if tool_name in self.available_tools:
                        tool_func = self.available_tools[tool_name]
                        try:
                            tool_result = tool_func.invoke(tool_call)
                            print(f"   工具结果: {tool_result}")
                            
                            # 记录执行历史
                            self.execution_history.append({
                                'tool': tool_name,
                                'args': tool_args,
                                'result': tool_result,
                                'timestamp': time.time()
                            })
                            
                            # 添加到对话历史
                            messages.append(tool_result)
                            
                        except Exception as e:
                            error_msg = f"工具 {tool_name} 执行失败: {str(e)}"
                            print(f"   ❌ {error_msg}")
                            messages.append(SystemMessage(error_msg))
                    else:
                        print(f"   ⚠️  未知工具: {tool_name}")
            else:
                # 没有更多工具调用，返回最终结果
                print("✅ 工具调用完成，生成最终回答")
                final_response = self.model.invoke(messages)
                return final_response.content
        
        return "⚠️  达到最大迭代次数，返回当前结果"

# 使用示例
orchestrator = ToolOrchestrator(model)
orchestrator.register_tool(get_current_time)
orchestrator.register_tool(search_weather)
orchestrator.register_tool(web_search)

result = orchestrator.execute_tool_chain("北京现在几点？明天天气怎么样？")
print(result)
```

## 🎯 实战应用场景

### 场景1：智能日程助理

```python
@tool
def create_calendar_event(title: str, date: str, time: str = None, duration: int = 60):
    """
    创建日历事件
    :param title: 事件标题
    :param date: 日期 (YYYY-MM-DD格式)
    :param time: 时间 (HH:MM格式，可选)
    :param duration: 持续时间(分钟)
    :return: 创建结果
    """
    try:
        # 验证日期格式
        datetime.datetime.strptime(date, '%Y-%m-%d')
        
        if time:
            datetime.datetime.strptime(time, '%H:%M')
        
        event_info = {
            'title': title,
            'date': date,
            'time': time or '全天',
            'duration': f"{duration}分钟"
        }
        
        # 模拟保存到日历
        print(f"📅 创建日历事件: {event_info}")
        
        return f"✅ 已为您创建日历事件 '{title}'，时间: {date} {time or ''}"
        
    except ValueError as e:
        return f"❌ 日期格式错误: {str(e)}"

@tool  
def check_availability(date: str, start_time: str, end_time: str):
    """
    检查时间段是否空闲
    :param date: 日期
    :param start_time: 开始时间
    :param end_time: 结束时间
    :return: 可用性检查结果
    """
    # 模拟日程检查
    busy_slots = {
        '2026-02-06': [('09:00', '10:30'), ('14:00', '15:00')],
        '2026-02-07': [('10:00', '12:00')]
    }
    
    date_slots = busy_slots.get(date, [])
    
    for busy_start, busy_end in date_slots:
        if (start_time <= busy_end and end_time >= busy_start):
            return f"❌ {date} {start_time}-{end_time} 时间段已有安排"
    
    return f"✅ {date} {start_time}-{end_time} 时间段空闲"

# 智能日程管理示例
def smart_schedule_assistant():
    schedule_model = init_model(model)
    schedule_tools = schedule_model.bind_tools([create_calendar_event, check_availability])
    
    conversation = [
        HumanMessage("我想预约下周三下午2点到4点的会议"),
        HumanMessage("帮我检查2026-02-06 14:00-16:00是否空闲"),
        HumanMessage("如果空闲的话，请创建一个'项目评审会议'的日程")
    ]
    
    for message in conversation:
        response = schedule_tools.invoke([message])
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'check_availability':
                    result = check_availability.invoke(tool_call)
                    print(f"可用性检查: {result}")
                elif tool_call['name'] == 'create_calendar_event':
                    result = create_calendar_event.invoke(tool_call)
                    print(f"日程创建: {result}")
```

### 场景2：数据分析助手

```python
import pandas as pd
import numpy as np

@tool
def load_data_sample(source: str, rows: int = 10):
    """
    加载数据样本进行分析
    :param source: 数据源标识
    :param rows: 样本行数
    :return: 数据样本摘要
    """
    # 模拟数据加载
    sample_data = pd.DataFrame({
        '用户ID': range(1, rows + 1),
        '年龄': np.random.randint(18, 65, rows),
        '消费金额': np.random.uniform(100, 5000, rows),
        '注册日期': pd.date_range('2023-01-01', periods=rows, freq='D')
    })
    
    return f"""数据样本摘要:
总行数: {len(sample_data)}
列名: {list(sample_data.columns)}
数值列统计:
{sample_data.describe().to_string()}"""

@tool
def analyze_trend(data_description: str, metric: str):
    """
    分析数据趋势
    :param data_description: 数据描述
    :param metric: 分析指标
    :return: 趋势分析结果
    """
    # 模拟趋势分析
    trends = {
        '消费金额': '呈上升趋势，月增长率约15%',
        '用户活跃度': '近期有所下降，建议加强用户召回',
        '转化率': '保持稳定在3.2%左右'
    }
    
    trend = trends.get(metric, '数据不足，无法确定明确趋势')
    return f"📊 {metric}趋势分析: {trend}"

# 数据分析工作流
def data_analysis_workflow():
    analysis_model = init_model(model)
    analysis_tools = analysis_model.bind_tools([load_data_sample, analyze_trend])
    
    queries = [
        "请分析我们的用户消费数据",
        "加载最近100条用户数据样本",
        "分析消费金额的趋势变化"
    ]
    
    context = []
    for query in queries:
        context.append(HumanMessage(query))
        response = analysis_tools.invoke(context)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'load_data_sample':
                    result = load_data_sample.invoke(tool_call)
                    print(f"📊 数据加载: {result}")
                    context.append(SystemMessage(result))
                elif tool_call['name'] == 'analyze_trend':
                    result = analyze_trend.invoke(tool_call)
                    print(f"📈 趋势分析: {result}")
                    context.append(SystemMessage(result))
```

## ⚡ 性能优化策略

### 1. 工具缓存机制

```python
from functools import lru_cache
import time

class CachedToolManager:
    def __init__(self, maxsize=128):
        self.cache = {}
        self.maxsize = maxsize
        self.stats = {'hits': 0, 'misses': 0}
    
    def cached_tool(self, func):
        """为工具函数添加缓存装饰器"""
        def wrapper(*args, **kwargs):
            # 创建缓存键
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            # 检查缓存
            if cache_key in self.cache:
                self.stats['hits'] += 1
                cached_result, timestamp = self.cache[cache_key]
                
                # 检查是否过期（5分钟）
                if time.time() - timestamp < 300:
                    print(f"💾 缓存命中: {func.__name__}")
                    return cached_result
                else:
                    # 缓存过期，删除
                    del self.cache[cache_key]
            
            self.stats['misses'] += 1
            
            # 执行实际函数
            result = func(*args, **kwargs)
            
            # 存储到缓存
            if len(self.cache) >= self.maxsize:
                # 删除最老的缓存项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = (result, time.time())
            print(f"🆕 缓存未命中，执行: {func.__name__}")
            
            return result
        
        return wrapper

# 使用示例
cache_manager = CachedToolManager()

@tool
@cache_manager.cached_tool
def get_stock_price(symbol: str):
    """获取股票价格（带缓存）"""
    # 模拟API调用延迟
    time.sleep(2)
    return f"{symbol}当前价格: ${np.random.uniform(100, 200):.2f}"

# 性能测试
def cache_performance_test():
    symbols = ['AAPL', 'GOOGL', 'MSFT'] * 3
    
    start_time = time.time()
    for symbol in symbols:
        price = get_stock_price.invoke({'symbol': symbol})
        print(price)
    end_time = time.time()
    
    print(f"\n📊 性能统计:")
    print(f"总执行时间: {end_time - start_time:.2f}秒")
    print(f"缓存命中率: {cache_manager.stats['hits']}/{cache_manager.stats['hits'] + cache_manager.stats['misses']}")
```

### 2. 并行工具执行

```python
import concurrent.futures

class ParallelToolExecutor:
    def __init__(self, max_workers=5):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.results = {}
    
    def execute_parallel(self, tool_calls):
        """并行执行多个工具调用"""
        futures = {}
        
        # 提交所有工具调用任务
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call['name']
            if tool_name == 'get_current_time':
                future = self.executor.submit(get_current_time.invoke, tool_call)
                futures[future] = f"time_query_{i}"
            elif tool_name == 'search_weather':
                future = self.executor.submit(search_weather.invoke, tool_call)
                futures[future] = f"weather_query_{i}"
        
        # 收集结果
        results = {}
        for future in concurrent.futures.as_completed(futures):
            query_id = futures[future]
            try:
                result = future.result(timeout=10)
                results[query_id] = result
                print(f"✅ {query_id} 执行完成")
            except Exception as e:
                results[query_id] = f"执行失败: {str(e)}"
                print(f"❌ {query_id} 执行失败: {e}")
        
        return results

# 使用示例
def parallel_tool_demo():
    executor = ParallelToolExecutor()
    
    # 多个并行工具调用
    parallel_calls = [
        {'name': 'get_current_time', 'args': {'location': '北京'}},
        {'name': 'get_current_time', 'args': {'location': '纽约'}},
        {'name': 'get_current_time', 'args': {'location': '伦敦'}}
    ]
    
    print("🚀 开始并行工具执行...")
    start_time = time.time()
    
    results = executor.execute_parallel(parallel_calls)
    
    end_time = time.time()
    print(f"⏱️  并行执行耗时: {end_time - start_time:.2f}秒")
    
    for query_id, result in results.items():
        print(f"{query_id}: {result}")
```

## 🛡️ 错误处理与安全

### 1. 工具调用安全控制

```python
class SecureToolManager:
    def __init__(self):
        self.allowed_tools = set()
        self.rate_limits = {}
        self.call_log = []
    
    def register_safe_tool(self, tool_func, permissions=None, rate_limit=10):
        """注册安全工具"""
        tool_name = tool_func.__name__
        
        # 权限检查
        if permissions and not self.check_permissions(permissions):
            raise PermissionError(f"权限不足，无法注册工具 {tool_name}")
        
        # 设置频率限制
        self.rate_limits[tool_name] = {
            'limit': rate_limit,
            'calls': [],
            'window': 60  # 60秒窗口
        }
        
        self.allowed_tools.add(tool_name)
        print(f"✅ 安全工具 {tool_name} 注册成功 (限制: {rate_limit}次/分钟)")
    
    def check_rate_limit(self, tool_name):
        """检查频率限制"""
        if tool_name not in self.rate_limits:
            return True
            
        limit_info = self.rate_limits[tool_name]
        current_time = time.time()
        
        # 清理过期记录
        limit_info['calls'] = [
            call_time for call_time in limit_info['calls'] 
            if current_time - call_time < limit_info['window']
        ]
        
        # 检查是否超过限制
        if len(limit_info['calls']) >= limit_info['limit']:
            return False
            
        # 记录本次调用
        limit_info['calls'].append(current_time)
        return True
    
    def secure_invoke(self, tool_call):
        """安全的工具调用"""
        tool_name = tool_call['name']
        
        # 检查工具是否被允许
        if tool_name not in self.allowed_tools:
            return f"❌ 工具 {tool_name} 未被授权使用"
        
        # 检查频率限制
        if not self.check_rate_limit(tool_name):
            return f"❌ 工具 {tool_name} 调用频率超限"
        
        # 执行工具调用
        try:
            # 这里应该调用实际的工具函数
            result = f"✅ {tool_name} 执行成功"
            
            # 记录调用日志
            self.call_log.append({
                'tool': tool_name,
                'args': tool_call.get('args', {}),
                'timestamp': time.time(),
                'result': 'success'
            })
            
            return result
            
        except Exception as e:
            error_msg = f"❌ {tool_name} 执行失败: {str(e)}"
            self.call_log.append({
                'tool': tool_name,
                'args': tool_call.get('args', {}),
                'timestamp': time.time(),
                'result': 'failed',
                'error': str(e)
            })
            return error_msg

# 使用示例
secure_manager = SecureToolManager()
secure_manager.register_safe_tool(get_current_time, rate_limit=5)

# 模拟高频调用测试
for i in range(8):
    result = secure_manager.secure_invoke({
        'name': 'get_current_time',
        'args': {'location': '北京'}
    })
    print(f"调用 {i+1}: {result}")
```

### 2. 参数验证与清理

```python
def sanitize_tool_parameters(tool_call):
    """清理和验证工具参数"""
    sanitized_args = {}
    errors = []
    
    args = tool_call.get('args', {})
    
    # 通用清理规则
    for key, value in args.items():
        if isinstance(value, str):
            # 移除危险字符
            cleaned_value = value.replace(';', '').replace('|', '').replace('&', '')
            # 限制长度
            if len(cleaned_value) > 1000:
                errors.append(f"参数 {key} 长度过长")
                continue
            sanitized_args[key] = cleaned_value
        else:
            sanitized_args[key] = value
    
    # 特定参数验证
    if 'location' in sanitized_args:
        allowed_locations = ['北京', '上海', '纽约', '伦敦', '东京']
        if sanitized_args['location'] not in allowed_locations:
            errors.append(f"不支持的位置: {sanitized_args['location']}")
    
    if 'days' in sanitized_args:
        try:
            days = int(sanitized_args['days'])
            if days < 1 or days > 7:
                errors.append("天数必须在1-7之间")
            sanitized_args['days'] = days
        except (ValueError, TypeError):
            errors.append("天数必须是数字")
    
    return sanitized_args, errors

# 使用示例
def safe_tool_execution(tool_call):
    """安全的工具执行流程"""
    print(f"📥 接收到工具调用: {tool_call}")
    
    # 参数清理和验证
    clean_args, validation_errors = sanitize_tool_parameters(tool_call)
    
    if validation_errors:
        error_msg = "参数验证失败: " + "; ".join(validation_errors)
        print(f"❌ {error_msg}")
        return error_msg
    
    # 更新工具调用参数
    safe_tool_call = tool_call.copy()
    safe_tool_call['args'] = clean_args
    
    print(f"✅ 参数验证通过: {clean_args}")
    
    # 这里执行实际的工具调用
    # result = actual_tool.invoke(safe_tool_call)
    # return result
    
    return f"✅ 工具 {tool_call['name']} 参数已清理验证"
```

## 📊 监控与分析

### 1. 工具使用统计

```python
class ToolUsageAnalytics:
    def __init__(self):
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_response_time': 0,
            'tool_popularity': {},
            'error_types': {}
        }
        self.call_history = []
    
    def record_tool_call(self, tool_name, success=True, response_time=None, error_type=None):
        """记录工具调用统计"""
        self.metrics['total_calls'] += 1
        
        if success:
            self.metrics['successful_calls'] += 1
        else:
            self.metrics['failed_calls'] += 1
            if error_type:
                self.metrics['error_types'][error_type] = \
                    self.metrics['error_types'].get(error_type, 0) + 1
        
        # 工具受欢迎程度
        self.metrics['tool_popularity'][tool_name] = \
            self.metrics['tool_popularity'].get(tool_name, 0) + 1
        
        # 响应时间统计
        if response_time:
            current_avg = self.metrics['avg_response_time']
            total_calls = self.metrics['total_calls']
            self.metrics['avg_response_time'] = \
                (current_avg * (total_calls - 1) + response_time) / total_calls
        
        # 记录调用历史
        self.call_history.append({
            'tool': tool_name,
            'success': success,
            'response_time': response_time,
            'error_type': error_type,
            'timestamp': time.time()
        })
    
    def generate_report(self):
        """生成使用报告"""
        success_rate = (self.metrics['successful_calls'] / 
                       max(self.metrics['total_calls'], 1)) * 100
        
        report = f"""
📊 工具使用分析报告
==================
总调用次数: {self.metrics['total_calls']}
成功率: {success_rate:.1f}%
平均响应时间: {self.metrics['avg_response_time']:.2f}秒

🔧 工具使用排行:
"""
        
        # 按使用频率排序
        sorted_tools = sorted(
            self.metrics['tool_popularity'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for tool, count in sorted_tools[:5]:
            percentage = (count / self.metrics['total_calls']) * 100
            report += f"  {tool}: {count}次 ({percentage:.1f}%)\n"
        
        if self.metrics['error_types']:
            report += "\n❌ 错误类型统计:\n"
            for error_type, count in self.metrics['error_types'].items():
                report += f"  {error_type}: {count}次\n"
        
        return report

# 使用示例
analytics = ToolUsageAnalytics()

# 模拟工具调用记录
test_calls = [
    ('get_current_time', True, 0.5),
    ('search_weather', True, 1.2),
    ('get_current_time', False, None, 'network_error'),
    ('web_search', True, 2.1)
]

for tool_name, success, response_time, *error in test_calls:
    analytics.record_tool_call(tool_name, success, response_time, error[0] if error else None)

print(analytics.generate_report())
```

## 🎨 高级应用示例

### 1. 自适应工具选择

```python
class AdaptiveToolSelector:
    def __init__(self, model):
        self.model = init_model(model)
        self.tools = {}
        self.performance_history = {}
    
    def register_adaptive_tool(self, tool_func, complexity_level=1):
        """注册自适应工具"""
        tool_name = tool_func.__name__
        self.tools[tool_name] = {
            'function': tool_func,
            'complexity': complexity_level,
            'performance': []  # 执行时间和成功率历史
        }
    
    def select_optimal_tool(self, query, context=None):
        """根据查询选择最优工具"""
        # 分析查询复杂度
        query_complexity = self.analyze_query_complexity(query)
        
        # 计算每个工具的适应度分数
        tool_scores = {}
        
        for tool_name, tool_info in self.tools.items():
            # 复杂度匹配度
            complexity_match = 1 - abs(tool_info['complexity'] - query_complexity) / 3
            
            # 历史性能得分
            if tool_info['performance']:
                avg_success_rate = sum(p['success'] for p in tool_info['performance']) / len(tool_info['performance'])
                avg_response_time = sum(p['time'] for p in tool_info['performance']) / len(tool_info['performance'])
                performance_score = avg_success_rate * (1 / (1 + avg_response_time))
            else:
                performance_score = 0.5  # 默认分数
            
            # 综合得分
            tool_scores[tool_name] = 0.6 * complexity_match + 0.4 * performance_score
        
        # 选择得分最高的工具
        best_tool = max(tool_scores.items(), key=lambda x: x[1])
        return best_tool[0], tool_scores
    
    def analyze_query_complexity(self, query):
        """分析查询复杂度"""
        # 基于关键词和长度的简单分析
        complex_keywords = ['详细', '全面', '深入', '比较', '分析']
        complexity_score = min(len(query) / 50, 2)  # 长度因素
        
        for keyword in complex_keywords:
            if keyword in query:
                complexity_score += 0.5
        
        return min(complexity_score, 3)  # 最高复杂度为3

# 使用示例
adaptive_selector = AdaptiveToolSelector(model)
adaptive_selector.register_adaptive_tool(get_current_time, complexity_level=1)
adaptive_selector.register_adaptive_tool(search_weather, complexity_level=2)
adaptive_selector.register_adaptive_tool(web_search, complexity_level=3)

queries = [
    "现在几点了？",
    "北京天气怎么样？",
    "详细分析人工智能发展趋势"
]

for query in queries:
    best_tool, scores = adaptive_selector.select_optimal_tool(query)
    print(f"查询: {query}")
    print(f"推荐工具: {best_tool}")
    print(f"各工具得分: {scores}")
    print("-" * 40)
```

### 2. 工具链编排系统

```python
class ToolChainOrchestrator:
    def __init__(self, model):
        self.model = init_model(model)
        self.tools = {}
        self.chain_templates = {}
    
    def define_tool_chain(self, chain_name, tool_sequence, conditions=None):
        """定义工具链模板"""
        self.chain_templates[chain_name] = {
            'sequence': tool_sequence,
            'conditions': conditions or {}
        }
    
    def execute_chain(self, chain_name, initial_params):
        """执行预定义的工具链"""
        if chain_name not in self.chain_templates:
            return f"❌ 未找到工具链: {chain_name}"
        
        chain_template = self.chain_templates[chain_name]
        results = {}
        current_params = initial_params.copy()
        
        print(f"🔗 执行工具链: {chain_name}")
        
        for step, tool_name in enumerate(chain_template['sequence']):
            print(f"  步骤 {step + 1}: 执行 {tool_name}")
            
            # 检查执行条件
            if tool_name in chain_template['conditions']:
                condition = chain_template['conditions'][tool_name]
                if not self.evaluate_condition(condition, results):
                    print(f"    ⚠️  条件不满足，跳过 {tool_name}")
                    continue
            
            # 执行工具
            if tool_name in self.tools:
                try:
                    tool_func = self.tools[tool_name]['function']
                    tool_result = tool_func.invoke(current_params)
                    results[tool_name] = tool_result
                    
                    print(f"    ✅ 执行成功: {str(tool_result)[:50]}...")
                    
                    # 更新参数供下一步使用
                    current_params.update(self.extract_params_from_result(tool_result))
                    
                except Exception as e:
                    print(f"    ❌ 执行失败: {str(e)}")
                    results[tool_name] = f"执行失败: {str(e)}"
            else:
                print(f"    ⚠️  工具未注册: {tool_name}")
        
        return results
    
    def evaluate_condition(self, condition, results):
        """评估执行条件"""
        # 简单的条件评估实现
        if isinstance(condition, dict):
            tool_name = condition.get('tool')
            expected_result = condition.get('result_contains')
            
            if tool_name and tool_name in results:
                return expected_result in str(results[tool_name])
        
        return True
    
    def extract_params_from_result(self, result):
        """从工具结果中提取参数"""
        # 简单的参数提取逻辑
        if isinstance(result, str) and ':' in result:
            parts = result.split(':', 1)
            return {parts[0].strip(): parts[1].strip()}
        return {}

# 使用示例
orchestrator = ToolChainOrchestrator(model)
orchestrator.tools['get_time'] = {'function': get_current_time}
orchestrator.tools['check_weather'] = {'function': search_weather}
orchestrator.tools['web_search'] = {'function': web_search}

# 定义天气查询工具链
orchestrator.define_tool_chain(
    'weather_inquiry',
    ['get_time', 'check_weather'],
    {
        'check_weather': {'tool': 'get_time', 'result_contains': '时间'}
    }
)

# 执行工具链
chain_result = orchestrator.execute_chain('weather_inquiry', {'location': '北京'})
print("工具链执行结果:", chain_result)
```

## 📝 总结

工具调用为LangChain应用带来了强大的扩展能力：

✅ **真实世界连接**：AI可以调用外部工具获取实时信息  
✅ **任务自动化**：复杂工作流的智能化执行  
✅ **性能优化**：并行处理和缓存机制提升效率  
✅ **安全保障**：完善的权限控制和错误处理  

## 🔗 相关资源

- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Agent Implementation Patterns](https://python.langchain.com/docs/modules/agents/)

---
*本教程深入解析了工具调用的核心机制。下一期我们将探索结构化输出的强大功能。*
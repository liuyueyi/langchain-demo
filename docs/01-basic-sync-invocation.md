# LangChain实战开发教程（一）：基础同步调用详解

> **掌握LangChain核心技能**：从基础同步调用开始你的AI应用开发之旅

## 🎯 本文目标

深入理解LangChain基础同步调用的核心机制，掌握环境配置、模型初始化、参数调优等关键技术要点。

## 📚 核心知识点概览

通过本文你将掌握：
- **环境配置管理**：安全的API密钥管理方式
- **模型初始化参数**：温度、超时、重试等关键配置
- **同步调用实现**：基础的invoke方法使用
- **输出美化技巧**：专业的响应结果显示
- **调试信息分析**：Token使用和性能监控

## 🔧 基础同步调用核心技术

### 什么是同步调用？

同步调用是指程序发送请求后等待模型完全处理完毕再返回结果的方式。这是最基础也是最重要的AI调用模式。

### 核心优势

✅ **简单直观**：代码逻辑清晰易懂  
✅ **完整结果**：一次性获取全部响应内容  
✅ **易于调试**：便于分析和测试  
✅ **稳定可靠**：成熟的调用模式

### 适用场景

- 简单的问答交互
- 内容生成任务
- 数据处理和分析
- 原型开发和测试

## 🚀 核心实现解析

### 1. 环境配置加载

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 智能路径查找配置文件
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env')
load_dotenv(config_path)

# 初始化环境变量映射
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model_name = os.getenv('MODEL')
```

**关键要点**：
- 使用相对路径确保配置文件可找到
- 通过环境变量映射实现配置标准化
- 支持多环境配置管理

### 2. 模型初始化配置

```python
from langchain.chat_models import init_chat_model

def init_model(model):
    """初始化LLM模型配置"""
    return init_chat_model(
        model=model,
        model_provider="openai",      # 指定模型提供商
        temperature=0.7,              # 温度参数：0-1，控制创造性
        timeout=30,                   # 超时时间：30秒
        max_tokens=1000,              # 最大输出token数
        max_retries=3                 # 最大重试次数
    )
```

**参数详解**：

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| temperature | 输出随机性 | 0.7 | 越高越creative |
| timeout | 等待超时 | 30秒 | 防止长时间阻塞 |
| max_tokens | 输出长度限制 | 1000 | 控制成本和响应时间 |
| max_retries | 失败重试 | 3次 | 提高调用成功率 |


这种主要是借助langchain的 `init_chat_model` 来实现model初始化，比如我们这里谁用OpenAI的接口风格，将`API_KEY/BASE_URL`维护在上面的环境上下文中，如果我的项目中，有多个OpenAI风格的模型需要呢，显然这种方式就不太合适了

此时我们可以通过在 `init_chat_model` 中传入参数，来达到不同场景的参数配置。

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="MODEL_NAME",
    model_provider="openai",
    base_url="BASE_URL",
    api_key="YOUR_API_KEY",
)
```

当然除了上面这种方式之外，我们也可以直接使用 `langchain_openai` 实现的ChatOpenAI来创建model，比如下面这个通过代理来直接访问ChatGpt的大模型，通过 `openai_proxy` 来设置代理（当然也可以直接通过api_key, base_url来指定支持OpenAI接口风格的大模型）

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4.1",
    openai_proxy="http://proxy.example.com:8080"
)
```

### 3. 同步调用核心实现

```python
def simple_invoke(model):
    """基础的同步调用实现"""
    # 初始化模型实例
    model_instance = init_model(model)
    
    # 执行同步调用
    response = model_instance.invoke("请写一首关于颜色的五言绝句")
    
    # 美化输出结果
    pretty_print_ai_response(response)
```

## 💡 关键技术要点

### 1. 响应美化输出

```python
def pretty_print_ai_response(response):
    """专业的AI响应美化输出"""
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
    
    # 技术信息分析
    print(f"\n{separator}")
    print("📊 技术详情:")
    print(f"  📁 类型: {type(response).__name__}")
    
    # Token使用情况
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
```

### 2. 参数调优策略

```python
def optimized_invoke(model, prompt, **kwargs):
    """参数优化的调用函数"""
    # 默认参数配置
    default_params = {
        'temperature': 0.7,
        'max_tokens': 1000,
        'timeout': 30
    }
    
    # 合并用户参数
    params = {**default_params, **kwargs}
    
    # 动态初始化模型
    model_instance = init_chat_model(
        model=model,
        model_provider="openai",
        **params
    )
    
    return model_instance.invoke(prompt)

# 使用示例
def demo_parameter_tuning():
    """参数调优演示"""
    prompts = ["写一首诗", "详细解释量子力学", "生成技术文档"]
    
    # 不同场景的参数配置
    configs = [
        {'temperature': 0.9, 'max_tokens': 200},    # 创意写作
        {'temperature': 0.3, 'max_tokens': 1500},   # 技术解释  
        {'temperature': 0.5, 'max_tokens': 800}     # 文档生成
    ]
    
    for prompt, config in zip(prompts, configs):
        print(f"\n📝 场景: {prompt}")
        print(f"⚙️  配置: {config}")
        response = optimized_invoke(model_name, prompt, **config)
        print(f"✅ 结果长度: {len(response.content)} 字符")
```

## 🎯 实战应用场景

### 场景1：内容创作助手

```python
def content_creation_assistant():
    """内容创作助手实现"""
    
    class ContentCreator:
        def __init__(self, model):
            self.model = init_model(model)
        
        def write_poem(self, theme, style="古典"):
            """写诗功能"""
            prompt = f"请用{style}风格写一首关于{theme}的诗"
            return self.model.invoke(prompt).content
        
        def generate_article(self, topic, length="short"):
            """生成文章"""
            length_map = {"short": 300, "medium": 800, "long": 1500}
            prompt = f"写一篇关于{topic}的文章，约{length_map[length]}字"
            return self.model.invoke(prompt).content
        
        def create_story(self, genre, characters=None):
            """创作故事"""
            char_desc = f"主角是{characters}" if characters else ""
            prompt = f"创作一个{genre}故事开头{char_desc}"
            return self.model.invoke(prompt).content
    
    # 使用示例
    creator = ContentCreator(model_name)
    
    poem = creator.write_poem("春天")
    print("🌸 春天诗歌:")
    print(poem)
    
    article = creator.generate_article("人工智能发展", "short")
    print("\n📰 AI发展短文:")
    print(article[:200] + "...")
```

### 场景2：代码生成工具

```python
def code_generation_tool():
    """代码生成工具实现"""
    
    class CodeGenerator:
        def __init__(self, model):
            self.model = init_model(model)
        
        def generate_function(self, purpose, language="Python"):
            """生成函数代码"""
            prompt = f"""用{language}写一个{purpose}的函数
要求：
1. 包含详细注释
2. 遵循PEP8规范
3. 添加错误处理"""
            
            response = self.model.invoke(prompt)
            return self.extract_code_block(response.content)
        
        def explain_code(self, code):
            """解释代码功能"""
            prompt = f"请解释以下代码的功能和实现思路：\n\n{code}"
            return self.model.invoke(prompt).content
        
        def extract_code_block(self, text):
            """提取代码块"""
            import re
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
            return code_blocks[0] if code_blocks else text
    
    # 使用示例
    coder = CodeGenerator(model_name)
    
    # 生成排序函数
    sort_function = coder.generate_function("冒泡排序算法")
    print("🐍 生成的排序函数:")
    print(sort_function)
    
    # 解释代码
    explanation = coder.explain_code(sort_function)
    print("\n📝 代码解释:")
    print(explanation[:300] + "...")
```

## ⚡ 性能优化技巧

### 1. 调用性能监控

```python
import time

def performance_monitoring_invoke(model, prompt):
    """带性能监控的调用函数"""
    start_time = time.time()
    
    try:
        model_instance = init_model(model)
        response = model_instance.invoke(prompt)
        end_time = time.time()
        
        # 性能指标计算
        execution_time = end_time - start_time
        content_length = len(response.content)
        tokens_used = getattr(response, 'usage_metadata', {}).get('total_tokens', 0)
        
        performance_data = {
            'execution_time': execution_time,
            'content_length': content_length,
            'tokens_used': tokens_used,
            'efficiency': content_length / execution_time if execution_time > 0 else 0,
            'cost_efficiency': tokens_used / content_length if content_length > 0 else 0
        }
        
        print(f"⏱️  执行时间: {execution_time:.2f}秒")
        print(f"📄 内容长度: {content_length}字符")
        print(f"💰 Token消耗: {tokens_used}")
        print(f"⚡ 效率指标: {performance_data['efficiency']:.1f}字符/秒")
        
        return response, performance_data
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        return None, {'error': str(e)}

# 性能测试
def performance_benchmark():
    """性能基准测试"""
    test_prompts = [
        "写一句话",
        "写一段话", 
        "写一篇文章"
    ]
    
    results = []
    for prompt in test_prompts:
        print(f"\n🎯 测试: {prompt}")
        response, metrics = performance_monitoring_invoke(model_name, prompt)
        if response:
            results.append(metrics)
    
    # 统计分析
    if results:
        avg_time = sum(r['execution_time'] for r in results) / len(results)
        avg_efficiency = sum(r['efficiency'] for r in results) / len(results)
        print(f"\n📈 平均性能:")
        print(f"  平均执行时间: {avg_time:.2f}秒")
        print(f"  平均效率: {avg_efficiency:.1f}字符/秒")
```

### 2. 错误处理与重试机制

```python
import random

def robust_invoke(model, prompt, max_retries=3, base_delay=1):
    """具有重试机制的稳健调用"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            print(f"📡 尝试第 {attempt + 1} 次调用...")
            
            model_instance = init_model(model)
            response = model_instance.invoke(prompt)
            
            print("✅ 调用成功!")
            return response
            
        except Exception as e:
            last_exception = e
            print(f"❌ 第 {attempt + 1} 次调用失败: {e}")
            
            if attempt < max_retries - 1:
                # 指数退避延迟
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⏳ 等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
    
    # 所有重试都失败
    print(f"💥 所有 {max_retries} 次重试都失败")
    raise last_exception

# 使用示例
def demo_robust_calling():
    """稳健调用演示"""
    try:
        response = robust_invoke(
            model_name, 
            "请写一首关于坚韧不拔精神的诗",
            max_retries=3
        )
        print("📬 最终结果:")
        print(response.content)
    except Exception as e:
        print(f"最终失败: {e}")
```

## 🛡️ 安全与最佳实践

### 1. 配置安全管理

```python
class SecureConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.required_keys = ['API_KEY', 'BASE_URL', 'MODEL']
        self.optional_keys = ['DEBUG', 'TIMEOUT']
    
    def validate_config(self):
        """验证配置完整性"""
        load_dotenv(self.config_path)
        
        missing_keys = []
        for key in self.required_keys:
            if not os.getenv(key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"缺少必要配置项: {missing_keys}")
        
        print("✅ 配置验证通过")
        return True
    
    def get_secure_config(self):
        """获取安全的配置信息"""
        self.validate_config()
        
        return {
            'api_key': os.getenv('API_KEY')[:8] + '...' if os.getenv('API_KEY') else None,
            'base_url': os.getenv('BASE_URL'),
            'model': os.getenv('MODEL'),
            'debug': os.getenv('DEBUG', 'false').lower() == 'true'
        }

# 使用示例
def secure_setup_demo():
    """安全配置演示"""
    config_manager = SecureConfigManager('config.env')
    
    try:
        secure_config = config_manager.get_secure_config()
        print("🔐 安全配置信息:")
        for key, value in secure_config.items():
            print(f"  {key}: {value}")
    except ValueError as e:
        print(f"配置错误: {e}")
```

### 2. 日志记录与监控

```python
import logging
from datetime import datetime

class InvocationLogger:
    def __init__(self, log_file="invocation.log"):
        self.logger = logging.getLogger('LangChainInvoke')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def log_invocation(self, prompt, response=None, error=None):
        """记录调用日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'prompt_length': len(prompt),
            'prompt_preview': prompt[:50] + '...' if len(prompt) > 50 else prompt
        }
        
        if response:
            log_data.update({
                'success': True,
                'response_length': len(response.content) if hasattr(response, 'content') else 0,
                'token_usage': getattr(response, 'usage_metadata', {}).get('total_tokens', 0)
            })
            self.logger.info(f"调用成功: {log_data}")
        else:
            log_data.update({
                'success': False,
                'error': str(error)
            })
            self.logger.error(f"调用失败: {log_data}")

# 使用示例
def logged_invoke(model, prompt):
    """带日志记录的调用"""
    logger = InvocationLogger()
    
    try:
        model_instance = init_model(model)
        response = model_instance.invoke(prompt)
        logger.log_invocation(prompt, response)
        return response
    except Exception as e:
        logger.log_invocation(prompt, error=e)
        raise

# 测试日志功能
def logging_demo():
    """日志记录演示"""
    response = logged_invoke(model_name, "写一句激励人心的话")
    print("💬 生成内容:")
    print(response.content)
    print("\n📝 详细日志已保存到 invocation.log 文件")
```

## 📊 调用模式对比分析

### 同步 vs 流式 vs 批量

| 特性 | 同步调用 | 流式调用 | 批量调用 |
|------|----------|----------|----------|
| 实现复杂度 | 简单 | 中等 | 复杂 |
| 用户体验 | 等待完整结果 | 实时逐字显示 | 并行处理 |
| 资源利用 | 一般 | 较好 | 最优 |
| 适用场景 | 简单交互 | 长文本生成 | 多任务处理 |
| 调试友好性 | 最好 | 中等 | 复杂 |

### 性能基准测试

```python
def comparative_analysis():
    """三种调用模式对比分析"""
    test_prompt = "请详细解释人工智能的发展历程"
    
    print("🔬 调用模式对比测试")
    print("=" * 50)
    
    # 同步调用测试
    print("\n🔄 同步调用测试:")
    start_time = time.time()
    sync_response = init_model(model_name).invoke(test_prompt)
    sync_time = time.time() - start_time
    
    print(f"执行时间: {sync_time:.2f}秒")
    print(f"内容长度: {len(sync_response.content)}字符")
    
    # 流式调用测试
    print("\n🌊 流式调用测试:")
    start_time = time.time()
    stream_content = ""
    for chunk in init_model(model_name).stream([test_prompt]):
        stream_content += chunk.content
    stream_time = time.time() - start_time
    
    print(f"执行时间: {stream_time:.2f}秒")
    print(f"内容长度: {len(stream_content)}字符")
    
    # 批量调用测试
    print("\n📦 批量调用测试:")
    start_time = time.time()
    batch_responses = init_model(model_name).batch([test_prompt])
    batch_time = time.time() - start_time
    
    print(f"执行时间: {batch_time:.2f}秒")
    print(f"内容长度: {len(batch_responses[0].content)}字符")
    
    # 性能对比总结
    print("\n📊 性能对比总结:")
    print(f"同步调用: {sync_time:.2f}秒")
    print(f"流式调用: {stream_time:.2f}秒 ({((sync_time-stream_time)/sync_time*100):+.1f}%)")
    print(f"批量调用: {batch_time:.2f}秒 ({((sync_time-batch_time)/sync_time*100):+.1f}%)")
```

## 🎨 高级应用示例

### 1. 模板化调用系统

```python
class TemplateInvokeSystem:
    def __init__(self, model):
        self.model = init_model(model)
        self.templates = {
            'poem': "请写一首关于{topic}的{style}风格{type}",
            'explanation': "请详细解释{subject}，包括{aspects}",
            'translation': "请将以下{source_lang}内容翻译成{target_lang}：{content}",
            'code_review': "请审查以下代码的质量和潜在问题：{code}"
        }
    
    def invoke_template(self, template_name, **kwargs):
        """使用模板进行调用"""
        if template_name not in self.templates:
            raise ValueError(f"未知模板: {template_name}")
        
        prompt = self.templates[template_name].format(**kwargs)
        return self.model.invoke(prompt).content
    
    def add_template(self, name, template):
        """添加新模板"""
        self.templates[name] = template

# 使用示例
def template_system_demo():
    """模板系统演示"""
    template_system = TemplateInvokeSystem(model_name)
    
    # 诗歌创作
    poem = template_system.invoke_template(
        'poem',
        topic='秋叶',
        style='古典',
        type='五言绝句'
    )
    print("🍂 秋叶诗:")
    print(poem)
    
    # 技术解释
    explanation = template_system.invoke_template(
        'explanation',
        subject='机器学习',
        aspects='基本概念、主要算法、应用场景'
    )
    print("\n🤖 机器学习解释:")
    print(explanation[:200] + "...")
```

### 2. 智能缓存系统

```python
import hashlib
from functools import lru_cache

class IntelligentCache:
    def __init__(self, maxsize=128):
        self.cache = {}
        self.maxsize = maxsize
        self.stats = {'hits': 0, 'misses': 0}
    
    def get_cache_key(self, prompt, model_config):
        """生成缓存键"""
        key_string = f"{prompt}_{str(model_config)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def cached_invoke(self, model, prompt, **model_params):
        """带缓存的智能调用"""
        # 生成缓存键
        cache_key = self.get_cache_key(prompt, model_params)
        
        # 检查缓存
        if cache_key in self.cache:
            self.stats['hits'] += 1
            print("💾 缓存命中")
            return self.cache[cache_key]
        
        self.stats['misses'] += 1
        print("🆕 缓存未命中，执行调用")
        
        # 执行实际调用
        try:
            model_instance = init_model(model)
            # 应用参数
            for param, value in model_params.items():
                setattr(model_instance, param, value)
            
            result = model_instance.invoke(prompt)
            
            # 存储到缓存
            if len(self.cache) >= self.maxsize:
                # 删除最老的缓存项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"❌ 调用失败: {e}")
            raise

# 使用示例
def caching_demo():
    """缓存系统演示"""
    cache = IntelligentCache()
    
    # 重复调用测试
    test_prompts = [
        "写一句励志的话",
        "解释人工智能概念",
        "写一句励志的话"  # 重复调用
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n📥 调用 {i+1}: {prompt}")
        result = cache.cached_invoke(model_name, prompt, temperature=0.7)
        print(f"📤 结果: {result.content[:50]}...")
    
    print(f"\n📊 缓存统计: 命中{cache.stats['hits']}次，未命中{cache.stats['misses']}次")
```

## 📝 总结

基础同步调用是LangChain应用开发的基石：

✅ **简单可靠**：最容易理解和实现的调用方式  
✅ **功能完整**：支持所有基础AI交互需求  
✅ **调试友好**：便于问题排查和性能分析  
✅ **扩展性强**：为其他高级功能奠定基础  

## 🔗 相关资源

- [LangChain官方文档 - Chat Models](https://python.langchain.com/docs/modules/model_io/chat/)
- [OpenAI API参数说明](https://platform.openai.com/docs/api-reference/chat)
- [Python dotenv使用指南](https://github.com/theskumar/python-dotenv)

---
*本教程详细解析了LangChain基础同步调用的核心技术。下一期我们将探讨批量调用的效率优化技巧。*
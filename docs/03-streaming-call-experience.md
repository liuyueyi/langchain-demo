# LangChain实战开发教程（三）：流式调用体验升级

> **深入理解流式调用**：LangChain流式输出的核心机制与最佳实践

## 🎯 本文目标

全面解析LangChain流式调用的实现原理，掌握实时输出技术，打造更自然的人机交互体验。

## 📚 核心知识点概览

通过本文你将深入理解：
- **流式调用工作机制**：逐token输出的技术原理
- **用户体验优化**：实时反馈vs完整响应的权衡
- **Token使用监控**：流式场景下的资源统计方法
- **异常处理策略**：网络中断时的优雅降级

## 🔍 流式调用核心技术解析

### 什么是流式调用？

流式调用是一种逐token返回AI生成结果的技术，用户可以看到模型"边思考边输出"的过程，就像真人打字一样。

### 工作机制对比

| 调用方式 | 数据传输 | 用户体验 | 资源消耗 |
|----------|----------|----------|----------|
| 同步调用 | 一次性返回完整结果 | 等待时间长 | 低 |
| 流式调用 | 逐token实时返回 | 即时反馈 | 中等 |
| 批量调用 | 多任务并发处理 | 高效但非实时 | 高 |

## 🚀 核心实现详解

### 1. 基础流式调用实现

```python
def stream_call(model):
    model_instance = init_model(model)
    
    pretty_print_ai_response_prefix("stream")
    
    # 关键：使用 stream() 方法而非 invoke()
    token_usage = None
    for chunk in model_instance.stream("请写一首关于颜色的五言绝句"):
        # 实时输出每个chunk的内容
        if chunk.usage_metadata:
            token_usage = chunk.usage_metadata
        print(chunk.content, end='', flush=True)
    
    # 输出最终统计信息
    pretty_print_ai_response_suffix(chunk, token_usage)
```

### 2. 流式调用的关键差异

**与同步调用的核心区别**：

```python
# 同步调用 - 等待完整结果
response = model.invoke("写一首诗")
print(response.content)  # 一次性输出全部内容

# 流式调用 - 实时输出
for chunk in model.stream("写一首诗"):
    print(chunk.content, end='')  # 逐字符输出
```

## 💡 技术实现要点

### 1. 实时输出控制

```python
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
```

### 2. Token使用监控

```python
def monitor_token_usage_during_stream(model):
    """流式调用中的实时Token监控"""
    total_tokens = 0
    chunks_received = 0
    
    for chunk in model.stream("请详细解释量子计算原理"):
        chunks_received += 1
        
        if chunk.usage_metadata:
            current_total = chunk.usage_metadata.get('total_tokens', 0)
            if current_total > total_tokens:
                total_tokens = current_total
                print(f"\n📊 Token进度: {total_tokens} tokens (已接收{chunks_received}个片段)")
    
    print(f"\n✅ 最终统计: 总共使用 {total_tokens} tokens")
```

## 🎯 实战应用场景

### 场景1：长文本生成

> 源码见 `basic01-model/scene/StreamScene.py`

```python
def long_text_streaming():
    """长文本流式生成示例"""
    long_prompt = """请写一篇关于人工智能未来发展的详细文章，
    包括技术趋势、社会影响、伦理考量等方面，至少1000字。"""
    
    print("📝 开始生成长篇文章...")
    full_text, stats = enhanced_stream_output(long_prompt, model)
    
    print(f"\n\n📋 文章统计:")
    print(f"字数: {len(full_text)} 字符")
    print(f"Token消耗: 输入{stats['input']}, 输出{stats['output']}")
    print(f"生成效率: {stats['output']/len(full_text):.2f} tokens/字符")
```

### 场景2：代码生成实时预览

```python
def code_generation_stream():
    """代码生成的流式预览"""
    code_prompt = "用Python写一个快速排序算法，并添加详细注释"
    
    print("💻 正在生成代码...")
    print("=" * 60)
    
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
```

## ⚡ 性能优化策略

### 1. 缓冲区优化

```python
def buffered_stream_output(prompt, model, buffer_size=10):
    """带缓冲的流式输出，减少频繁打印"""
    buffer = ""
    char_count = 0
    
    for chunk in model.stream(prompt):
        if chunk.content:
            buffer += chunk.content
            char_count += len(chunk.content)
            
            # 达到缓冲区大小或遇到句子结束符时输出
            if len(buffer) >= buffer_size or chunk.content in '.!?。！？':
                print(buffer, end='', flush=True)
                buffer = ""
    
    # 输出剩余缓冲内容
    if buffer:
        print(buffer, end='', flush=True)
```

### 2. 自适应流速控制

```python
import time

def adaptive_stream_speed(prompt, model, target_wpm=200):
    """自适应流速控制，模拟人工打字速度"""
    chars_per_second = target_wpm * 5 / 60  # 每秒字符数
    
    for chunk in model.stream(prompt):
        if chunk.content:
            print(chunk.content, end='', flush=True)
            # 根据内容长度延迟输出
            delay = len(chunk.content) / chars_per_second
            time.sleep(delay)
```

## 🛡️ 异常处理与容错

### 1. 网络中断处理

```python
def resilient_stream_call(prompt, model, max_retries=3):
    """具有重试机制的流式调用"""
    for attempt in range(max_retries):
        try:
            print(f"📡 尝试第 {attempt + 1} 次连接...")
            full_response = ""
            
            for chunk in model.stream(prompt):
                if chunk.content:
                    print(chunk.content, end='', flush=True)
                    full_response += chunk.content
                    
            return full_response
            
        except Exception as e:
            print(f"\n❌ 第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries - 1:
                print("⏳ 等待重试...")
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print("🔄 切换到同步调用模式...")
                return model.invoke(prompt).content
```

### 2. 断点续传机制

```python
def resumeable_stream_call(prompt, model, checkpoint_file="stream_checkpoint.txt"):
    """支持断点续传的流式调用"""
    # 检查是否存在断点
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            completed_content = f.read()
        print(f"🔁 从断点恢复: 已完成 {len(completed_content)} 字符")
    else:
        completed_content = ""
    
    try:
        for chunk in model.stream(prompt):
            if chunk.content:
                print(chunk.content, end='', flush=True)
                completed_content += chunk.content
                
                # 定期保存断点
                if len(completed_content) % 100 == 0:
                    with open(checkpoint_file, 'w') as f:
                        f.write(completed_content)
                        
    finally:
        # 清理断点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
```

## 📊 性能监控与分析

### 1. 实时性能指标

```python
def performance_metrics_stream(prompt, model):
    """流式调用性能指标监控"""
    import time
    
    start_time = time.time()
    token_count = 0
    chunk_count = 0
    
    print("🚀 开始性能监控...")
    
    for chunk in model.stream(prompt):
        chunk_count += 1
        if chunk.content:
            token_count += len(chunk.content.split())
            print(chunk.content, end='', flush=True)
            
        # 每10个chunk输出一次统计
        if chunk_count % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_second = token_count / elapsed if elapsed > 0 else 0
            print(f"\n📊 实时统计: {tokens_per_second:.1f} tokens/sec")
    
    total_time = time.time() - start_time
    print(f"\n📈 最终性能:")
    print(f"总时间: {total_time:.2f} 秒")
    print(f"平均速度: {token_count/total_time:.1f} tokens/sec")
    print(f"总chunk数: {chunk_count}")
```

### 2. 用户体验评估

```python
def ux_evaluation_stream(prompt, model):
    """用户体验评估指标"""
    metrics = {
        'first_token_latency': 0,  # 首token延迟
        'average_chunk_interval': 0,  # 平均chunk间隔
        'total_generation_time': 0,  # 总生成时间
        'content_fluency': 0  # 内容流畅度
    }
    
    start_time = time.time()
    first_token_time = None
    chunk_times = []
    
    for chunk in model.stream(prompt):
        current_time = time.time()
        
        if first_token_time is None:
            first_token_time = current_time
            metrics['first_token_latency'] = first_token_time - start_time
            
        chunk_times.append(current_time)
        print(chunk.content, end='', flush=True)
    
    # 计算各项指标
    metrics['total_generation_time'] = time.time() - start_time
    
    if len(chunk_times) > 1:
        intervals = [chunk_times[i] - chunk_times[i-1] for i in range(1, len(chunk_times))]
        metrics['average_chunk_interval'] = sum(intervals) / len(intervals)
    
    # 简单的流畅度评估（基于chunk间隔的一致性）
    if len(intervals) > 1:
        variance = sum((x - metrics['average_chunk_interval']) ** 2 for x in intervals) / len(intervals)
        metrics['content_fluency'] = 1 / (1 + variance)  # 方差越小，流畅度越高
    
    return metrics
```

## 🎨 高级应用示例

### 1. 多模态流式输出

```python
def multimodal_stream_output(text_prompt, image_generator=None):
    """结合文本和图像的流式输出"""
    print("🎨 开始多模态内容生成...")
    
    # 文本流式输出
    text_content = ""
    for chunk in model.stream(text_prompt):
        if chunk.content:
            print(chunk.content, end='', flush=True)
            text_content += chunk.content
    
    # 如果有图像生成器，同时生成相关图片
    if image_generator:
        print("\n🖼️  正在生成相关图像...")
        image_url = image_generator.generate(text_content[:100])  # 基于前100字符
        print(f"图像生成完成: {image_url}")
```

### 2. 交互式流式对话

```python
def interactive_stream_dialogue():
    """交互式流式对话系统"""
    print("💬 欢迎使用流式对话系统！输入 'quit' 退出")
    
    conversation_history = []
    
    while True:
        user_input = input("\n👤 你说: ")
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("👋 再见！")
            break
            
        conversation_history.append(("user", user_input))
        
        # 构造对话上下文
        full_prompt = "\n".join([f"{role}: {content}" for role, content in conversation_history])
        
        print("🤖 AI: ", end='')
        ai_response = ""
        
        for chunk in model.stream(full_prompt + "\nassistant:"):
            if chunk.content:
                print(chunk.content, end='', flush=True)
                ai_response += chunk.content
        
        conversation_history.append(("assistant", ai_response))
        print()  # 换行
```

## 📝 总结

流式调用为LangChain应用带来了革命性的用户体验提升：

✅ **实时反馈**：用户可以立即看到生成过程  
✅ **自然交互**：模拟人类思考和表达的方式  
✅ **资源透明**：实时监控Token使用情况  
✅ **容错性强**：支持中断恢复和优雅降级  

## 🔗 相关资源

- [LangChain Stream Documentation](https://python.langchain.com/docs/modules/model_io/chat/streaming)
- [Server-Sent Events 规范](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [WebSocket vs HTTP Streaming](https://www.smashingmagazine.com/2018/02/websocket-api/)

---
*本教程深入解析了流式调用的核心机制。下一期我们将探索多轮对话的实现技巧。*
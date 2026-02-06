# LangChain实战开发教程（七）：提示词工程深度解析

> **深入理解提示词工程**：掌握LangChain中f-string和mustache两种提示词模板的使用技巧

## 🎯 本文目标

全面解析LangChain提示词工程技术，深入理解不同模板语法的特点和适用场景，掌握提示词设计的最佳实践。

## 📚 核心知识点概览

通过本文你将掌握：
- **提示词基础概念**：理解提示词在AI应用中的核心作用
- **f-string模板语法**：Python原生字符串格式化的便捷使用
- **mustache模板语法**：功能更强大的模板引擎使用
- **模板选择策略**：根据不同场景选择合适的模板方式
- **安全注意事项**：防范提示词注入等安全风险

## 🔧 提示词工程核心技术解析

### 什么是提示词工程？

提示词工程（Prompt Engineering）是指设计和优化输入给AI模型的提示文本，以获得更准确、更符合预期的输出结果的技术。

### 核心价值

1. **控制输出质量**：通过精心设计的提示词引导模型输出
2. **提高准确性**：减少模型幻觉和无关回答
3. **增强可控性**：精确控制输出格式和内容
4. **降低成本**：减少反复调用的token消耗

## 🚀 两种提示词模板详解

### 方式一：f-string模板（Python原生）

```python
from langchain_core.prompts import PromptTemplate

def f_string_prompt_example():
    """f-string提示词模板示例"""
    
    # 基础f-string模板
    template = """
你是一个起名大师，擅长结合古诗词、五行八字给人取出好听、寓意好、五行圆满的名字。
你应该返回五个名字，并解释每个名字的寓意。
下面是需要取名的信息: 
{info}
"""
    
    # 创建PromptTemplate对象
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["info"]  # 定义需要替换的变量
    )
    
    # 格式化提示词
    prompt = prompt_template.format(
        info="26年2月6日 10:01分出生的小女孩，姓:钱"
    )
    
    print("=== f-string模板生成的提示词 ===")
    print(prompt)
    
    return prompt
```

### 方式二：mustache模板（推荐）

```python
def mustache_prompt_example():
    """mustache提示词模板示例"""
    
    # mustache模板语法
    template = """
你是一个起名大师，擅长结合古诗词、五行八字给人取出好听、寓意好、五行圆满的名字。
你应该返回五个名字，并解释每个名字的寓意。
下面是需要取名的信息: 
{{info}}
"""
    
    # 创建mustache格式的PromptTemplate
    prompt_template = PromptTemplate(
        template=template,
        template_format="mustache",  # 指定使用mustache格式
        input_variables=["info"]
    )
    
    # 格式化提示词
    prompt = prompt_template.format(
        info="26年2月6日 10:01分出生的小女孩，姓:钱"
    )
    
    print("=== mustache模板生成的提示词 ===")
    print(prompt)
    
    return prompt
```

## 💡 两种模板语法对比

### f-string模板特点

**优势**：
✅ Python原生支持，学习成本低  
✅ 语法简洁直观  
✅ 性能较好  

**限制**：
❌ 不支持嵌套访问：如 `{user.name}` 非法  
❌ 不支持格式化：如 `{price:.2f}` 不行  
❌ 不支持表达式：如 `{x+y}` 非法  
❌ 不支持函数调用：如 `{str.upper()}` 不行  
❌ 不支持条件判断和循环  

### mustache模板特点

**优势**：
✅ 支持嵌套对象访问：`{{user.name}}`  
✅ 支持条件渲染：`{{#condition}}...{{/condition}}`  
✅ 支持列表迭代：`{{#items}}...{{/items}}`  
✅ 支持反向条件：`{{^condition}}...{{/condition}}`  
✅ 更强的表达能力  

**注意事项**：
⚠️ 需要额外学习mustache语法  
⚠️ 对于简单场景可能过于复杂  


**✅ 适合使用Mustache的场景：**
- 复杂对象的嵌套属性访问
- 条件逻辑渲染（if/else）
- 列表数据的迭代处理
- 需要动态内容生成的模板

**❌ 不适合的场景：**
- 简单的字符串拼接（用f-string更合适）
- 复杂的业务逻辑处理
- 需要表达式计算的场景

## 🎯 Mustache模板替换核心知识点总结

### 1. **基础语法结构**
```mustache
{{variable}}           # 基础变量替换
{{{variable}}}         # 无转义变量替换
{{&variable}}          # HTML转义变量替换
```

### 2. **条件渲染机制**
```mustache
{{#condition}}         # 条件为真时渲染
  内容块
{{/condition}}         # 条件结束标记

{{^condition}}         # 条件为假时渲染（反向条件）
  内容块
{{/condition}}         # 反向条件结束标记
```

### 3. **对象嵌套访问**
```mustache
{{object.property}}           # 一级属性访问
{{object.nested.property}}    # 多级嵌套访问
{{array.index}}               # 数组索引访问
```

### 4. **循环迭代处理**
```mustache
{{#list}}              # 列表迭代开始
  {{index}}. {{name}}  # 列表项内容
{{/list}}              # 列表迭代结束
```

### 5. **模板定义与使用**
```python
from langchain_core.prompts import PromptTemplate

# 1. 定义模板
template = """
{{#user}}
姓名：{{name}}
年龄：{{age}}
{{#address}}
地址：{{address.city}} {{address.street}}
{{/address}}
{{/user}}
"""

# 2. 创建模板对象
prompt_template = PromptTemplate(
    template=template,
    template_format="mustache",  # 指定mustache格式
    input_variables=["user"]     # 定义输入变量
)

# 3. 格式化模板
result = prompt_template.format(user=user_data)
```

### 6. **数据结构映射**
```python
# 复杂数据结构示例
user_data = {
    "name": "张三",
    "age": 25,
    "address": {
        "city": "北京",
        "street": "长安街1号"
    },
    "hobbies": ["读书", "游泳", "编程"]  # 列表处理
}
```

### 7. **常见错误避免** ⚠️
```python
# ❌ 错误：变量名不匹配
template = "{{name}}"  # 模板中是name
data = {"username": "张三"}  # 数据中是username，不匹配！

# ✅ 正确：变量名保持一致
template = "{{username}}"
data = {"username": "张三"}
```


## 🎯 实战应用场景

### 场景1：基础信息填充

```python
def basic_info_filling():
    """基础信息填充场景"""
    
    # f-string方式更适合这种简单场景 ; 当然这里使用的 mustache 进行的演示
    simple_template = "请为{{gender}}孩起名，出生日期：{{birth_date}}，姓氏：{{surname}}"
    
    prompt_template = PromptTemplate(
        template=simple_template,
        template_format="mustache",
        input_variables=["gender", "birth_date", "surname"]
    )
    
    result = prompt_template.format(
        gender="女",
        birth_date="2026年2月6日",
        surname="钱"
    )
    
    print("基础信息填充结果：")
    print(result)
```

### 场景2：复杂对象处理

```python
def complex_object_handling():
    """复杂对象处理场景"""
    
    # mustache方式（复杂场景推荐）
    complex_template = """
{{#child}}
孩子信息：
姓名：{{name}}
性别：{{gender}}
出生时间：{{birth_info.date}} {{birth_info.time}}

{{#parent}}
父母信息：
父亲：{{parent.father}}
母亲：{{parent.mother}}
{{/parent}}

{{^parent}}
父母信息：未提供
{{/parent}}
{{/child}}

请根据以上信息起名。
"""
    
    prompt_template = PromptTemplate(
        template=complex_template,
        template_format="mustache",
        input_variables=["child"]
    )
    
    # 复杂数据结构
    child_data = {
        "name": "小宝贝",
        "gender": "女",
        "birth_info": {
            "date": "2026年2月6日",
            "time": "上午10:01"
        },
        "parent": {
            "father": "钱先生",
            "mother": "李女士"
        }
    }
    
    result = prompt_template.format(child=child_data)
    print("复杂对象处理结果：")
    print(result)
```

### 场景3：条件渲染

```python
def conditional_rendering():
    """条件渲染场景"""
    
    template = """
{{#is_vip}}
VIP客户专属服务：
姓名：{{name}}
等级：{{level}}
特权：{{privileges}}
{{/is_vip}}

{{^is_vip}}
普通客户服务：
姓名：{{name}}
欢迎使用我们的基础服务
{{/is_vip}}
"""
    
    prompt_template = PromptTemplate(
        template=template,
        template_format="mustache",
        input_variables=["is_vip", "name", "level", "privileges"]
    )
    
    # VIP用户
    vip_result = prompt_template.format(
        is_vip=True,
        name="张三",
        level="钻石会员",
        privileges="专属客服、优先处理、免费升级"
    )
    
    print("VIP用户版本：")
    print(vip_result)
    
    # 普通用户
    regular_result = prompt_template.format(
        is_vip=False,
        name="李四",
        level="",
        privileges=""
    )
    
    print("\n普通用户版本：")
    print(regular_result)
```

## ⚡ 性能优化策略

### 1. 模板缓存机制

```python
class PromptTemplateCache:
    def __init__(self):
        self.template_cache = {}
    
    def get_cached_template(self, template_string, format_type="f-string"):
        """获取缓存的模板对象"""
        cache_key = f"{format_type}:{hash(template_string)}"
        
        if cache_key in self.template_cache:
            print("💾 使用缓存的模板")
            return self.template_cache[cache_key]
        
        # 创建新模板
        if format_type == "mustache":
            template = PromptTemplate(
                template=template_string,
                template_format="mustache",
                input_variables=[]  # 动态推断变量
            )
        else:
            template = PromptTemplate(
                template=template_string,
                input_variables=[]
            )
        
        # 缓存模板
        self.template_cache[cache_key] = template
        print("🆕 创建并缓存新模板")
        
        return template

# 使用示例
def cached_prompt_demo():
    cache = PromptTemplateCache()
    
    template_str = "请为{{gender}}孩起名，姓{{surname}}"
    
    # 第一次调用
    template1 = cache.get_cached_template(template_str, "mustache")
    
    # 第二次调用（使用缓存）
    template2 = cache.get_cached_template(template_str, "mustache")
    
    # 验证是同一个对象
    print(f"是否为同一对象: {template1 is template2}")
```

### 2. 批量提示词生成

```python
def batch_prompt_generation():
    """批量提示词生成优化"""
    
    base_template = """
你是一个{{expert_type}}专家，请为以下情况提供专业建议：
情况描述：{{situation}}
关键要求：{{requirement}}
"""
    
    prompt_template = PromptTemplate(
        template=base_template,
        template_format="mustache",
        input_variables=["expert_type", "situation", "requirement"]
    )
    
    # 批量数据
    batch_data = [
        {
            "expert_type": "育儿",
            "situation": "2岁孩子不爱吃饭",
            "requirement": "提供实用的解决方案"
        },
        {
            "expert_type": "教育",
            "situation": "小学生注意力不集中",
            "requirement": "给出科学的训练方法"
        },
        {
            "expert_type": "心理",
            "situation": "青少年情绪波动大",
            "requirement": "建议有效的沟通技巧"
        }
    ]
    
    # 批量生成
    prompts = []
    for data in batch_data:
        prompt = prompt_template.format(**data)
        prompts.append(prompt)
    
    print("批量生成的提示词：")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. {prompt}")
    
    return prompts
```

## 🛡️ 安全注意事项

### 1. 转义处理机制

```python
def escape_handling_examples():
    """转义处理示例"""
    
    template = """
原始内容：{{content}}        {{! 默认HTML转义：特殊字符会被转义为HTML实体 }}
转义内容：{{{content}}}     {{! 无转义输出：内容原样显示，包括HTML标签 }}
HTML转义：{{&content}}      {{! 同{{{content}}}，不转义输出 }}
"""
    
    prompt_template = PromptTemplate(
        template=template,
        template_format="mustache",
        input_variables=["content"]
    )
    
    # 测试包含潜在危险内容
    dangerous_content = "<script>alert('XSS攻击测试')</script>"
    
    safe_prompt = prompt_template.format(content=dangerous_content)
    print("安全转义示例：")
    print(safe_prompt)
    
    print("\n⚠️  安全提醒：")
    print("- {{content}} 默认进行HTML转义，相对安全")
    print("- {{{content}}} 和 {{&content}} 不转义，使用时要确保内容可信")
    print("- 处理用户输入时优先使用默认转义方式")
```

### 2. 输入验证和清理

```python
import re

def validate_and_clean_input(input_data):
    """输入验证和清理"""
    
    cleaned_data = {}
    
    for key, value in input_data.items():
        if isinstance(value, str):
            # 移除潜在危险字符
            cleaned_value = re.sub(r'[<>"\']', '', value)
            
            # 限制长度
            if len(cleaned_value) > 1000:
                cleaned_value = cleaned_value[:1000] + "...(已截断)"
            
            # 验证内容合理性
            if key == "name" and not re.match(r'^[\u4e00-\u9fffA-Za-z]+$', cleaned_value):
                raise ValueError(f"姓名格式不正确: {cleaned_value}")
            
            cleaned_data[key] = cleaned_value
        else:
            cleaned_data[key] = value
    
    return cleaned_data

# 使用示例
def safe_prompt_generation():
    user_input = {
        "name": "小明<script>",  # 包含潜在危险内容
        "age": "5",
        "hobby": "画画"
    }
    
    try:
        clean_input = validate_and_clean_input(user_input)
        print("清理后的输入：", clean_input)
        
        template = "孩子的姓名是{{name}}，年龄{{age}}岁，爱好{{hobby}}"
        prompt_template = PromptTemplate(
            template=template,
            template_format="mustache",
            input_variables=["name", "age", "hobby"]
        )
        
        safe_prompt = prompt_template.format(**clean_input)
        print("安全的提示词：", safe_prompt)
        
    except ValueError as e:
        print(f"输入验证失败：{e}")
```

## 📊 模板选择决策树

```python
def template_selection_guide(complexity, nesting_needed, conditional_logic):
    """模板选择决策辅助"""
    
    print("🔍 模板选择决策分析：")
    print(f"复杂度: {complexity}")
    print(f"需要嵌套: {nesting_needed}")
    print(f"需要条件逻辑: {conditional_logic}")
    
    if complexity == "simple" and not nesting_needed and not conditional_logic:
        recommendation = "✅ 推荐使用 f-string 模板"
        reason = "简单场景，f-string足够且性能更好"
    elif nesting_needed or conditional_logic:
        recommendation = "✅ 推荐使用 mustache 模板"
        reason = "需要复杂的数据处理和逻辑控制"
    else:
        recommendation = "🔶 建议使用 mustache 模板"
        reason = "虽然f-string可行，但mustache提供更多灵活性"
    
    print(f"\n{recommendation}")
    print(f"理由: {reason}")
    
    return recommendation

# 使用示例
template_selection_guide(
    complexity="complex",
    nesting_needed=True,
    conditional_logic=True
)
```

## 🎨 高级应用示例

### 1. 动态模板生成

```python
def dynamic_template_generation():
    """动态模板生成系统"""
    
    class DynamicPromptBuilder:
        def __init__(self):
            self.templates = {
                "naming": {
                    "simple": "请为{{gender}}孩起名，姓{{surname}}",
                    "detailed": """
你是一个专业的起名大师，请为以下孩子起名：
孩子信息：
- 性别：{{gender}}
- 姓氏：{{surname}}
- 出生日期：{{birth_date}}
{{#parent_info}}
- 父亲：{{parent_info.father}}
- 母亲：{{parent_info.mother}}
{{/parent_info}}
要求：提供5个名字及寓意
"""
                },
                "consultation": {
                    "simple": "请回答：{{question}}",
                    "detailed": """
作为{{expert_type}}专家，请回答以下问题：
问题：{{question}}
背景：{{background}}
要求：{{requirements}}
"""
                }
            }
        
        def build_prompt(self, category, detail_level, **kwargs):
            """构建提示词"""
            if category not in self.templates:
                raise ValueError(f"不支持的类别: {category}")
            
            if detail_level not in self.templates[category]:
                raise ValueError(f"不支持的详细程度: {detail_level}")
            
            template_str = self.templates[category][detail_level]
            
            prompt_template = PromptTemplate(
                template=template_str,
                template_format="mustache",
                input_variables=list(kwargs.keys())
            )
            
            return prompt_template.format(**kwargs)
    
    # 使用示例
    builder = DynamicPromptBuilder()
    
    # 简单起名
    simple_naming = builder.build_prompt(
        category="naming",
        detail_level="simple",
        gender="女",
        surname="钱"
    )
    print("简单起名提示词：")
    print(simple_naming)
    
    # 详细起名
    detailed_naming = builder.build_prompt(
        category="naming",
        detail_level="detailed",
        gender="女",
        surname="钱",
        birth_date="2026年2月6日",
        parent_info={
            "father": "钱先生",
            "mother": "李女士"
        }
    )
    print("\n详细起名提示词：")
    print(detailed_naming)
```

### 2. 模板继承和组合

```python
def template_composition_example():
    """模板组合示例"""
    
    # 基础模板
    header_template = """
{{#header}}
================================================
{{title}}
================================================
{{/header}}
"""
    
    content_template = """
{{#content}}
{{message}}
{{/content}}
"""
    
    footer_template = """
{{#footer}}
------------------------------------------------
{{signature}}
{{date}}
{{/footer}}
"""
    
    # 组合模板
    full_template = header_template + content_template + footer_template
    
    prompt_template = PromptTemplate(
        template=full_template,
        template_format="mustache",
        input_variables=["header", "content", "footer"]
    )
    
    # 数据组装
    data = {
        "header": {
            "title": "起名服务报告"
        },
        "content": {
            "message": "根据您提供的信息，我们为您精心挑选了以下名字：\n1. 钱思雨 - 思绪如雨，温润如玉\n2. 钱雅馨 - 优雅芬芳，品德高尚"
        },
        "footer": {
            "signature": "起名大师团队",
            "date": "2026年2月6日"
        }
    }
    
    composed_prompt = prompt_template.format(**data)
    print("组合模板结果：")
    print(composed_prompt)
```

## 📝 总结

提示词工程是LangChain应用开发的核心技能：

✅ **f-string模板**：适合简单场景，性能优异  
✅ **mustache模板**：功能强大，支持复杂逻辑  
✅ **安全意识**：始终考虑输入验证和转义处理  
✅ **性能优化**：合理使用缓存和批量处理  
✅ **场景适配**：根据具体需求选择合适的模板方式  

## 🔗 相关资源

- [Mustache官方文档](https://mustache.github.io/)
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)
- [Prompt Engineering Guide](https://promptingguide.ai/)

---
*本教程深入解析了提示词工程的核心技术。下一期我们将探索消息系统的高级应用。*
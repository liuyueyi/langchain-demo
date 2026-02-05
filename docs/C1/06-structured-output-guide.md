# LangChain实战开发教程（六）：结构化输出终极指南

> **深入理解结构化输出**：LangChain结构化返回的核心机制与三种实现方式详解

## 🎯 本文目标

全面解析LangChain结构化输出技术，掌握如何让AI返回标准化的数据结构，提升应用的可靠性和可处理性。

## 📚 核心知识点概览

通过本文你将深入掌握：
- **Pydantic模型方式**：功能最丰富的结构化输出方案
- **TypedDict方式**：轻量级的类型注解方案
- **JSON Schema方式**：最灵活的自定义结构定义
- **验证机制**：输入输出的完整性保障
- **性能优化**：结构化输出的效率调优

## 🔍 结构化输出核心技术解析

### 什么是结构化输出？

结构化输出是指让AI模型按照预定义的数据结构返回结果，而不是自由格式的文本。这种方式确保了输出的一致性和可预测性。

### 核心价值

1. **数据一致性**：保证返回格式统一
2. **类型安全**：编译时就能发现类型错误
3. **易于处理**：直接可用的结构化数据
4. **验证机制**：内置数据校验功能

## 🚀 三种实现方式详解

### 方式一：Pydantic模型（推荐）

```python
from pydantic import BaseModel, Field
from typing import List, Optional

def structured_output_pydantic(model):
    """使用Pydantic模型定义结构化输出"""
    
    # 1. 定义数据模型
    class MovieInfo(BaseModel):
        """电影信息结构"""
        title: str = Field(..., description="电影标题")
        year: int = Field(..., description="上映年份", ge=1900, le=2030)
        director: str = Field(..., description="导演姓名")
        genre: List[str] = Field(..., description="电影类型列表")
        rating: float = Field(..., description="评分(0-10)", ge=0, le=10)
        box_office: Optional[float] = Field(None, description="票房收入(百万美元)")
        description: str = Field(..., description="简短剧情描述", max_length=500)
    
    # 2. 绑定结构化输出
    model_instance = init_model(model)
    structured_model = model_instance.with_structured_output(MovieInfo)
    
    # 3. 调用并获取结构化结果
    query = "请提供电影《肖申克的救赎》的详细信息"
    result = structured_model.invoke(query)
    
    # 4. 使用结构化数据
    print(f"🎬 电影: {result.title}")
    print(f"📅 年份: {result.year}")
    print(f"🎥 导演: {result.director}")
    print(f"🎭 类型: {', '.join(result.genre)}")
    print(f"⭐ 评分: {result.rating}/10")
    if result.box_office:
        print(f"💰 票房: ${result.box_office}百万")
    print(f"📝 简介: {result.description}")
    
    return result
```

### 方式二：TypedDict方式

```python
from typing_extensions import TypedDict, Annotated

def structured_output_typeddict(model):
    """使用TypedDict定义结构化输出"""
    
    # 1. 定义类型字典
    class BookInfo(TypedDict):
        """书籍信息结构"""
        title: Annotated[str, "书名"]
        author: Annotated[str, "作者姓名"]
        publication_year: Annotated[int, "出版年份"]
        isbn: Annotated[str, "ISBN号码"]
        pages: Annotated[int, "页数"]
        genres: Annotated[List[str], "书籍类型列表"]
        summary: Annotated[str, "内容简介"]
    
    # 2. 绑定结构化输出
    model_instance = init_model(model)
    structured_model = model_instance.with_structured_output(BookInfo)
    
    # 3. 调用示例
    query = "请提供《三体》这本书的详细信息"
    result = structured_model.invoke(query)
    
    # 4. 处理结果
    print("📚 书籍信息:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return result
```

### 方式三：JSON Schema方式

```python
def structured_output_jsonschema(model):
    """使用JSON Schema定义结构化输出"""
    
    # 1. 定义JSON Schema
    person_schema = {
        "title": "Person",
        "description": "个人信息结构",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "姓名"
            },
            "age": {
                "type": "integer", 
                "description": "年龄",
                "minimum": 0,
                "maximum": 150
            },
            "email": {
                "type": "string",
                "description": "邮箱地址",
                "format": "email"
            },
            "skills": {
                "type": "array",
                "description": "技能列表",
                "items": {
                    "type": "string"
                }
            },
            "address": {
                "type": "object",
                "description": "地址信息",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "country": {"type": "string"}
                },
                "required": ["city", "country"]
            }
        },
        "required": ["name", "age", "email"]
    }
    
    # 2. 绑定结构化输出
    model_instance = init_model(model)
    structured_model = model_instance.with_structured_output(
        person_schema,
        method="json_schema"
    )
    
    # 3. 调用示例
    query = "请提供一个软件工程师的个人信息"
    result = structured_model.invoke(query)
    
    # 4. 验证和使用结果
    print("👤 个人信息:")
    print(f"  姓名: {result['name']}")
    print(f"  年龄: {result['age']}")
    print(f"  邮箱: {result['email']}")
    print(f"  技能: {', '.join(result['skills'])}")
    if 'address' in result:
        addr = result['address']
        print(f"  地址: {addr.get('city', '')}, {addr.get('country', '')}")
    
    return result
```

## 💡 高级特性详解

### 1. 嵌套结构支持

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

def nested_structured_output(model):
    """复杂的嵌套结构化输出"""
    
    # 定义嵌套模型
    class Address(BaseModel):
        street: str
        city: str
        country: str
        postal_code: Optional[str] = None
    
    class ContactInfo(BaseModel):
        phone: str
        email: str
        address: Address
    
    class Employee(BaseModel):
        id: int = Field(..., description="员工ID")
        name: str = Field(..., description="姓名")
        position: str = Field(..., description="职位")
        department: str = Field(..., description="部门")
        contact: ContactInfo
        skills: List[str] = Field(..., description="技能列表")
        hire_date: datetime = Field(..., description="入职日期")
        salary: Optional[float] = Field(None, description="薪资")
    
    model_instance = init_model(model)
    structured_model = model_instance.with_structured_output(Employee)
    
    query = "请提供一位高级软件工程师的完整职业信息"
    employee = structured_model.invoke(query)
    
    # 展示嵌套数据的访问
    print(f"👨‍💼 员工: {employee.name}")
    print(f"📍 地址: {employee.contact.address.city}, {employee.contact.address.country}")
    print(f"📱 电话: {employee.contact.phone}")
    print(f"💼 技能: {', '.join(employee.skills)}")
    
    return employee
```

### 2. 枚举类型支持

```python
from enum import Enum
from pydantic import BaseModel, Field

def enum_structured_output(model):
    """使用枚举类型的结构化输出"""
    
    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium" 
        HIGH = "high"
        URGENT = "urgent"
    
    class Status(str, Enum):
        TODO = "todo"
        IN_PROGRESS = "in_progress"
        REVIEW = "review"
        DONE = "done"
    
    class Task(BaseModel):
        title: str = Field(..., description="任务标题")
        description: str = Field(..., description="任务描述")
        priority: Priority = Field(..., description="优先级")
        status: Status = Field(..., description="状态")
        assignee: str = Field(..., description="负责人")
        estimated_hours: float = Field(..., description="预估工时", gt=0)
        tags: List[str] = Field(default=[], description="标签列表")
    
    model_instance = init_model(model)
    structured_model = model_instance.with_structured_output(Task)
    
    query = "创建一个紧急的代码审查任务"
    task = structured_model.invoke(query)
    
    print(f"📋 任务: {task.title}")
    print(f"🚨 优先级: {task.priority.value}")
    print(f"📊 状态: {task.status.value}")
    print(f"👤 负责人: {task.assignee}")
    
    return task
```

### 3. 条件字段支持

```python
from pydantic import BaseModel, Field
from typing import Union, Optional

def conditional_structured_output(model):
    """基于条件的结构化输出"""
    
    class ProductReview(BaseModel):
        product_name: str = Field(..., description="产品名称")
        rating: int = Field(..., description="评分1-5", ge=1, le=5)
        review_text: str = Field(..., description="评价内容")
        would_recommend: bool = Field(..., description="是否会推荐")
        
        # 条件字段：只有在低分时才需要原因
        reason_for_low_rating: Optional[str] = Field(
            None, 
            description="低分原因（仅在评分≤2时需要）"
        )
        
        # 条件字段：只有在推荐时才需要购买渠道
        purchase_channel: Optional[str] = Field(
            None,
            description="购买渠道（仅在推荐时提供）"
        )
    
    model_instance = init_model(model)
    structured_model = model_instance.with_structured_output(ProductReview)
    
    # 测试不同场景
    scenarios = [
        "给iPhone 15 Pro一个4星好评，会推荐给朋友",
        "给某品牌耳机2星差评，音质太差，不会推荐"
    ]
    
    for scenario in scenarios:
        print(f"\n📝 场景: {scenario}")
        review = structured_model.invoke(scenario)
        
        print(f"  产品: {review.product_name}")
        print(f"  评分: {'★' * review.rating}")
        print(f"  推荐: {'是' if review.would_recommend else '否'}")
        
        if review.reason_for_low_rating:
            print(f"  低分原因: {review.reason_for_low_rating}")
        
        if review.purchase_channel:
            print(f"  购买渠道: {review.purchase_channel}")
```

## 🎯 实战应用场景

### 场景1：数据提取与标准化

```python
def data_extraction_pipeline(model):
    """从非结构化文本中提取结构化数据"""
    
    class FinancialReport(BaseModel):
        company_name: str = Field(..., description="公司名称")
        quarter: str = Field(..., description="季度", pattern=r"Q[1-4] \d{4}")
        revenue: float = Field(..., description="营收(亿元)")
        profit: float = Field(..., description="净利润(亿元)")
        eps: float = Field(..., description="每股收益(元)")
        growth_rate: float = Field(..., description="同比增长率(%)")
        key_highlights: List[str] = Field(..., description="关键亮点列表")
    
    model_instance = init_model(model)
    extractor = model_instance.with_structured_output(FinancialReport)
    
    # 从财报文本中提取结构化信息
    financial_text = """
    苹果公司发布2024年Q1财报，总营收达1200亿美元，净利润250亿美元。
    每股收益6.13美元，同比增长15%。主要亮点包括iPhone销量创新高，
    服务业务收入大幅增长，中国市场表现强劲。
    """
    
    report = extractor.invoke(financial_text)
    
    print("📊 财务报告提取结果:")
    print(f"公司: {report.company_name}")
    print(f"季度: {report.quarter}")
    print(f"营收: ${report.revenue}亿")
    print(f"净利润: ${report.profit}亿")
    print(f"EPS: ${report.eps}")
    print(f"增长率: {report.growth_rate}%")
    print("关键亮点:")
    for highlight in report.key_highlights:
        print(f"  • {highlight}")
    
    return report
```

### 场景2：API响应标准化

```python
def api_response_standardization(model):
    """标准化API响应格式"""
    
    class APIResponse(BaseModel):
        success: bool = Field(..., description="请求是否成功")
        data: Union[dict, list, None] = Field(..., description="响应数据")
        message: str = Field(..., description="响应消息")
        error_code: Optional[str] = Field(None, description="错误码")
        timestamp: str = Field(..., description="时间戳")
        request_id: str = Field(..., description="请求ID")
    
    model_instance = init_model(model)
    response_formatter = model_instance.with_structured_output(APIResponse)
    
    # 模拟不同场景的API响应生成
    scenarios = [
        "用户查询成功，返回用户信息",
        "用户不存在，返回错误信息"
    ]
    
    for scenario in scenarios:
        print(f"\n🌐 场景: {scenario}")
        response = response_formatter.invoke(scenario)
        
        print(f"  成功: {response.success}")
        print(f"  消息: {response.message}")
        if response.error_code:
            print(f"  错误码: {response.error_code}")
        print(f"  时间戳: {response.timestamp}")
        print(f"  请求ID: {response.request_id}")
```

### 场景3：多语言内容结构化

```python
def multilingual_content_structuring(model):
    """多语言内容的结构化处理"""
    
    class MultilingualContent(BaseModel):
        original_text: str = Field(..., description="原文")
        translations: dict = Field(..., description="翻译内容，键为语言代码")
        detected_language: str = Field(..., description="检测到的源语言")
        confidence: float = Field(..., description="语言检测置信度", ge=0, le=1)
        cultural_notes: List[str] = Field(..., description="文化背景说明")
    
    model_instance = init_model(model)
    translator = model_instance.with_structured_output(MultilingualContent)
    
    chinese_text = "春眠不觉晓，处处闻啼鸟"
    
    result = translator.invoke(f"请将以下中文诗句翻译成英文和法文：{chinese_text}")
    
    print("🌏 多语言翻译结果:")
    print(f"原文: {result.original_text}")
    print(f"源语言: {result.detected_language} (置信度: {result.confidence})")
    print("翻译:")
    for lang, translation in result.translations.items():
        print(f"  {lang.upper()}: {translation}")
    print("文化注释:")
    for note in result.cultural_notes:
        print(f"  • {note}")
```

## ⚡ 性能优化策略

### 1. 模型选择优化

```python
def performance_comparison_structured(model_options):
    """不同模型的结构化输出性能对比"""
    import time
    
    class SimpleStructure(BaseModel):
        name: str
        age: int
        city: str
    
    results = {}
    
    for model_name in model_options:
        try:
            print(f"🧪 测试模型: {model_name}")
            
            # 初始化模型
            model = init_model(model_name)
            structured_model = model.with_structured_output(SimpleStructure)
            
            # 性能测试
            start_time = time.time()
            result = structured_model.invoke("张三，25岁，北京")
            end_time = time.time()
            
            results[model_name] = {
                'execution_time': end_time - start_time,
                'result': result,
                'success': True
            }
            
            print(f"  ✓ 执行时间: {results[model_name]['execution_time']:.3f}秒")
            print(f"  ✓ 结果: {result}")
            
        except Exception as e:
            results[model_name] = {
                'execution_time': 0,
                'result': None,
                'success': False,
                'error': str(e)
            }
            print(f"  ✗ 失败: {e}")
    
    # 性能排名
    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        sorted_models = sorted(successful_results.items(), 
                             key=lambda x: x[1]['execution_time'])
        
        print("\n🏆 性能排名:")
        for i, (model_name, data) in enumerate(sorted_models, 1):
            print(f"  {i}. {model_name}: {data['execution_time']:.3f}秒")
    
    return results
```

### 2. 缓存机制优化

```python
from functools import lru_cache
import hashlib

class StructuredOutputCache:
    def __init__(self, maxsize=128):
        self.cache = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get_cache_key(self, model_name, schema_hash, input_text):
        """生成缓存键"""
        key_string = f"{model_name}_{schema_hash}_{input_text}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def cached_structured_output(self, model, schema, input_text):
        """带缓存的结构化输出"""
        # 生成schema哈希
        schema_str = str(schema) if not isinstance(schema, str) else schema
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()
        
        # 生成缓存键
        cache_key = self.get_cache_key(model.__class__.__name__, schema_hash, input_text)
        
        # 检查缓存
        if cache_key in self.cache:
            self.hits += 1
            print("💾 缓存命中")
            return self.cache[cache_key]
        
        self.misses += 1
        print("🆕 缓存未命中，执行模型调用")
        
        # 执行实际调用
        try:
            structured_model = model.with_structured_output(schema)
            result = structured_model.invoke(input_text)
            
            # 存储到缓存
            if len(self.cache) >= self.maxsize:
                # 删除最老的项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            raise

# 使用示例
def cached_structured_demo():
    cache = StructuredOutputCache()
    
    # 定义简单结构
    class UserInfo(BaseModel):
        name: str
        age: int
        occupation: str
    
    model = init_model(model)
    
    # 重复调用测试缓存效果
    test_inputs = [
        "李四，30岁，程序员",
        "王五，28岁，设计师", 
        "李四，30岁，程序员"  # 重复输入，应该缓存命中
    ]
    
    for input_text in test_inputs:
        print(f"\n📥 处理: {input_text}")
        result = cache.cached_structured_output(model, UserInfo, input_text)
        print(f"📤 结果: {result}")
    
    print(f"\n📊 缓存统计: 命中{cache.hits}次，未命中{cache.misses}次")
```

## 🛡️ 错误处理与验证

### 1. 数据验证机制

```python
from pydantic import ValidationError, validator

def robust_structured_output(model):
    """具有强验证机制的结构化输出"""
    
    class RobustData(BaseModel):
        user_id: int = Field(..., gt=0, description="用户ID")
        username: str = Field(..., min_length=3, max_length=20, description="用户名")
        email: str = Field(..., description="邮箱地址")
        age: int = Field(..., ge=13, le=120, description="年龄")
        balance: float = Field(..., ge=0, description="账户余额")
        tags: List[str] = Field(..., max_items=10, description="标签列表")
        
        @validator('email')
        def validate_email(cls, v):
            if '@' not in v:
                raise ValueError('邮箱格式无效')
            return v
        
        @validator('username')
        def validate_username(cls, v):
            if not v.isalnum():
                raise ValueError('用户名只能包含字母和数字')
            return v
    
    model_instance = init_model(model)
    validator_model = model_instance.with_structured_output(RobustData)
    
    # 测试各种边界情况
    test_cases = [
        "用户ID: 12345, 用户名: john123, 邮箱: john@example.com, 年龄: 25, 余额: 1000.50, 标签: [科技, 编程]",
        "用户ID: -1, 用户名: ab, 邮箱: invalid-email, 年龄: 200, 余额: -50, 标签: []",  # 无效数据
        "用户名: user123, 邮箱: user@test.com, 年龄: 18"  # 缺少必需字段
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 测试用例 {i+1}: {test_case}")
        try:
            result = validator_model.invoke(test_case)
            print("✅ 验证通过:")
            print(f"  ID: {result.user_id}")
            print(f"  用户名: {result.username}")
            print(f"  邮箱: {result.email}")
            print(f"  年龄: {result.age}")
            print(f"  余额: ${result.balance}")
            print(f"  标签: {result.tags}")
        except ValidationError as e:
            print("❌ 验证失败:")
            for error in e.errors():
                print(f"  - {error['loc'][0]}: {error['msg']}")
        except Exception as e:
            print(f"❌ 其他错误: {e}")
```

### 2. 优雅降级处理

```python
def graceful_degradation_structured(model):
    """结构化输出的优雅降级机制"""
    
    class PrimaryStructure(BaseModel):
        title: str
        content: str
        metadata: dict
    
    class FallbackStructure(BaseModel):
        raw_response: str
        error_reason: str
    
    def safe_structured_output(model, input_text, primary_schema, fallback_schema):
        """安全的结构化输出，支持降级"""
        try:
            # 首先尝试主要结构
            primary_model = model.with_structured_output(primary_schema)
            result = primary_model.invoke(input_text)
            return {"type": "primary", "data": result}
            
        except Exception as primary_error:
            print(f"⚠️  主要结构化失败: {primary_error}")
            try:
                # 降级到备用结构
                fallback_model = model.with_structured_output(fallback_schema)
                fallback_result = fallback_model.invoke(
                    f"原始请求失败，请提供错误信息。输入: {input_text}, 错误: {str(primary_error)}"
                )
                return {"type": "fallback", "data": fallback_result}
                
            except Exception as fallback_error:
                print(f"❌ 备用结构化也失败: {fallback_error}")
                # 最后的兜底方案
                return {
                    "type": "raw",
                    "data": {
                        "raw_response": "结构化处理完全失败",
                        "error": str(fallback_error)
                    }
                }
    
    model_instance = init_model(model)
    
    # 测试降级机制
    test_inputs = [
        "这是一个正常的结构化请求",
        "这是一个可能导致解析失败的复杂请求，包含特殊字符@#$%^&*()"
    ]
    
    for input_text in test_inputs:
        print(f"\n📥 处理: {input_text}")
        result = safe_structured_output(
            model_instance, 
            input_text, 
            PrimaryStructure, 
            FallbackStructure
        )
        
        print(f"📤 结果类型: {result['type']}")
        if result['type'] == 'primary':
            print(f"  标题: {result['data'].title}")
            print(f"  内容长度: {len(result['data'].content)}字符")
        elif result['type'] == 'fallback':
            print(f"  错误原因: {result['data'].error_reason}")
        else:
            print(f"  原始错误: {result['data']['error']}")
```

## 📊 监控与分析

### 1. 结构化输出质量监控

```python
class StructureQualityMonitor:
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_structuring': 0,
            'validation_errors': 0,
            'fallback_used': 0,
            'avg_processing_time': 0
        }
        self.error_patterns = {}
    
    def monitor_structured_output(self, model, schema, input_text, expected_fields=None):
        """监控结构化输出质量"""
        import time
        
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            structured_model = model.with_structured_output(schema)
            result = structured_model.invoke(input_text)
            processing_time = time.time() - start_time
            
            # 更新成功指标
            self.metrics['successful_structuring'] += 1
            
            # 更新平均处理时间
            current_avg = self.metrics['avg_processing_time']
            total_success = self.metrics['successful_structuring']
            self.metrics['avg_processing_time'] = \
                (current_avg * (total_success - 1) + processing_time) / total_success
            
            # 验证字段完整性
            if expected_fields:
                missing_fields = set(expected_fields) - set(result.dict().keys())
                if missing_fields:
                    print(f"⚠️  缺少期望字段: {missing_fields}")
            
            print(f"✅ 结构化成功 (耗时: {processing_time:.3f}秒)")
            return result
            
        except ValidationError as e:
            self.metrics['validation_errors'] += 1
            error_type = type(e).__name__
            self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1
            print(f"❌ 验证错误: {e}")
            raise
            
        except Exception as e:
            self.metrics['fallback_used'] += 1
            error_type = type(e).__name__
            self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1
            print(f"⚠️  使用备用方案: {e}")
            # 这里可以实现备用逻辑
            raise
    
    def generate_quality_report(self):
        """生成质量报告"""
        success_rate = (self.metrics['successful_structuring'] / 
                       max(self.metrics['total_requests'], 1)) * 100
        
        report = f"""
📊 结构化输出质量报告
====================
总请求数: {self.metrics['total_requests']}
成功结构化: {self.metrics['successful_structuring']}
成功率: {success_rate:.1f}%
验证错误: {self.metrics['validation_errors']}
备用方案使用: {self.metrics['fallback_used']}
平均处理时间: {self.metrics['avg_processing_time']:.3f}秒

常见错误类型:
"""
        
        for error_type, count in sorted(self.error_patterns.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / self.metrics['total_requests']) * 100
            report += f"  {error_type}: {count}次 ({percentage:.1f}%)\n"
        
        return report

# 使用示例
def quality_monitoring_demo():
    monitor = StructureQualityMonitor()
    
    class TestStructure(BaseModel):
        name: str
        value: int
        description: str
    
    model = init_model(model)
    
    test_cases = [
        "名称: 测试项目, 值: 42, 描述: 这是一个测试",
        "name: another test, value: 100, description: another description"
    ]
    
    for test_case in test_cases:
        try:
            result = monitor.monitor_structured_output(
                model, TestStructure, test_case, ['name', 'value', 'description']
            )
            print(f"结果: {result}")
        except Exception as e:
            print(f"处理失败: {e}")
    
    print(monitor.generate_quality_report())
```

### 2. 性能基准测试

```python
def benchmark_structured_outputs(model):
    """结构化输出性能基准测试"""
    import time
    import statistics
    
    class BenchmarkStructure(BaseModel):
        id: int
        name: str
        score: float
        tags: List[str]
        metadata: dict
    
    model_instance = init_model(model)
    structured_model = model_instance.with_structured_output(BenchmarkStructure)
    
    # 测试数据
    test_inputs = [
        f"ID: {i}, 名称: 测试{i}, 分数: {i*10.5}, 标签: [标签1, 标签2], 元数据: {{'key': 'value'}}"
        for i in range(1, 11)
    ]
    
    # 执行基准测试
    execution_times = []
    memory_usage = []
    
    print("🚀 开始性能基准测试...")
    
    for i, input_text in enumerate(test_inputs):
        print(f"  测试 {i+1}/10...", end="")
        
        start_time = time.time()
        try:
            result = structured_model.invoke(input_text)
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            print(f" ✓ {execution_time:.3f}秒")
            
        except Exception as e:
            print(f" ✗ 失败: {e}")
    
    # 生成基准报告
    if execution_times:
        report = f"""
⏱️  性能基准测试报告
==================
测试次数: {len(execution_times)}
最小时间: {min(execution_times):.3f}秒
最大时间: {max(execution_times):.3f}秒
平均时间: {statistics.mean(execution_times):.3f}秒
时间标准差: {statistics.stdev(execution_times):.3f}秒
95%置信区间: ±{1.96 * statistics.stdev(execution_times) / (len(execution_times)**0.5):.3f}秒

性能评级:
"""
        
        avg_time = statistics.mean(execution_times)
        if avg_time < 1.0:
            report += "  🚀 优秀 (< 1秒)"
        elif avg_time < 3.0:
            report += "  ✅ 良好 (1-3秒)"
        elif avg_time < 5.0:
            report += "  ⚠️  一般 (3-5秒)"
        else:
            report += "  ❌ 较差 (> 5秒)"
        
        print(report)
    
    return execution_times
```

## 🎨 高级应用示例

### 1. 动态结构生成

```python
def dynamic_structure_generation(model):
    """根据需求动态生成结构定义"""
    
    def create_dynamic_structure(field_definitions):
        """动态创建Pydantic模型"""
        from pydantic import create_model
        
        # 解析字段定义
        fields = {}
        for field_def in field_definitions:
            name = field_def['name']
            field_type = field_def['type']
            description = field_def.get('description', '')
            constraints = field_def.get('constraints', {})
            
            # 处理类型和约束
            if field_type == 'string':
                field_info = (str, Field(..., description=description, **constraints))
            elif field_type == 'integer':
                field_info = (int, Field(..., description=description, **constraints))
            elif field_type == 'float':
                field_info = (float, Field(..., description=description, **constraints))
            elif field_type == 'boolean':
                field_info = (bool, Field(..., description=description))
            elif field_type == 'array':
                item_type = field_def.get('item_type', 'string')
                if item_type == 'string':
                    field_info = (List[str], Field(..., description=description))
                elif item_type == 'integer':
                    field_info = (List[int], Field(..., description=description))
            
            fields[name] = field_info
        
        # 动态创建模型
        DynamicModel = create_model('DynamicStructure', **fields)
        return DynamicModel
    
    # 动态结构示例
    customer_fields = [
        {
            'name': 'customer_id',
            'type': 'string',
            'description': '客户ID',
            'constraints': {'min_length': 5, 'max_length': 20}
        },
        {
            'name': 'name', 
            'type': 'string',
            'description': '客户姓名'
        },
        {
            'name': 'age',
            'type': 'integer',
            'description': '年龄',
            'constraints': {'ge': 18, 'le': 100}
        },
        {
            'name': 'preferences',
            'type': 'array',
            'item_type': 'string',
            'description': '偏好列表'
        }
    ]
    
    # 创建动态模型
    CustomerModel = create_dynamic_structure(customer_fields)
    
    model_instance = init_model(model)
    dynamic_model = model_instance.with_structured_output(CustomerModel)
    
    # 测试动态结构
    customer_data = "客户ID: CUST001, 姓名: 张三, 年龄: 30, 偏好: [科技, 旅游, 美食]"
    result = dynamic_model.invoke(customer_data)
    
    print("👥 动态结构化客户数据:")
    for field, value in result.dict().items():
        print(f"  {field}: {value}")
    
    return result
```

### 2. 多模型结构化集成

```python
class MultiModelStructuredProcessor:
    def __init__(self, models_config):
        self.models = {}
        self.structures = {}
        
        # 初始化多个模型
        for name, config in models_config.items():
            self.models[name] = init_model(config['model_name'])
            self.structures[name] = config['structure_class']
    
    def process_with_best_model(self, input_text, criteria='accuracy'):
        """使用最适合的模型处理结构化输出"""
        results = {}
        
        # 并行处理所有模型
        for model_name, model in self.models.items():
            try:
                structure_class = self.structures[model_name]
                structured_model = model.with_structured_output(structure_class)
                result = structured_model.invoke(input_text)
                results[model_name] = {
                    'result': result,
                    'success': True,
                    'confidence': self.calculate_confidence(result)
                }
            except Exception as e:
                results[model_name] = {
                    'result': None,
                    'success': False,
                    'error': str(e),
                    'confidence': 0
                }
        
        # 根据标准选择最佳结果
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if not successful_results:
            raise Exception("所有模型都处理失败")
        
        if criteria == 'accuracy':
            best_model = max(successful_results.items(), 
                           key=lambda x: x[1]['confidence'])
        elif criteria == 'speed':
            # 这里应该基于处理时间选择
            best_model = list(successful_results.items())[0]
        
        print(f"🎯 选择模型: {best_model[0]} (置信度: {best_model[1]['confidence']:.2f})")
        return best_model[1]['result']
    
    def calculate_confidence(self, result):
        """计算结果置信度"""
        # 简单的置信度计算示例
        if hasattr(result, '__dict__'):
            # 检查必填字段是否完整
            fields = result.__dict__
            required_fields = [field for field in fields if field != '_sa_instance_state']
            filled_fields = [field for field, value in fields.items() 
                           if value is not None and value != '']
            return len(filled_fields) / len(required_fields) if required_fields else 0
        return 0.5

# 使用示例
def multi_model_demo():
    # 配置多个模型
    models_config = {
        'gpt_model': {
            'model_name': 'gpt-4',
            'structure_class': MovieInfo  # 假设已定义
        },
        'claude_model': {
            'model_name': 'claude-3',
            'structure_class': MovieInfo
        }
    }
    
    processor = MultiModelStructuredProcessor(models_config)
    
    try:
        result = processor.process_with_best_model(
            "请提供电影《阿凡达》的详细信息",
            criteria='accuracy'
        )
        print("🎬 最佳结果:", result)
    except Exception as e:
        print(f"处理失败: {e}")
```

## 📝 总结

结构化输出是LangChain应用的重要技术：

✅ **数据一致性**：确保输出格式标准化  
✅ **类型安全**：编译时错误检测  
✅ **易于集成**：直接可用的结构化数据  
✅ **验证机制**：内置数据完整性保障  
✅ **性能优化**：缓存和并行处理支持  

## 🔗 相关资源

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangChain Structured Output Guide](https://python.langchain.com/docs/modules/model_io/chat/structured_output)
- [JSON Schema Specification](https://json-schema.org/)

---
*本教程完整覆盖了LangChain结构化输出的所有核心技术和应用场景。至此，LangChain实战开发系列教程圆满完成！*
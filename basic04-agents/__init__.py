"""
LangChain Agents 模块
包含完整的Agents学习示例和实用工具
"""

# 基础Agents功能
from .BasicAgents import (
    init_model,
    calculator,
    weather_checker,
    web_search,
    create_basic_agent,
    basic_agent_demo
)

# 高级工具功能
from .AdvancedToolsInAgents import (
    safe_calculator,
    react_weather_checker,
    DynamicToolFactory,
    create_advanced_agent
)

# 结构化输出功能
from .StructuredOutputAgents import (
    CalculationResult,
    WeatherReport,
    FinancialSummary,
    AnalysisReport,
    structured_calculator,
    weather_analyzer
)

# 记忆功能
from .MemoryAgents import (
    EnhancedConversationMemory,
    MemoryAwareAgent,
    personalized_greeting,
    preference_based_recommendation
)

# 流式功能
from .StreamingAgents import (
    StreamingCallbackHandler,
    StreamingAgent,
    async_calculator,
    async_weather_checker
)

__all__ = [
    # 基础功能
    'init_model',
    'calculator',
    'weather_checker',
    'basic_agent_demo',
    

    # 结构化输出
    'CalculationResult',
    'WeatherReport',
    'FinancialSummary',
    'AnalysisReport',
    'structured_calculator',
    'weather_analyzer',
    
    # 记忆功能
    'EnhancedConversationMemory',
    'MemoryAwareAgent',
    'personalized_greeting',
    'preference_based_recommendation',
    
    # 流式功能
    'StreamingCallbackHandler',
    'StreamingAgent',
    'async_calculator',
    'async_weather_checker',
]

# 版本信息
__version__ = "1.0.0"
__author__ = "一灰灰"
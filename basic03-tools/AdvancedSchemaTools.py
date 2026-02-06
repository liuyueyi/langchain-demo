"""
LangChain Tools 高级Schema定义示例
展示如何定义具有复杂参数类型的工具，包括Pydantic模型和详细的参数验证
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 加载环境变量
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config-zhipu.env')
load_dotenv(config_path)

# 初始化环境变量
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('BASE_URL')
model_name = os.getenv('MODEL')


def pretty_print_schema_info(tool_obj):
    """美化的Schema信息输出"""
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"🔧 工具名称: {tool_obj.name}")
    print(f"📝 工具描述: {tool_obj.description}")
    print(f"📋 工具Schema:")
    if hasattr(tool_obj, 'args_schema'):
        print(f"   Schema: {tool_obj.args_schema.model_json_schema()}")
    elif hasattr(tool_obj, 'args'):
        print(f"   Args: {tool_obj.args}")
    print(separator)


class UserSearchInput(BaseModel):
    """用户搜索的输入参数"""
    username: str = Field(description="用户名", min_length=1, max_length=50)
    department: Optional[str] = Field(default=None, description="部门名称")
    active_only: bool = Field(default=True, description="是否只返回活跃用户")


class DatabaseQueryInput(BaseModel):
    """数据库查询的输入参数"""
    table_name: str = Field(description="表名", min_length=1)
    columns: List[str] = Field(description="要查询的列名列表", default_factory=list)
    conditions: Dict[str, Any] = Field(description="查询条件", default_factory=dict)
    limit: Optional[int] = Field(default=100, description="查询结果数量限制")


@tool(args_schema=UserSearchInput)
def search_user(username: str, department: Optional[str] = None, active_only: bool = True) -> Dict[str, Any]:
    """
    根据用户名搜索用户信息的工具
    
    Args:
        username: 用户名
        department: 部门名称（可选）
        active_only: 是否只返回活跃用户（默认为True）
        
    Returns:
        Dict: 包含用户信息的字典
    """
    print(f"搜索用户: {username}, 部门: {department}, 仅活跃用户: {active_only}")
    
    # 模拟用户数据库
    mock_users = {
        "张三": {"id": 1, "name": "张三", "department": "技术部", "active": True},
        "李四": {"id": 2, "name": "李四", "department": "销售部", "active": False},
        "王五": {"id": 3, "name": "王五", "department": "技术部", "active": True},
        "赵六": {"id": 4, "name": "赵六", "department": "人事部", "active": True},
    }
    
    user = mock_users.get(username)
    if user:
        # 应用过滤条件
        if department and user["department"] != department:
            return {"error": f"用户 {username} 不在 {department} 部门"}
        if active_only and not user["active"]:
            return {"error": f"用户 {username} 不活跃"}
        return {"user_found": True, "user_info": user}
    else:
        return {"user_found": False, "message": f"未找到用户 {username}"}


@tool(args_schema=DatabaseQueryInput)
def query_database(table_name: str, columns: List[str], conditions: Dict[str, Any], limit: Optional[int] = 100) -> Dict[str, Any]:
    """
    查询数据库的工具
    
    Args:
        table_name: 表名
        columns: 要查询的列名列表
        conditions: 查询条件
        limit: 查询结果数量限制
        
    Returns:
        Dict: 查询结果
    """
    print(f"查询表: {table_name}")
    print(f"列: {columns}")
    print(f"条件: {conditions}")
    print(f"限制: {limit}")
    
    # 模拟数据库查询
    mock_data = {
        "employees": [
            {"id": 1, "name": "张三", "department": "技术部", "salary": 15000},
            {"id": 2, "name": "李四", "department": "销售部", "salary": 12000},
            {"id": 3, "name": "王五", "department": "技术部", "salary": 18000},
        ],
        "departments": [
            {"id": 1, "name": "技术部", "head": "张主任"},
            {"id": 2, "name": "销售部", "head": "李主任"},
            {"id": 3, "name": "人事部", "head": "王主任"},
        ]
    }
    
    table_data = mock_data.get(table_name, [])
    
    # 应用筛选条件
    filtered_data = []
    for row in table_data:
        match = True
        for key, value in conditions.items():
            if key not in row or row[key] != value:
                match = False
                break
        if match:
            filtered_data.append(row)
    
    # 应用列筛选
    if columns:
        filtered_data = [{k: v for k, v in row.items() if k in columns} for row in filtered_data]
    
    # 应用数量限制
    if limit:
        filtered_data = filtered_data[:limit]
    
    return {
        "table": table_name,
        "total_rows": len(filtered_data),
        "data": filtered_data
    }


def create_structured_tool_manually():
    """手动创建带Schema的工具"""
    def advanced_calculator(numbers: List[float], operation: str) -> Dict[str, Any]:
        """高级计算器，支持对数字列表执行操作"""
        print(f"对数字列表 {numbers} 执行 {operation} 操作")
        
        if operation == "sum":
            result = sum(numbers)
        elif operation == "average":
            result = sum(numbers) / len(numbers) if numbers else 0
        elif operation == "max":
            result = max(numbers) if numbers else 0
        elif operation == "min":
            result = min(numbers) if numbers else 0
        else:
            raise ValueError(f"不支持的操作: {operation}")
        
        return {
            "operation": operation,
            "numbers": numbers,
            "result": result,
            "count": len(numbers)
        }
    
    # 定义输入Schema
    class CalculatorInput(BaseModel):
        numbers: List[float] = Field(description="要计算的数字列表", min_items=1)
        operation: str = Field(
            description="计算操作 (sum, average, max, min)",
            json_schema_extra={"enum": ["sum", "average", "max", "min"]}
        )
    
    structured_tool = StructuredTool(
        name="AdvancedCalculator",
        description="高级计算器，支持对数字列表执行多种数学操作",
        func=advanced_calculator,
        args_schema=CalculatorInput
    )
    
    return structured_tool


def advanced_schema_demo():
    """高级Schema工具演示"""
    print("🚀 开始 LangChain Tools 高级Schema示例演示")

    model = init_chat_model(model=model_name,
                           model_provider="openai",
                           temperature=0.7,
                           timeout=30,
                           max_tokens=1000,
                           max_retries=3)
    
    # 1. 使用Pydantic Schema的用户搜索工具
    print("\n1️⃣ 用户搜索工具 (带Schema):")
    pretty_print_schema_info(search_user)
    
    # 使用模型触发工具调用（模拟工具回调）
    tools = [search_user, query_database]
    model_with_tools = model.bind_tools(tools)
    
    user_request = "请帮我查找技术部的张三用户信息，只返回活跃用户"
    print(f"   用户请求: {user_request}")
    
    response = model_with_tools.invoke([HumanMessage(content=user_request)])
    
    if response.tool_calls:
        print(f"   模型决定调用工具: {response.tool_calls[0]['name']}")
        print(f"   工具参数: {response.tool_calls[0]['args']}")
        
        # 执行工具调用（这才是真正的工具回调）
        for tool_call in response.tool_calls:
            if tool_call['name'] == search_user.name:
                result = search_user.invoke(tool_call['args'])
                print(f"   工具调用结果: {result}")
    else:
        print("   模型决定不需要调用工具")
    
    # 2. 使用Pydantic Schema的数据库查询工具
    print("\n2️⃣ 数据库查询工具 (带Schema):")
    pretty_print_schema_info(query_database)
    
    # 使用模型触发工具调用
    db_request = "查询员工表中技术部员工的姓名和薪资信息，限制10条"
    print(f"   用户请求: {db_request}")
    
    response2 = model_with_tools.invoke([HumanMessage(content=db_request)])
    
    if response2.tool_calls:
        print(f"   模型决定调用工具: {response2.tool_calls[0]['name']}")
        print(f"   工具参数: {response2.tool_calls[0]['args']}")
        
        # 执行工具调用
        for tool_call in response2.tool_calls:
            if tool_call['name'] == query_database.name:
                result = query_database.invoke(tool_call['args'])
                print(f"   工具调用结果: {result}")
    else:
        print("   模型决定不需要调用工具")
    
    # 3. 手动创建的结构化工具
    print("\n3️⃣ 手动创建的高级计算器工具:")
    manual_tool = create_structured_tool_manually()
    pretty_print_schema_info(manual_tool)
    
    # 将手动创建的工具也加入工具列表
    all_tools = [search_user, query_database, manual_tool]
    model_with_all_tools = model.bind_tools(all_tools)
    
    calc_request = "计算数字列表 [10, 20, 30, 40] 的平均值"
    print(f"   用户请求: {calc_request}")
    
    response3 = model_with_all_tools.invoke([HumanMessage(content=calc_request)])
    
    if response3.tool_calls:
        print(f"   模型决定调用工具: {response3.tool_calls[0]['name']}")
        print(f"   工具参数: {response3.tool_calls[0]['args']}")
        
        # 执行工具调用
        for tool_call in response3.tool_calls:
            if tool_call['name'] == manual_tool.name:
                result = manual_tool.invoke(tool_call['args'])
                print(f"   工具调用结果: {result}")
    else:
        print("   模型决定不需要调用工具")
    
    # 4. Schema验证示例
    print("\n4️⃣ Schema验证示例:")
    validation_request = "请帮我查找空用户名的用户信息"
    
    response4 = model_with_tools.invoke([HumanMessage(content=validation_request)])
    
    if response4.tool_calls:
        print(f"   模型决定调用工具: {response4.tool_calls[0]['name']}")
        print(f"   工具参数: {response4.tool_calls[0]['args']}")
        
        # 尝试执行工具调用，可能会因验证失败而抛出异常
        try:
            for tool_call in response4.tool_calls:
                if tool_call['name'] == search_user.name:
                    result = search_user.invoke(tool_call['args'])
                    print(f"   工具调用结果: {result}")
        except Exception as e:
            print(f"   ✅ 正确捕获验证错误: {e}")
    else:
        print("   模型决定不需要调用工具")


if __name__ == "__main__":
    advanced_schema_demo()
"""
Mustache模板使用示例
展示LangChain中PromptTemplate配合mustache格式的各种用法
"""

from langchain_core.prompts import PromptTemplate


def basic_mustache_example():
    """基础mustache模板示例"""
    print("=== 基础mustache模板示例 ===")
    
    template = """
你是一个起名大师，请为{{gender}}孩起名。
孩子信息：{{info}}
要求：返回5个名字及寓意。
"""
    
    prompt_template = PromptTemplate(
        template=template,
        template_format="mustache",
        input_variables=["gender", "info"]
    )
    
    prompt = prompt_template.format(
        gender="女",
        info="26年2月6日出生，姓钱"
    )
    
    print(prompt)
    print()


def nested_object_example():
    """嵌套对象示例"""
    print("=== 嵌套对象示例 ===")

    
    template = """
---------------------------
{{#child}}  {{! 嵌套对象开始：整个child对象作为上下文 }}
姓名：{{name}}  {{! 访问child.name }}
性别：{{gender}}  {{! 访问child.gender }}
出生时间：{{birth_info.date}} {{birth_info.time}}  {{! 嵌套对象访问：child.birth_info.date 和 child.birth_info.time }}

{{#parent}}  {{! 条件判断开始：如果parent存在且不为null/false/empty }}
父亲：{{parent.father}}  {{! 访问child.parent.father }}
母亲：{{parent.mother}}  {{! 访问child.parent.mother }}
{{/parent}}  {{! 条件判断结束 }}

{{^parent}}  {{! 反向条件判断：如果parent不存在或为null/false/empty }}
父母信息缺失  {{! 当parent为空时显示此内容 }}
{{/parent}}  {{! 反向条件判断结束 }}
{{/child}}  {{! 嵌套对象结束 }}

请根据以上信息起名。
---------------------------
"""
    
    prompt_template = PromptTemplate(
        template=template,
        template_format="mustache",
        input_variables=["child"]
    )
    
    data = {
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
    
    prompt = prompt_template.format(child=data)
    print("有父母信息的情况：")
    print(prompt)
    
    # 演示条件判断：当parent为None时
    print("\n无父母信息的情况：")
    data_no_parent = {
        "name": "小宝贝",
        "gender": "女",
        "birth_info": {
            "date": "2026年2月6日",
            "time": "上午10:01"
        },
        "parent": None  # parent为None，触发{{^parent}}条件
    }
    
    prompt_no_parent = prompt_template.format(child=data_no_parent)
    print(prompt_no_parent)
    print()


def conditional_rendering_example():
    """条件渲染示例"""
    print("=== 条件渲染示例 ===")
    # {{#}}} 表示为真的时候执行，对应的 {{/}} 表示条件结束
    # {{^}} 表示非真的时候执行，同样的 {{/}} 表示条件结束
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
    vip_prompt = prompt_template.format(
        is_vip=True,
        name="张三",
        level="钻石会员",
        privileges="专属客服、优先处理、免费升级"
    )
    print("VIP用户版本：")
    print(vip_prompt)
    
    # 普通用户
    regular_prompt = prompt_template.format(
        is_vip=False,
        name="李四",
        level="",  # 不会被使用
        privileges=""  # 不会被使用
    )
    print("普通用户版本：")
    print(regular_prompt)
    print()


def list_iteration_example():
    """列表迭代示例"""
    print("=== 列表迭代示例 ===")
    
    template = """
可选的名字：
{{#names}}
{{index}}. {{name}} - {{meaning}}
{{/names}}
{{^names}}
暂无推荐名字
{{/names}}
"""
    
    prompt_template = PromptTemplate(
        template=template,
        template_format="mustache",
        input_variables=["names"]
    )
    
    name_list = [
        {"index": 1, "name": "钱思雨", "meaning": "思绪如雨，温润如玉"},
        {"index": 2, "name": "钱雅馨", "meaning": "优雅芬芳，品德高尚"},
        {"index": 3, "name": "钱婉儿", "meaning": "温婉可人，气质出众"},
        {"index": 4, "name": "钱诗涵", "meaning": "诗意盎然，内涵丰富"},
        {"index": 5, "name": "钱梦瑶", "meaning": "美梦成真，瑶池仙境"}
    ]
    
    prompt = prompt_template.format(names=name_list)
    print(prompt)
    print()


def partial_template_example():
    """部分模板示例（模拟）"""
    print("=== 部分模板示例 ===")
    
    # 注意：LangChain的PromptTemplate对mustache的部分模板支持有限
    # 这里展示一种变通的方法
    
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
    
    # 组合使用
    full_template = header_template + content_template + """
{{#footer}}
------------------------------------------------
{{signature}}
{{/footer}}
"""
    
    prompt_template = PromptTemplate(
        template=full_template,
        template_format="mustache",
        input_variables=["header", "content", "footer"]
    )
    
    data = {
        "header": {
            "title": "起名服务报告"
        },
        "content": {
            "message": "根据您提供的信息，我们为您精心挑选了以下名字："
        },
        "footer": {
            "signature": "起名大师团队敬上"
        }
    }
    
    prompt = prompt_template.format(**data)
    print(prompt)
    print()


def escape_example():
    """转义示例
    
    演示mustache模板中的三种转义方式：
    1. 默认转义：{{content}} - 自动HTML转义，防止XSS攻击
    2. 无转义：{{{content}}} - 不进行任何转义，原样输出
    3. HTML转义：{{&content}} - 同{{{content}}}，不转义输出
    
    安全提醒：使用无转义方式时要确保内容来源可信，避免安全风险
    """
    print("=== 转义示例 ===")
    
    template = """
原始内容：{{content}}        {{! 默认HTML转义：特殊字符会被转义为HTML实体 }}
转义内容：{{{content}}}     {{! 无转义输出：内容原样显示，包括HTML标签 }}
HTML转义：{{&content}}      {{! 同样无转义：&符号是{{{}}}的简写形式 }}
"""
    
    prompt_template = PromptTemplate(
        template=template,
        template_format="mustache",
        input_variables=["content"]
    )
    
    # 测试包含HTML标签的内容
    prompt = prompt_template.format(
        content="<script>alert('test')</script>"  # 恶意脚本代码，用于测试转义效果
    )
    
    print(prompt)
    print()
    print("预期输出说明：")
    print("- 原始内容：会显示转义后的HTML实体")
    print("- 转义内容：会执行JavaScript弹窗（危险！）")
    print("- HTML转义：同转义内容，也会执行脚本")
    print("⚠️  安全提示：实际项目中应避免使用无转义方式处理用户输入")



def practical_naming_example():
    """实用的起名示例"""
    print("=== 实用起名示例 ===")
    
    template = """
{{#request}}
起名请求详情：
================================================
申请人：{{request.applicant}}
孩子信息：
  - 姓名：{{request.child_name}}
  - 性别：{{request.gender}}
  - 出生时间：{{request.birth_date}}
  - 姓氏：{{request.surname}}
特殊要求：{{request.requirements}}
{{#request.parent_names}}
父母姓名：{{request.parent_names.father}} & {{request.parent_names.mother}}
{{/request.parent_names}}
{{^request.parent_names}}
父母姓名：未提供
{{/request.parent_names}}
================================================

{{#expert}}
起名专家建议：
{{#names}}
{{index}}. {{name}}
   寓意：{{meaning}}
   五行：{{five_elements}}
   {{#is_recommended}}★ 推荐 ★{{/is_recommended}}
   
{{/names}}
{{/expert}}
{{/request}}
"""
    
    prompt_template = PromptTemplate(
        template=template,
        template_format="mustache",
        input_variables=["request", "expert"]
    )
    
    # 构造数据
    request_data = {
        "applicant": "钱先生",
        "child_name": "小宝贝",
        "gender": "女",
        "birth_date": "2026年2月6日上午10:01",
        "surname": "钱",
        "requirements": "希望名字优雅、有文化内涵，避免生僻字",
        "parent_names": {
            "father": "钱建国",
            "mother": "李雅琴"
        }
    }
    
    expert_data = {
        "names": [
            {
                "index": 1,
                "name": "钱思雨",
                "meaning": "思绪如春雨般细腻温润，寓意聪慧温柔",
                "five_elements": "水木相生",
                "is_recommended": True
            },
            {
                "index": 2,
                "name": "钱雅馨",
                "meaning": "雅致芬芳，品德高尚如兰花",
                "five_elements": "土金相生",
                "is_recommended": False
            },
            {
                "index": 3,
                "name": "钱婉清",
                "meaning": "温婉清雅，如清水芙蓉般纯净",
                "five_elements": "水金相生",
                "is_recommended": True
            }
        ]
    }
    
    prompt = prompt_template.format(
        request=request_data,
        expert=expert_data
    )
    
    print(prompt)


if __name__ == "__main__":
    print("📚 Mustache模板使用示例")
    print("=" * 50)
    
    basic_mustache_example()
    nested_object_example()
    conditional_rendering_example()
    list_iteration_example()
    partial_template_example()
    escape_example()
    practical_naming_example()
    
    print("\n 所有示例演示完毕！")
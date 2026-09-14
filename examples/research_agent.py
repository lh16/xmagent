import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent


load_dotenv()
    
# 接入模型（兼容OpenAI接口）
model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 使用Google内置搜索（无需额外安装）
#internet_search = {"google_search": {}}

# 使用OpenAI内置搜索
internet_search = {"type": "web_search"}

# 创建研究助手
agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt="""
    你是一个专业的研究助手。
    你的工作是进行深入研究，然后撰写一份完整的报告。
    你可以使用互联网搜索工具来收集信息。
    
    研究报告格式：
    - 标题
    - 摘要（核心发现）
    - 详细分析（分章节）
    - 结论与建议
    """,
)

# 运行
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "请研究2026年AI Agent框架的主要趋势，并总结3个关键发现"
    }]
})

print(result["messages"][-1].content)
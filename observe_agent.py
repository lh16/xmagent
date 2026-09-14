# observe_agent.py
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from tavily import TavilyClient
from deepagents import create_deep_agent


def main():
    # 从 .env 加载环境变量
    load_dotenv()

    # 初始化 Tavily 客户端
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # 定义搜索工具
    @tool
    def internet_search(query: str, max_results: int = 5) -> str:
        """运行网络搜索，返回相关信息。"""
        result = tavily_client.search(query, max_results=max_results)
        return str(result)

    # 接入模型（兼容 OpenAI 接口）
    model = ChatOpenAI(
        model=os.getenv("SILICONFLOW_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    )

    agent = create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt="你是一个研究专家。对于复杂任务，先制定计划再执行。",
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": """
            请完成以下任务：
            1. 搜索2026年AI Agent框架的最新对比
            2. 分析LangGraph和CrewAI的优劣
            3. 给出选型建议
            """
        }]
    })

    # 查看 Agent 的完整思考过程
    messages = result["messages"]
    for msg in messages:
        print(f"[{msg.type}]: {msg.content[:200]}...")


if __name__ == "__main__":
    main()

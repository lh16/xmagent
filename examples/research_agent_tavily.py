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
    # 我们可以先去 https://tavily.com/ 注册一个账号，然后在 https://tavily.com/account/api-keys 创建一个 API key，免费的额度支持个人学习基本够用
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

    # 创建 Agent
    agent = create_deep_agent(
        model=model,
        tools=[internet_search],
        system_prompt="进行研究并撰写一份完整的报告。",
    )

    # 运行
    result = agent.invoke({
        "messages": [{"role": "user", "content": "什么是Deep Agents？"}]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent


def main():
    # 从 .env 文件加载环境变量（不会覆盖已存在的系统环境变量）
    load_dotenv()
    
    # 接入模型（兼容OpenAI接口）
    model = ChatOpenAI(
        model=os.getenv("SILICONFLOW_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    )

    # 创建Deep Agent
    agent = create_deep_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是一个有帮助的助手，可以回答天气问题。",
    )

    # 运行Agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
    )
    print(result["messages"][-1].content)


# 定义自定义工具函数
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。
        
    Args:
        city: 城市名称
    """
    return f"{city}的天气：晴，25°C"

if __name__ == "__main__":
    main()

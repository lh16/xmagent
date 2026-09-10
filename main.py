import os

from dotenv import load_dotenv


def main():
    # 从 .env 文件加载环境变量（不会覆盖已存在的系统环境变量）
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if api_key:
        # 只显示前后几位，避免泄露完整密钥
        print(f"已加载 API Key: {api_key[:6]}...{api_key[-4:]}")
    else:
        print("未找到 OPENAI_API_KEY，请检查 .env 文件")

    print(f"Base URL: {base_url}")


if __name__ == "__main__":
    main()

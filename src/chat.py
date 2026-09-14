"""
交互式对话脚本
用于在终端中与智能客服进行对话
"""

import sys
from src.agents.main_agent import CustomerServiceAgent


def main():
    """主函数：交互式对话"""
    # 创建Agent
    agent = CustomerServiceAgent()
    
    print("=" * 60)
    print("智能客服系统（输入 'quit' 退出）")
    print("=" * 60)
    print()
    
    # 对话历史
    messages = []
    
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("再见！")
                break
            
            # 添加到历史
            messages.append({"role": "user", "content": user_input})
            
            # 调用Agent
            reply = agent.chat_with_history(messages)
            
            # 添加到历史
            messages.append({"role": "assistant", "content": reply})
            
            # 打印回复
            print(f"小智: {reply}")
            print()
            
            # 限制历史长度
            if len(messages) > 20:
                messages = messages[-20:]
                
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            continue


if __name__ == "__main__":
    main()
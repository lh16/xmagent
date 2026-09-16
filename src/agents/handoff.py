"""
任务交接模块
处理子Agent之间的任务交接
"""

from datetime import datetime
from typing import Dict, Optional
from src.agents.intent import Intent
from src.utils.logger import log


class HandoffManager:
    """任务交接管理器"""
    
    def __init__(self, subagents: Optional[Dict] = None):
        """
        初始化交接管理器
        
        Args:
            subagents: 子Agent字典
        """
        self.subagents = subagents or {}
        self.handoff_history = []
    
    def handoff(self, from_agent: str, to_intent: Intent, 
                message: str, context: Optional[str] = None) -> str:
        """
        执行任务交接
        
        Args:
            from_agent: 来源Agent名称
            to_intent: 目标意图
            message: 交接的消息
            context: 附加上下文
        
        Returns:
            目标Agent的回复
        """
        log.info(f"[任务交接] {from_agent} → {to_intent.value}")
        
        # 记录交接
        self.handoff_history.append({
            "from": from_agent,
            "to": to_intent.value,
            "message": message,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        })
        
        # 检查目标Agent是否存在
        if to_intent not in self.subagents:
            return f"抱歉，{to_intent.value}功能暂未开放。"
        
        # 上下文必须以独立 system 消息传入，不能拼进用户消息。
        # 订单Agent 从最后一条用户消息里正则取第一个订单号，context 带历史单号时
        # 会被优先匹配：实测问 12345、context 含 12346，预取成了 12346，
        # 最终答复"订单不存在"（12345 实际存在）。
        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": (
                    f"【来自{from_agent}的任务交接】\n"
                    f"交接原因：用户的问题需要{to_intent.value}的专业知识\n"
                    f"上下文信息：{context}"
                ),
            })
        messages.append({"role": "user", "content": message})
        
        # 调用目标Agent
        target_agent = self.subagents[to_intent]
        if hasattr(target_agent, "chat_with_history"):
            reply = target_agent.chat_with_history(messages)
        else:
            # 不支持历史消息的子Agent：只传用户问题，宁可丢上下文。
            # 拼进用户消息会污染订单号提取，丢上下文只是少点信息，
            # 拼错则直接查成另一笔订单。
            reply = target_agent.chat(message)
        
        log.info(f"[任务交接] 完成，回复长度: {len(reply)}")
        
        return reply
    
    def get_history(self):
        """获取交接历史"""
        return self.handoff_history
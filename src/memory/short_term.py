"""
短期记忆模块
管理当前会话的对话历史
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from config.settings import settings
from src.utils.logger import log

# 单条摘要的最大字符数
SUMMARY_ITEM_MAX_CHARS = 100

# 摘要最多保留的条目数。
# 只裁剪 messages 而不限制摘要的话，长期会话下注入模型的 token 仍会无限增长：
# 实测 200 轮后 messages 稳定在 10 条，摘要却占 get_messages 总字符的 99%。
DEFAULT_MAX_SUMMARY_ITEMS = 20


@dataclass
class Message:
    """消息数据结构"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self, include_metadata: bool = False) -> Dict:
        """
        转换为API需要的格式

        Args:
            include_metadata: 是否附带 metadata 与 timestamp。
                默认 False：多数对话接口只接受 role/content，多带字段会被拒绝；
                需要完整信息时显式传 True。
        """
        data = {"role": self.role, "content": self.content}

        if include_metadata:
            data["metadata"] = self.metadata
            data["timestamp"] = self.timestamp

        return data


class ShortTermMemory:
    """短期记忆管理器"""
    
    def __init__(
        self,
        max_turns: int = None,
        session_id: str = "default",
        max_summary_items: int = None
    ):
        """
        初始化短期记忆
        
        Args:
            max_turns: 最大保留对话轮数，0 表示不在 messages 中保留历史。
                注意用 is None 判断而非 or，否则传入的 0 会被静默替换成默认值
            session_id: 会话ID
            max_summary_items: 摘要最多保留的条目数
        """
        self.max_turns = (
            max_turns if max_turns is not None else settings.MAX_HISTORY_TURNS
        )
        self.max_summary_items = (
            max_summary_items
            if max_summary_items is not None
            else DEFAULT_MAX_SUMMARY_ITEMS
        )
        self.session_id = session_id
        self.messages: List[Message] = []
        self._summary_items: List[str] = []  # 被压缩掉的消息摘要条目
        self.working_memory: Dict = {}  # 工作记忆
    
    def add_message(self, role: str, content: str, **metadata):
        """添加一条消息"""
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        
        # 检查是否需要压缩
        if len(self.messages) > self.max_turns * 2:
            self._compress()
        
        # 客服消息常含手机号、收货地址等敏感信息，日志只记角色与长度，不落原文
        log.debug(f"[短期记忆] 添加消息: {role} - {len(content)} 字符")
    
    def add_user_message(self, content: str, **metadata):
        """添加用户消息"""
        self.add_message("user", content, **metadata)
    
    def add_assistant_message(self, content: str, **metadata):
        """添加助手消息"""
        self.add_message("assistant", content, **metadata)
    
    def _compress(self):
        """
        压缩早期对话为摘要
        
        保留最近max_turns轮，将更早的对话压缩成摘要。
        摘要条目数同样受 max_summary_items 限制，避免长期会话下越积越长。
        """
        # 保留最近max_turns轮对话（2条消息 = 1轮）
        keep_count = self.max_turns * 2

        if len(self.messages) <= keep_count:
            return

        # keep_count 为 0 时 -0 等于 0，self.messages[:-0] 会得到空列表，
        # 这些消息既没进摘要也没被保留、直接静默丢失，因此必须单独处理
        if keep_count > 0:
            to_compress = self.messages[:-keep_count]
        else:
            to_compress = list(self.messages)

        # 生成摘要（简化版：拼接关键信息）
        for msg in to_compress:
            self._summary_items.append(
                f"{msg.role}: {msg.content[:SUMMARY_ITEM_MAX_CHARS]}"
            )

        # 摘要只保留最近若干条，防止长期会话下摘要无限增长
        if self.max_summary_items <= 0:
            self._summary_items = []
        elif len(self._summary_items) > self.max_summary_items:
            self._summary_items = self._summary_items[-self.max_summary_items:]

        # 保留最近的消息。同理 -0 会让 [-keep_count:] 返回全部消息而非清空
        if keep_count > 0:
            self.messages = self.messages[-keep_count:]
        else:
            self.messages = []

        log.info(
            f"[短期记忆] 压缩完成，保留 {len(self.messages)} 条消息，"
            f"摘要 {len(self._summary_items)} 条"
        )
    
    @property
    def summary(self) -> Optional[str]:
        """早期对话摘要，无摘要时为 None"""
        if not self._summary_items:
            return None

        return " | ".join(self._summary_items)

    def get_messages(self) -> List[Dict]:
        """
        获取消息列表（用于API调用）
        
        Returns:
            消息字典列表
        """
        result = []
        
        # 添加摘要（如果有）
        if self.summary:
            result.append({
                "role": "system",
                "content": f"[历史对话摘要] {self.summary}"
            })
        
        # 添加最近的消息
        for msg in self.messages:
            result.append(msg.to_dict())
        
        return result
    
    def set_working(self, key: str, value):
        """设置工作记忆"""
        self.working_memory[key] = value
        # 同 add_message：工作记忆可能存手机号、地址等，日志不落具体值
        log.debug(f"[工作记忆] 设置 {key}")
    
    def get_working(self, key: str, default=None):
        """获取工作记忆"""
        return self.working_memory.get(key, default)
    
    def clear(self):
        """清空记忆"""
        self.messages = []
        self._summary_items = []
        self.working_memory = {}
        log.info(f"[短期记忆] 会话 {self.session_id} 已清空")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "has_summary": self.summary is not None,
            "summary_items": len(self._summary_items),
            "summary_chars": len(self.summary) if self.summary else 0,
            "working_keys": list(self.working_memory.keys()),
            "max_turns": self.max_turns,
        }
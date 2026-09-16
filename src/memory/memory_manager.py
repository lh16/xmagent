"""
统一记忆管理器
整合短期记忆和长期记忆
"""

import re
from typing import Dict, List

from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory, EMPTY_CONTEXT
from src.utils.logger import log

# 姓名提取的边界：名字后必须紧跟标点或句尾。
# 没有这个约束时，"我是来问订单的" 会被提取出姓名"来问订单"
_NAME_BOUNDARY = r"(?=[，。！？,.\s]|$)"


class MemoryManager:
    """统一记忆管理器"""
    
    def __init__(self, session_id: str = "default"):
        """
        初始化记忆管理器
        
        Args:
            session_id: 会话ID
        """
        self.session_id = session_id
        self.short_term = ShortTermMemory(session_id=session_id)
        self.long_term = LongTermMemory()
        
        log.info(f"记忆管理器初始化完成，会话ID: {session_id}")
    
    # ============================================================
    # 对话消息
    # ============================================================
    
    def add_user_message(self, content: str, user_id: str = None):
        """添加用户消息"""
        self.short_term.add_user_message(content, user_id=user_id)
    
    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.short_term.add_assistant_message(content)
    
    def get_messages(self) -> List[Dict]:
        """获取消息列表"""
        return self.short_term.get_messages()
    
    # ============================================================
    # 记忆检索
    # ============================================================
    
    def build_context(self, user_id: str) -> str:
        """
        构建完整的记忆上下文
        
        Args:
            user_id: 用户ID
        
        Returns:
            格式化的上下文；无任何记忆时返回空字符串
        """
        # 第二参是最近交互的条数，不是查询文本。
        # 传成文本会落到 SQL 的 LIMIT 上，直接 datatype mismatch
        long_term_context = self.long_term.retrieve_context(user_id)
        
        # 获取工作记忆
        working_memory = self.short_term.working_memory
        working_str = ""
        if working_memory:
            working_str = "【当前工作记忆】\n" + "\n".join([
                f"- {k}: {v}" for k, v in working_memory.items()
            ])
        
        # 拼接
        parts = []
        # 无记忆时长期记忆返回的是兜底文案而非空串，不能当成有效内容拼进去
        if long_term_context and long_term_context != EMPTY_CONTEXT:
            parts.append(long_term_context)
        if working_str:
            parts.append(working_str)
        
        if not parts:
            return ""
        
        return "\n\n".join(parts)
    
    # ============================================================
    # 记忆存储
    # ============================================================
    
    def remember_user(self, user_id: str, name: str = None, 
                      preferences: Dict = None):
        """记住用户信息"""
        self.long_term.save_user(user_id, name, preferences)
    
    def remember_order(self, user_id: str, order_id: str, 
                       product_name: str = None):
        """
        记住订单号
        
        请只在订单确实存在时调用。此前从对话文本里盲提订单号，
        查无此单的单号也会被永久记住，后续模型据此认定该订单存在。
        """
        self.long_term.remember_order(user_id, order_id, product_name)
        # 同时写入工作记忆
        self.short_term.set_working("current_order_id", order_id)
    
    def remember_fact(self, user_id: str, fact_type: str, 
                      content: str):
        """记住用户事实"""
        self.long_term.save_fact(user_id, fact_type, content)
    
    def save_interaction(self, user_id: str, question: str, 
                         intent: str = None, answer: str = None):
        """保存交互记录"""
        self.long_term.save_interaction(user_id, question, intent, answer)
    
    # ============================================================
    # 记忆提取（从对话中）
    # ============================================================
    
    def extract_and_remember(self, user_id: str, conversation: str):
        """
        从对话中提取关键信息并存储
        
        这是简化的规则提取版本，实际项目中可以用LLM提取。
        只提取姓名与偏好：订单号不做文本提取，由调用方在订单查询成功后
        显式调用 remember_order，避免把查无此单的单号记成历史订单。
        
        Args:
            user_id: 用户ID
            conversation: 对话文本
        """
        # 1. 提取姓名（"我叫XXX"、"我是XXX"）
        name_patterns = [
            r"我叫([^\s，。,\.]{2,4})" + _NAME_BOUNDARY,
            r"我是([^\s，。,\.]{2,4})" + _NAME_BOUNDARY,
            r"我的名字是([^\s，。,\.]{2,4})" + _NAME_BOUNDARY,
        ]
        for pattern in name_patterns:
            match = re.search(pattern, conversation)
            if match:
                name = match.group(1)
                self.long_term.save_user(user_id, name=name)
                self.short_term.set_working("user_name", name)
                # 姓名属个人信息，日志不落具体内容
                log.info(f"[记忆提取] 记住用户姓名（{len(name)} 字）")
                break
        
        # 2. 提取偏好（"我喜欢XXX"、"我预算XXX"）
        pref_patterns = [
            (r"我(?:喜欢|偏好)([^\s，。,\.]{2,10})", "preference"),
            (r"我(?:的)?预算[是为:]?(\d+[-到]?\d*)", "budget"),
        ]
        for pattern, pref_type in pref_patterns:
            matches = re.findall(pattern, conversation)
            for value in matches:
                self.long_term.save_fact(
                    user_id, pref_type, 
                    f"用户{pref_type}: {value}"
                )
                log.info(f"[记忆提取] 记住偏好: {pref_type}={value}")
    
    # ============================================================
    # 统计
    # ============================================================
    
    def get_stats(self, user_id: str = None) -> Dict:
        """
        获取记忆统计
        
        Args:
            user_id: 提供时一并统计该用户的长期记忆规模
        """
        stats = {
            "short_term": self.short_term.get_stats(),
            "session_id": self.session_id,
        }
        
        if user_id:
            stats["long_term"] = {
                "user_id": user_id,
                "has_profile": self.long_term.get_user(user_id) is not None,
                "orders": len(self.long_term.get_user_orders(user_id)),
                "facts": len(self.long_term.get_user_facts(user_id)),
            }
        
        return stats
    
    def clear(self, user_id: str = None):
        """
        清空短期记忆
        
        Args:
            user_id: 提供时一并删除该用户的全部长期记忆，
                用于用户要求清理个人信息的场景
        """
        self.short_term.clear()
        
        if user_id:
            self.long_term.delete_user(user_id)

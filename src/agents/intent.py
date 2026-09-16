"""
意图识别模块
判断用户消息的意图类型
"""

import re
from enum import Enum
from typing import Any, Dict, Optional
from src.utils.model import create_model
from src.utils.logger import log


class Intent(str, Enum):
    """意图类型枚举"""
    PRODUCT_INQUIRY = "产品咨询"      # 商品功能、规格、价格
    ORDER_QUERY = "订单查询"          # 订单状态、物流
    AFTER_SALES = "售后政策"          # 退换货、退款、保修
    CHITCHAT = "闲聊"                 # 问候、感谢、告别
    TRANSFER_HUMAN = "转人工"         # 投诉、复杂问题


# 意图识别Prompt
INTENT_PROMPT = """请判断以下用户消息的意图，从以下类别中选择一个：

- 产品咨询：询问商品功能、规格、价格、库存、推荐
- 订单查询：查询订单状态、物流进度、预计送达时间
- 售后政策：退换货、退款、保修、维修
- 闲聊：问候、感谢、告别、无意义的对话
- 转人工：投诉、建议、情绪激动、复杂问题

用户消息：{message}

请只输出意图类别名称（产品咨询/订单查询/售后政策/闲聊/转人工），不要输出其他内容。
"""

# 模型返回的类别名 → 枚举（提到模块级，避免每次分类重复构造）
INTENT_MAP = {
    "产品咨询": Intent.PRODUCT_INQUIRY,
    "订单查询": Intent.ORDER_QUERY,
    "售后政策": Intent.AFTER_SALES,
    "闲聊": Intent.CHITCHAT,
    "转人工": Intent.TRANSFER_HUMAN,
}

# 订单号特征：5 位以上连续数字
ORDER_ID_PATTERN = re.compile(r"\d{5,}")


class IntentClassifier:
    """意图分类器"""
    
    def __init__(self):
        """初始化分类器"""
        self.model = create_model(temperature=0.0)
        
        # 规则优先：关键词映射（未命中时才交给LLM兜底）
        # 声明顺序即优先级，转人工放在最前
        self.keyword_rules = {
            Intent.TRANSFER_HUMAN: [
                "投诉", "人工", "客服", "经理", "骗子", "垃圾"
            ],
            Intent.ORDER_QUERY: [
                "订单", "物流", "快递", "发货", "到哪", "运单",
                "什么时候到", "预计送达"
            ],
            Intent.AFTER_SALES: [
                "退货", "退款", "换货", "保修", "维修", "售后",
                "七天无理由", "三包"
            ],
            Intent.PRODUCT_INQUIRY: [
                "多少钱", "价格", "规格", "参数", "推荐", "有什么",
                "哪个好", "区别", "配置", "颜色", "内存"
            ],
        }
    
    def classify(self, message: str) -> Intent:
        """
        识别用户意图

        Args:
            message: 用户消息

        Returns:
            意图类型
        """
        return self.classify_with_confidence(message)["intent"]
    
    def _rule_based_classify(self, message: str) -> Optional[Intent]:
        """基于关键词的规则匹配"""
        message_lower = message.lower()

        # 1. 转人工最优先：用户情绪激动或明确要求人工时，不应再走业务流程
        if any(k in message_lower for k in self.keyword_rules[Intent.TRANSFER_HUMAN]):
            return Intent.TRANSFER_HUMAN

        # 2. 订单号特征：5 位以上连续数字。
        #    原先硬编码 "12345" 等种子单号，既认不全，也会把价格等数字误判成订单
        if ORDER_ID_PATTERN.search(message):
            return Intent.ORDER_QUERY

        # 3. 其余按 keyword_rules 的声明顺序匹配
        for intent, keywords in self.keyword_rules.items():
            if intent is Intent.TRANSFER_HUMAN:
                continue
            if any(keyword in message_lower for keyword in keywords):
                return intent

        return None

    def _llm_classify(self, message: str) -> Intent:
        """使用LLM分类（规则未命中时的兜底）"""
        prompt = INTENT_PROMPT.format(message=message)

        try:
            response = self.model.invoke(prompt)
        except Exception as e:
            # 分类失败不能中断对话，降级为闲聊交给主控处理
            log.error(f"[意图识别] LLM分类失败，降级为闲聊: {e}")
            return Intent.CHITCHAT

        intent_text = response.content.strip()

        # 模型可能返回"产品咨询。"或"意图：产品咨询"，用包含匹配而非全等
        for name, intent in INTENT_MAP.items():
            if name in intent_text:
                return intent

        log.warning(f"[意图识别] 无法解析的LLM输出，降级为闲聊: {intent_text}")
        return Intent.CHITCHAT

    def classify_with_confidence(self, message: str) -> Dict[str, Any]:
        """
        带置信度的意图识别

        Returns:
            {"intent": Intent, "confidence": float, "method": str}
        """
        rule_intent = self._rule_based_classify(message)
        if rule_intent:
            log.info(f"[意图识别] 规则匹配: {message} → {rule_intent}")
            return {
                "intent": rule_intent,
                "confidence": 0.95,
                "method": "rule"
            }

        llm_intent = self._llm_classify(message)
        log.info(f"[意图识别] LLM分类: {message} → {llm_intent}")
        return {
            "intent": llm_intent,
            "confidence": 0.8,
            "method": "llm"
        }


# 全局分类器实例
intent_classifier = IntentClassifier()
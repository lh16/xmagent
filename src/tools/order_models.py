"""
订单数据模型
定义订单相关的数据结构
"""

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from src.utils.logger import log


class OrderItem(BaseModel):
    """订单商品项"""
    product_id: str = Field(description="商品ID")
    product_name: str = Field(description="商品名称")
    quantity: int = Field(description="数量")
    price: float = Field(description="单价")
    subtotal: float = Field(description="小计")


class LogisticsInfo(BaseModel):
    """物流信息"""
    company: str = Field(description="快递公司")
    tracking_number: str = Field(description="运单号")
    status: str = Field(description="物流状态")
    current_location: Optional[str] = Field(default=None, description="当前位置")
    estimated_delivery: Optional[str] = Field(default=None, description="预计送达时间")
    updates: List[dict] = Field(default_factory=list, description="物流轨迹")


class Order(BaseModel):
    """订单模型"""
    order_id: str = Field(description="订单号")
    user_id: str = Field(description="用户ID")
    status: str = Field(description="订单状态")
    items: List[OrderItem] = Field(description="商品列表")
    total_amount: float = Field(description="订单总额")
    # 时间统一用字符串，格式约定为 "YYYY-MM-DD HH:MM:SS"，便于直接展示给用户
    created_at: str = Field(description="下单时间（YYYY-MM-DD HH:MM:SS）")
    paid_at: Optional[str] = Field(default=None, description="支付时间（YYYY-MM-DD HH:MM:SS）")
    shipped_at: Optional[str] = Field(default=None, description="发货时间（YYYY-MM-DD HH:MM:SS）")
    logistics: Optional[LogisticsInfo] = Field(default=None, description="物流信息")
    shipping_address: str = Field(description="收货地址")

    @model_validator(mode="after")
    def _check_amounts(self):
        """补齐商品小计，并校验订单总额与商品小计之和是否一致

        只告警不阻断：订单总额可能包含运费、优惠等，与商品小计之和存在差额属正常。
        """
        for item in self.items:
            if not item.subtotal:
                item.subtotal = round(item.price * item.quantity, 2)

        items_total = round(sum(item.subtotal for item in self.items), 2)
        if abs(items_total - self.total_amount) > 0.01:
            log.warning(
                f"订单 {self.order_id} 总额({self.total_amount})与商品小计之和({items_total})不一致"
            )
        return self
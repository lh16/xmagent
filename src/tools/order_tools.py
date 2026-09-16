"""
订单查询工具
提供给Agent调用的工具函数
"""

from typing import List
from src.tools.order_database import order_db
from src.tools.order_models import LogisticsInfo, Order
from src.utils.logger import log


def query_order(order_id: str) -> str:
    """
    查询订单状态和物流信息。

    当用户询问订单状态、物流进度、预计送达时间时，使用此工具。
    需要用户提供订单号才能查询。

    Args:
        order_id: 订单号，如"12345"

    Returns:
        订单详情的格式化字符串
    """
    log.info(f"[订单查询] 查询订单: {order_id}")

    try:
        order = order_db.get_order(order_id)
    except Exception as e:
        # 查询异常不能中断Agent，返回友好提示由模型转述给用户
        log.error(f"[订单查询] 查询失败: {order_id}, {e}")
        return "订单查询失败，请稍后重试或联系人工客服。"

    if not order:
        return f"未找到订单号为 {order_id} 的订单。请确认订单号是否正确。"

    log.info(f"[订单查询] 查询成功: {order_id}")
    return format_order_info(order)


def query_user_orders(user_id: str, limit: int = 5) -> str:
    """
    查询用户的最近订单列表。

    当用户询问"我的订单"、"我最近买了什么"时，使用此工具。

    Args:
        user_id: 用户ID
        limit: 返回的订单数量

    Returns:
        订单列表的格式化字符串
    """
    log.info(f"[订单查询] 查询用户订单: {user_id}")

    try:
        orders = order_db.get_user_orders(user_id, limit)
    except Exception as e:
        log.error(f"[订单查询] 查询失败: {user_id}, {e}")
        return "订单查询失败，请稍后重试或联系人工客服。"

    if not orders:
        return f"用户 {user_id} 暂无订单记录。"

    lines = [f"您最近有 {len(orders)} 个订单：\n"]
    for i, order in enumerate(orders, 1):
        items_desc = "、".join(
            f"{item.product_name} × {item.quantity}" for item in order.items
        )
        lines.append(
            f"{i}. 订单号：{order.order_id}\n"
            f"   商品：{items_desc}\n"
            f"   状态：{order.status}\n"
            f"   金额：¥{order.total_amount:.2f}\n"
            f"   下单时间：{order.created_at}\n"
        )

    return "\n".join(lines)


def get_logistics(order_id: str) -> str:
    """
    查询订单的物流信息。

    当用户专门询问物流进度、快递状态时，使用此工具。

    Args:
        order_id: 订单号

    Returns:
        物流信息的格式化字符串
    """
    log.info(f"[物流查询] 查询物流: {order_id}")

    try:
        order = order_db.get_order(order_id)
    except Exception as e:
        log.error(f"[物流查询] 查询失败: {order_id}, {e}")
        return "物流查询失败，请稍后重试或联系人工客服。"

    if not order:
        return f"未找到订单号为 {order_id} 的订单。"

    if not order.logistics:
        return f"订单 {order_id} 暂无物流信息，可能还未发货。当前状态：{order.status}"

    return "\n".join(_format_logistics(order.logistics, with_order_id=order_id))


def _format_logistics(logistics: LogisticsInfo, with_order_id: str = None) -> List[str]:
    """格式化物流信息（订单详情与物流查询共用）

    Args:
        logistics: 物流信息模型
        with_order_id: 传入时额外输出订单号

    Returns:
        按行拆分的物流信息
    """
    lines = ["", "🚚 物流信息", ""]
    if with_order_id:
        lines.append(f"订单号：{with_order_id}")

    lines.extend([
        f"快递公司：{logistics.company}",
        f"运单号：{logistics.tracking_number}",
        f"物流状态：{logistics.status}",
    ])

    if logistics.current_location:
        lines.append(f"当前位置：{logistics.current_location}")

    if logistics.estimated_delivery:
        lines.append(f"预计送达：{logistics.estimated_delivery}")

    if logistics.updates:
        lines.extend(["", "物流轨迹："])
        # 轨迹元素形如 {"time": "2024-01-01 10:00", "desc": "已揽收"}
        lines.extend(f"  {u['time']}  {u['desc']}" for u in logistics.updates)

    return lines


def format_order_info(order: Order) -> str:
    """格式化订单信息"""
    items_desc = "\n".join(
        f"  - {item.product_name} × {item.quantity} （¥{item.price:.2f}）"
        for item in order.items
    )

    lines = [
        "📦 订单信息",
        "",
        f"订单号：{order.order_id}",
        f"订单状态：{order.status}",
        f"下单时间：{order.created_at}",
        f"订单金额：¥{order.total_amount:.2f}",
        "",
        "商品清单：",
        items_desc,
        "",
        f"收货地址：{order.shipping_address}",
    ]

    if order.logistics:
        lines.extend(_format_logistics(order.logistics))

    return "\n".join(lines)

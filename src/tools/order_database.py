"""
订单数据库
使用SQLite模拟真实订单系统
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional, List
from config.settings import settings
from src.tools.order_models import Order
from src.utils.logger import log


class OrderDatabase:
    """订单数据库管理"""
    
    def __init__(self, db_path: str = None):
        """初始化数据库连接"""
        self.db_path = db_path or settings.SQLITE_DB_PATH
        # data/ 目录通常不入库，SQLite 又不会自动创建目录，
        # 缺目录会让"import 本模块"直接崩溃，这里提前建好。
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_database()
        self._seed_data()

    def _connect(self) -> sqlite3.Connection:
        """建立数据库连接（统一设置 row_factory，便于按列名取值）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """初始化数据库表"""
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    items TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    paid_at TEXT,
                    shipped_at TEXT,
                    logistics TEXT,
                    shipping_address TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()
        log.info("订单数据库初始化完成")
    
    def _seed_data(self):
        """填充模拟数据（仅当数据库为空时）"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM orders")
        count = cursor.fetchone()[0]

        if count > 0:
            conn.close()
            return
        
        # 模拟订单数据
        now = datetime.now()
        
        orders = [
            {
                "order_id": "12345",
                "user_id": "user_001",
                "status": "已发货",
                "items": json.dumps([
                    {"product_id": "P001", "product_name": "星辰X10 Pro 12GB+256GB",
                     "quantity": 1, "price": 4999.0, "subtotal": 4999.0}
                ], ensure_ascii=False),
                "total_amount": 4999.0,
                "created_at": (now - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"),
                "paid_at": (now - timedelta(days=3) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "shipped_at": (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "logistics": json.dumps({
                    "company": "顺丰速运",
                    "tracking_number": "SF1234567890",
                    "status": "运输中",
                    "current_location": "北京分拨中心",
                    "estimated_delivery": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "updates": [
                        {"time": (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
                         "desc": "已揽收"},
                        {"time": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
                         "desc": "到达北京分拨中心"},
                    ]
                }, ensure_ascii=False),
                "shipping_address": "北京市朝阳区XX路XX号"
            },
            {
                "order_id": "12346",
                "user_id": "user_001",
                "status": "待发货",
                "items": json.dumps([
                    {"product_id": "P002", "product_name": "星辰Watch Pro",
                     "quantity": 1, "price": 1299.0, "subtotal": 1299.0}
                ], ensure_ascii=False),
                "total_amount": 1299.0,
                "created_at": (now - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                "paid_at": (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "shipped_at": None,
                "logistics": None,
                "shipping_address": "北京市朝阳区XX路XX号"
            },
            {
                "order_id": "12347",
                "user_id": "user_002",
                "status": "已完成",
                "items": json.dumps([
                    {"product_id": "P003", "product_name": "星辰Buds Pro",
                     "quantity": 2, "price": 899.0, "subtotal": 1798.0}
                ], ensure_ascii=False),
                "total_amount": 1798.0,
                "created_at": (now - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"),
                "paid_at": (now - timedelta(days=10) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "shipped_at": (now - timedelta(days=9)).strftime("%Y-%m-%d %H:%M:%S"),
                "logistics": json.dumps({
                    "company": "京东物流",
                    "tracking_number": "JD9876543210",
                    "status": "已签收",
                    "current_location": "已签收",
                    "estimated_delivery": None,
                    "updates": [
                        {"time": (now - timedelta(days=9)).strftime("%Y-%m-%d %H:%M"),
                         "desc": "已揽收"},
                        {"time": (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M"),
                         "desc": "已签收"},
                    ]
                }, ensure_ascii=False),
                "shipping_address": "上海市浦东新区XX路XX号"
            },
            {
                "order_id": "12348",
                "user_id": "user_002",
                "status": "待付款",
                "items": json.dumps([
                    {"product_id": "P004", "product_name": "星辰A5 6GB+128GB",
                     "quantity": 1, "price": 1499.0, "subtotal": 1499.0}
                ], ensure_ascii=False),
                "total_amount": 1499.0,
                "created_at": (now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                "paid_at": None,
                "shipped_at": None,
                "logistics": None,
                "shipping_address": "广州市天河区XX路XX号"
            },
        ]
        
        try:
            for order in orders:
                cursor.execute("""
                    INSERT INTO orders (
                        order_id, user_id, status, items, total_amount,
                        created_at, paid_at, shipped_at, logistics, shipping_address
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order["order_id"], order["user_id"], order["status"],
                    order["items"], order["total_amount"], order["created_at"],
                    order["paid_at"], order["shipped_at"], order["logistics"],
                    order["shipping_address"]
                ))
            conn.commit()
        finally:
            conn.close()
        log.info(f"填充了 {len(orders)} 条模拟订单数据")
    
    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> Order:
        """把数据库行转换为 Order 模型（items / logistics 以 JSON 字符串存储）"""
        data = dict(row)
        data["items"] = json.loads(data["items"])
        if data["logistics"]:
            data["logistics"] = json.loads(data["logistics"])
        return Order(**data)

    def get_order(self, order_id: str) -> Optional[Order]:
        """按订单号查询订单"""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()
        finally:
            conn.close()

        if not row:
            return None
        return self._row_to_order(row)

    def get_user_orders(self, user_id: str, limit: int = 5) -> List[Order]:
        """查询用户的订单列表（按下单时间倒序）"""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        return [self._row_to_order(row) for row in rows]


# 全局数据库实例
order_db = OrderDatabase()
"""
长期记忆模块
跨会话保存用户画像、历史订单、用户事实与交互记录，基于 SQLite 持久化。

与短期记忆的区别：短期记忆只保留当前会话最近若干轮（进程内、随会话结束丢弃），
长期记忆按 user_id 沉淀，下次用户再来时仍可取用。
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import settings
from src.utils.logger import log

# 建表语句。orders 与 facts 均带唯一约束：
# 同一订单重复记住只更新不叠加，同一事实重复保存不产生冗余记录
_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    name        TEXT,
    preferences TEXT,
    created_at  TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    order_id     TEXT NOT NULL,
    product_name TEXT,
    created_at   TEXT,
    UNIQUE(user_id, order_id)
);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    category   TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT,
    UNIQUE(user_id, category, content)
);

CREATE TABLE IF NOT EXISTS interactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    question   TEXT,
    intent     TEXT,
    answer     TEXT,
    created_at TEXT
);
"""

# 检索上下文时带回的最近交互条数
DEFAULT_INTERACTION_LIMIT = 3

# 用户无任何记忆时的兜底文案
EMPTY_CONTEXT = "暂无该用户的历史记忆。"


def _now() -> str:
    """当前时间字符串"""
    return datetime.now().isoformat(timespec="seconds")


class LongTermMemory:
    """长期记忆管理器（按 user_id 维度持久化）"""

    def __init__(self, db_path: str = None):
        """
        初始化长期记忆

        Args:
            db_path: SQLite 文件路径，默认取 settings.LONG_TERM_DB_PATH。
                测试请用临时目录建库，不要用 ":memory:"：
                每次操作都会新建连接，而内存库每次连接都是新的空库，写入会立即丢失
        """
        self.db_path = db_path or settings.LONG_TERM_DB_PATH

    @contextmanager
    def _get_conn(self):
        """获取数据库连接，用完自动提交并关闭"""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        """建立连接并确保建表"""
        if self.db_path != ":memory:":
            directory = os.path.dirname(self.db_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(_CREATE_TABLES)
        return conn

    def save_user(
        self,
        user_id: str,
        name: str = None,
        preferences: Dict = None
    ):
        """
        保存用户画像

        偏好按 key 与已有值合并，重复调用不会覆盖此前记下的偏好；
        name 为 None 时保留原名。

        Args:
            user_id: 用户ID
            name: 用户姓名
            preferences: 偏好字典，如 {"预算": "3000-5000"}
        """
        now = _now()

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT preferences FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()

            merged = {}
            if row and row["preferences"]:
                merged = json.loads(row["preferences"])
            merged.update(preferences or {})

            conn.execute(
                """
                INSERT INTO users (user_id, name, preferences, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name = COALESCE(excluded.name, users.name),
                    preferences = excluded.preferences,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    name,
                    json.dumps(merged, ensure_ascii=False),
                    now,
                    now,
                ),
            )

        # 用户画像含姓名等个人信息，日志不记录具体内容
        log.info(f"[长期记忆] 保存用户 {user_id} 的画像")

    def remember_order(self, user_id: str, order_id: str, product_name: str = None):
        """
        记住用户的订单

        同一 (user_id, order_id) 重复记录时只更新商品名，不产生重复条目。

        Args:
            user_id: 用户ID
            order_id: 订单号
            product_name: 商品名称
        """
        now = _now()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO orders (user_id, order_id, product_name, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, order_id) DO UPDATE SET
                    product_name = COALESCE(excluded.product_name, orders.product_name),
                    created_at = excluded.created_at
                """,
                (user_id, order_id, product_name, now),
            )

        log.info(f"[长期记忆] 用户 {user_id} 记住订单 {order_id}")

    def save_fact(self, user_id: str, category: str, content: str):
        """
        保存一条用户事实

        相同 (user_id, category, content) 只会保留一条，避免多轮对话反复写入。

        Args:
            user_id: 用户ID
            category: 事实类别，如 preference / context
            content: 事实内容
        """
        now = _now()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO facts (user_id, category, content, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, category, content) DO NOTHING
                """,
                (user_id, category, content, now),
            )

        # 事实内容可能含地址等敏感信息，日志不记录内容
        log.info(f"[长期记忆] 用户 {user_id} 保存事实（{category}）")

    def save_interaction(
        self,
        user_id: str,
        question: str,
        intent: str = None,
        answer: str = None
    ):
        """
        保存一次交互记录

        Args:
            user_id: 用户ID
            question: 用户问题
            intent: 命中的意图
            answer: 助手回复
        """
        now = _now()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO interactions (user_id, question, intent, answer, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, question, intent, answer, now),
            )

        log.info(f"[长期记忆] 用户 {user_id} 记录一次交互")

    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        获取用户画像

        Returns:
            画像字典，用户不存在时返回 None
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()

        if row is None:
            return None

        preferences = json.loads(row["preferences"]) if row["preferences"] else {}

        return {
            "user_id": row["user_id"],
            "name": row["name"],
            "preferences": preferences,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_user_orders(self, user_id: str) -> List[Dict]:
        """获取用户历史订单，按记录顺序返回"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT order_id, product_name, created_at FROM orders
                WHERE user_id = ? ORDER BY id
                """,
                (user_id,),
            ).fetchall()

        return [
            {
                "order_id": row["order_id"],
                "product_name": row["product_name"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_user_facts(self, user_id: str, category: str = None) -> List[Dict]:
        """
        获取用户事实

        Args:
            user_id: 用户ID
            category: 指定时只返回该类别的事实

        Returns:
            事实列表
        """
        with self._get_conn() as conn:
            if category:
                rows = conn.execute(
                    """
                    SELECT category, content, created_at FROM facts
                    WHERE user_id = ? AND category = ? ORDER BY id
                    """,
                    (user_id, category),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT category, content, created_at FROM facts
                    WHERE user_id = ? ORDER BY id
                    """,
                    (user_id,),
                ).fetchall()

        return [
            {
                "category": row["category"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_recent_interactions(
        self,
        user_id: str,
        limit: int = DEFAULT_INTERACTION_LIMIT
    ) -> List[Dict]:
        """获取最近若干次交互，按时间倒序"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT question, intent, answer, created_at FROM interactions
                WHERE user_id = ? ORDER BY id DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        return [
            {
                "question": row["question"],
                "intent": row["intent"],
                "answer": row["answer"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def retrieve_context(
        self,
        user_id: str,
        interaction_limit: int = DEFAULT_INTERACTION_LIMIT
    ) -> str:
        """
        检索用户记忆，拼成可直接注入提示词的上下文

        Args:
            user_id: 用户ID
            interaction_limit: 带回的最近交互条数

        Returns:
            上下文文本，无任何记忆时返回 EMPTY_CONTEXT
        """
        user = self.get_user(user_id)
        orders = self.get_user_orders(user_id)
        facts = self.get_user_facts(user_id)
        interactions = self.get_recent_interactions(user_id, interaction_limit)

        if not user and not orders and not facts and not interactions:
            return EMPTY_CONTEXT

        sections = []

        if user:
            lines = []
            if user["name"]:
                lines.append(f"姓名：{user['name']}")
            if user["preferences"]:
                prefs = "；".join(
                    f"{key}: {value}" for key, value in user["preferences"].items()
                )
                lines.append(f"偏好：{prefs}")

            if lines:
                sections.append("[用户画像]\n" + "\n".join(lines))

        if orders:
            lines = [
                f"- {order['order_id']}"
                + (f"（{order['product_name']}）" if order["product_name"] else "")
                for order in orders
            ]
            sections.append("[历史订单]\n" + "\n".join(lines))

        if facts:
            lines = [f"- {fact['category']}: {fact['content']}" for fact in facts]
            sections.append("[已知事实]\n" + "\n".join(lines))

        if interactions:
            lines = []
            # 取出时是倒序，展示前翻回正序，便于模型按时间理解
            for item in reversed(interactions):
                text = f"- 用户问：{item['question']}"
                if item["intent"]:
                    text += f"（{item['intent']}）"
                lines.append(text)

            sections.append("[最近交互]\n" + "\n".join(lines))

        log.info(f"[长期记忆] 检索用户 {user_id} 的上下文")

        return "\n\n".join(sections)

    def delete_user(self, user_id: str):
        """
        删除该用户的全部长期记忆

        用户要求清理个人信息时应可调，四个表一并清除。
        """
        with self._get_conn() as conn:
            for table in ("users", "orders", "facts", "interactions"):
                conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))

        log.info(f"[长期记忆] 已删除用户 {user_id} 的全部记忆")

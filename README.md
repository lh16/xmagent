# xmagent

基于 LangChain / deepagents 的多 Agent 智能客服系统。

一个主控 Agent 按意图把问题分派给子 Agent：产品咨询和售后走 RAG 知识库，订单查询走订单库，投诉转人工生成工单。带短期 + 长期双层记忆，支持跨会话记住用户。

> 项目定位是学习用的 demo：订单数据是本地 mock，模型走 SiliconFlow 的 OpenAI 兼容接口。

## 特性

- **意图路由**：规则优先、LLM 兜底，命中 5 类意图（产品咨询 / 订单查询 / 售后政策 / 转人工 / 闲聊）
- **三个子 Agent**：产品咨询、订单查询、售后政策，各自独立提示词与工具
- **任务交接**：订单 + 售后同时命中时，先查单再携带订单上下文交接给售后
- **混合检索**：BM25（jieba 分词）与向量检索融合（权重 0.4 / 0.6），再用 reranker 重排
- **双层记忆**：短期按会话隔离并自动压缩摘要，长期落 SQLite 跨会话保留画像、订单、事实
- **防编造**：订单字段只能来自工具返回；查无此单直接返回工具原文，不让模型补全

## 技术栈

Python 3.12+ · LangChain · Chroma · SQLite · loguru · uv（依赖管理）

模型与嵌入走 SiliconFlow（OpenAI 兼容接口），因此不需要本地显卡。

## 目录结构

```
xmagent/
├── config/
│   └── settings.py              # 全局配置，import 即自动加载根目录 .env
├── data/
│   ├── knowledge/               # 知识库源文档：product_manual.md / faq.md / after_sales.md
│   ├── chroma_db/               # Chroma 向量库持久化目录
│   ├── orders.db                # 订单库，首次运行自动播种 12345~12348
│   └── long_term_memory.db      # 长期记忆库
├── examples/                    # deepagents 官方示例，与客服系统无关
├── scripts/                     # 构建脚本与测试脚本
├── src/
│   ├── agents/                  # 主控、三个子 Agent、意图识别、交接管理
│   ├── memory/                  # 短期 / 长期记忆
│   ├── rag/                     # 加载、切分、向量库、混合检索、重排
│   ├── tools/                   # 订单工具、RAG 工具
│   ├── utils/                   # 日志、模型工厂
│   └── chat.py                  # 极简交互式对话（不接 RAG，闲聊兜底）
├── .env.example
├── pyproject.toml
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

或用 pip：

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入密钥：

```bash
SILICONFLOW_API_KEY=你的密钥
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

> 注意：`.env.example` 里写的是 `OPENAI_API_KEY`，但 `config/settings.py` 实际读的是 `SILICONFLOW_API_KEY`，照抄示例文件会拿不到密钥。

完整可配置项见 [配置说明](#配置说明)。

### 3. 构建知识库

```bash
uv run python scripts/build_knowledge_base.py
```

首次运行会调用云端嵌入模型，需要密钥且耗时较久。之后向量库持久化在 `data/chroma_db/`，重复运行会复用。

也可以跳过这步：直接跑 `scripts/build_full_system.py`，它检测到空库会自动构建。

### 4. 开始对话

```bash
uv run python scripts/run_customer_service.py
```

输入 `quit` 退出。

## 运行方式

| 入口 | 说明 |
|---|---|
| `uv run python scripts/run_customer_service.py` | 交互式对话，接产品咨询 + 订单查询两个子 Agent |
| `uv run python -m src.chat` | 极简交互式对话，不接 RAG 和子 Agent，只有闲聊兜底 |
| `uv run python scripts/build_full_system.py` | 构建完整系统（三子 Agent + 主控），供代码调用 |

三个子 Agent 全接入的交互式入口目前没有，需要在代码里调用：

```python
from scripts.build_full_system import build_system

agent = build_system(session_id="demo")
print(agent.chat("我的订单12345到哪了？", user_id="user_001"))
```

`scripts/run_customer_service.py` 只装配了产品和订单两个子 Agent，且向量库为空时会直接报错退出；`build_full_system.py` 则会把空库自动建起来。两者共用同一个 `product_knowledge` 向量库。

## 架构

```
用户提问
   │
   ├─ 意图识别（规则优先，未命中走 LLM）
   │
   └─ 主控路由 CustomerServiceAgent
        ├─ PRODUCT_INQUIRY  → 产品咨询子 Agent（RAG 检索知识库）
        ├─ ORDER_QUERY      → 订单查询子 Agent（查 orders.db）
        ├─ AFTER_SALES      → 售后政策子 Agent（RAG，与产品共用同一向量库）
        ├─ TRANSFER_HUMAN   → 生成 8 位工单号入队，返回安抚话术
        └─ CHITCHAT         → 主控直接回复
```

订单与售后关键词同时出现时，先查单拿到真实订单，再携带订单上下文交接给售后，避免售后凭空发问。

### RAG 流程

```
文档加载 → 切分（500 字 / 重叠 50）→ Chroma 向量库
        → BM25 + 向量混合检索（0.4 / 0.6）→ reranker 重排 → Top 3
```

### 记忆

| 层 | 实现 | 作用域 |
|---|---|---|
| 短期 `ShortTermMemory` | 内存中，超过 `MAX_HISTORY_TURNS * 2` 条自动压缩为摘要 | 按 session 隔离 |
| 工作记忆 | key-value，如 `current_order_id` | 按 session 隔离 |
| 长期 `LongTermMemory` | SQLite 四表：users / orders / facts / interactions | 跨会话 |

长期记忆不按 session 隔离，换会话仍能读到用户画像与历史订单——这是跨会话能力的来源。

## 配置说明

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SILICONFLOW_API_KEY` | 空 | **必填**，模型调用密钥 |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | 模型接口地址 |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | 对话模型 |
| `SILICONFLOW_EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | 云端嵌入模型 |
| `RERANKER_MODEL` / `SILICONFLOW_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排模型 |
| `KNOWLEDGE_BASE_DIR` | `./data/knowledge` | 知识库文档目录 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | 文档切分参数 |
| `RETRIEVE_TOP_K` / `RERANK_TOP_K` | `10` / `3` | 检索与重排返回条数 |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | 向量库目录 |
| `SQLITE_DB_PATH` | `./data/orders.db` | 订单库路径 |
| `LONG_TERM_DB_PATH` | `./data/long_term_memory.db` | 长期记忆库路径 |
| `MAX_HISTORY_TURNS` | `10` | 短期记忆保留轮数 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

`config/settings.py` 被 import 时会自动加载项目根目录的 `.env`，无需手动调用 `load_dotenv`。

## 测试

测试脚本都在 `scripts/` 下，均为命令行脚本，会真实调用 LLM：

```bash
uv run python scripts/test_full_system.py  # 完整端到端，断言数据层
uv run python scripts/test_full_system.py --scenario 5,6  # 只跑记忆与跨会话
uv run python scripts/test_intent.py  # 意图识别
uv run python scripts/test_order_agent.py  # 订单子 Agent 防编造
uv run python scripts/test_rag.py  # 检索链路冒烟
uv run python scripts/test_memory_manager.py  # 记忆管理器（用临时库）
```

`test_full_system.py` 覆盖 6 个场景（产品咨询 / 订单查询 / 售后 / 转人工 / 记忆 / 跨会话），断言 5 项数据层结果：记住姓名、记住预算、查到的订单写入长期记忆、转人工工单入队、切会话后仍读到长期记忆。失败以非 0 退出，便于 CI 发现。

调试记忆相关行为时用 `--scenario` 指定场景编号，不必每次跑满约 20 次 LLM 调用。指定 5 或 6 时会自动补跑场景 1——否则记忆里没有姓名，断言必然失败，看起来像 bug 实为误用。

注意它每次运行会清空 `user_001` 的长期记忆，以保证断言可复现。

查看中间结果的辅助脚本：

```bash
uv run python scripts/test_splitter.py  # 文档切分结果
uv run python scripts/test_short_term.py  # 短期记忆压缩
uv run python scripts/test_long_term.py  # 长期记忆增删改查
```

## 注意事项

- **订单是 mock 数据**：`data/orders.db` 首次运行时自动创建并播种 `12345`~`12348` 四条订单，并非真实订单系统。
- **向量库为空的症状**：产品咨询会一直回答"没有相关信息"，容易被误判成子 Agent 没接上。`run_customer_service.py` 会直接报错拦截，`build_full_system.py` 则自动构建。
- **测试有成本**：多数测试脚本会调用云端模型，反复跑会产生费用。`test_memory_manager.py` 用临时库，不写真实数据。
- **`examples/` 与本项目无关**：里面是 deepagents 的官方示例（研究助手、Tavily 搜索等），不是客服系统的一部分。

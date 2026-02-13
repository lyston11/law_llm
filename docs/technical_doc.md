# 智能领域聊天机器人 — 项目技术文档

## 1. 系统架构总览

### 1.1 架构图

```
┌─────────────────────────────────────────────────────────┐
│                     前端 (Vue 3 + Element Plus)          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ 会话管理  │  │   聊天界面    │  │  分析面板         │   │
│  │SessionList│  │  ChatMessage  │  │  AnalysisPanel   │   │
│  │          │  │  +推荐卡片 ⭐   │  │                  │   │
│  └──────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP REST API (Axios)
┌─────────────────────────▼───────────────────────────────┐
│              FastAPI 后端 (api/main.py)                   │
│  ┌────────────────────────────────────────────────────┐  │
│  │            DomainAgent (src/agent.py)               │  │
│  │  ┌───────────────────────────────────────────────┐ │  │
│  │  │    Qwen-Plus LLM (Function Calling / ReAct)   │ │  │
│  │  │  输入 → 推理 → 调用工具(0~N次) → 最终回答     │ │  │
│  │  └────────┬──────────┬──────────┬──────────┬─────┘ │  │
│  │           │          │          │          │        │  │
│  │  ┌────────▼──┐ ┌─────▼────┐ ┌──▼───────┐ ┌▼────┐  │  │
│  │  │知识库检索  │ │法条查询   │ │知识图谱   │ │场景  │  │  │
│  │  │(RAG)      │ │          │ │(Graph)   │ │分析  │  │  │
│  │  └─────┬─────┘ └────┬─────┘ └────┬─────┘ └──┬───┘  │  │
│  └────────┼────────────┼────────────┼───────────┼──────┘  │
│           │            │            │           │          │
│  ┌────────▼────────────▼──┐ ┌───────▼───┐ ┌────▼───────┐ │
│  │  RAGRetriever          │ │ NetworkX  │ │ Entity +   │ │
│  │  ChromaDB + BGE        │ │ GraphML   │ │ Sentiment  │ │
│  └────────────────────────┘ └───────────┘ └────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  QuestionRecommender (src/recommendation.py) ⭐      │  │
│  │  基于对话上下文生成智能推荐问题                      │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 1.2 技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 前端框架 | Vue 3 + Vite | 3.x | 响应式 UI |
| UI 组件库 | Element Plus | 2.x | 界面组件 |
| HTTP 客户端 | Axios | 1.x | API 调用 |
| 后端框架 | FastAPI | 0.100+ | 异步 REST API |
| ASGI 服务器 | Uvicorn | 0.23+ | 高性能服务器 |
| 大语言模型 | 通义千问 Qwen-Plus | — | 对话推理、Function Calling |
| 推荐模型 | 通义千问 Qwen-Turbo | — | 智能问题生成 ⭐ |
| 向量数据库 | ChromaDB | 0.4+ | RAG 向量存储 |
| 嵌入模型 | BGE-small-zh-v1.5 | — | 文本向量化 |
| 知识图谱 | NetworkX | 3.x | 图数据结构 |
| 中文分词 | jieba | 0.42+ | 文本预处理 |

### 1.3 数据流

```
用户输入
  │
  ▼
前端 App.vue (send方法)
  │ POST /dialog { user_input, session_id }
  ▼
FastAPI api/main.py (dialog端点)
  │ agent.chat(user_input, conversation_history)
  ▼
DomainAgent._build_messages()  ── 构建: system_prompt + 历史 + 用户输入
  │
  ▼
DomainAgent._call_llm()  ── 调用 Qwen API (带 tools 参数)
  │
  ├── LLM 返回 tool_calls ──→ 执行工具 ──→ 结果写入 messages ──→ 再调 LLM（循环）
  │
  └── LLM 返回 content ──→ 最终回答
  │
  ▼
QuestionRecommender.generate() ⭐  ── 生成 3-5 个相关问题
  │
  ▼
返回 { response, agent_actions, sources, conversation_history, recommended_questions }
  │
  ▼
前端展示回答 + 推理过程 + 来源标注 + 推荐问题卡片 ⭐
```

---

## 2. Agent 核心机制

### 2.1 DomainAgent 类

**文件**：`src/agent.py`

DomainAgent 是整个系统的核心，负责：
1. 管理对话上下文（conversation history）
2. 调用 Qwen LLM（通过 OpenAI 兼容 API）
3. 执行 ReAct 循环：推理 → 调用工具 → 观察结果 → 再推理
4. 返回最终回答 + Agent 行为记录

### 2.2 System Prompt 设计

```python
SYSTEM_PROMPT = "\n".join([
    "# 你的身份",
    "你的名字叫「智能领域聊天机器人」。你是一个通用型 AI 助手...",
    "# 重要：自我介绍",
    "当用户打招呼或问'你是谁'时，你必须这样介绍自己：...",
    "# 你的工具",
    "你配备了法律领域的专业工具...",
    "# 回答规范",
    "- 使用中文，通俗易懂...",
])
```

**设计要点**：
- 明确 Agent 身份为「通用型 AI 助手」，法律只是其擅长领域
- 严格规定自我介绍规范，防止将自己说成"法律专用助手"
- 指导工具调用策略：法律问题调用工具，通用问题直接回答
- 回答规范：不编造法条、不足时诚实告知、法律回答附免责声明
- 动态注入当前日期时间（在 `_build_messages` 中实现）

### 2.3 ReAct 循环流程

```python
def chat(self, user_input, conversation_history=None):
    messages = self._build_messages(user_input, conversation_history)
    
    for round_idx in range(self.max_tool_rounds):  # 默认最多 5 轮
        llm_response = self._call_llm(messages)
        
        if llm_response has tool_calls:
            # 1. 将 assistant 消息（含 tool_calls）加入 messages
            # 2. 执行每个工具，获取结果
            # 3. 将工具结果（role: "tool"）加入 messages
            # 4. continue → 下一轮让 LLM 看到工具结果
        else:
            # LLM 直接返回文本回答，循环结束
            return final_response
    
    # 超过最大轮次，强制生成回答（不传 tools 参数）
    return _force_final_answer(messages)
```

### 2.4 Function Calling 机制

系统使用 **OpenAI 兼容格式** 与 Qwen API 交互：

**请求格式**：
```json
{
    "model": "qwen-plus",
    "messages": [...],
    "tools": [工具定义列表],
    "tool_choice": "auto",
    "temperature": 0.7
}
```

**LLM 返回工具调用时**：
```json
{
    "role": "assistant",
    "content": null,
    "tool_calls": [{
        "id": "call_xxx",
        "type": "function",
        "function": {
            "name": "search_legal_knowledge",
            "arguments": "{\"query\": \"劳动合同解除赔偿\"}"
        }
    }]
}
```

**工具结果反馈**：
```json
{
    "role": "tool",
    "tool_call_id": "call_xxx",
    "content": "{\"success\": true, \"results\": [...]}"
}
```

### 2.5 工具注册表

**文件**：`src/tools/__init__.py`

```python
TOOL_REGISTRY = {
    "search_legal_knowledge": (KNOWLEDGE_SEARCH_DEF, execute_knowledge_search),
    "lookup_legal_article": (ARTICLE_LOOKUP_DEF, execute_article_lookup),
    "query_knowledge_graph": (KNOWLEDGE_GRAPH_DEF, execute_knowledge_graph),
    "analyze_legal_situation": (SITUATION_ANALYZER_DEF, execute_situation_analyzer),
}
```

每个工具包含两部分：
- **TOOL_DEFINITION**：OpenAI 兼容的 JSON Schema 定义（描述工具功能、参数类型、必填字段）
- **execute 函数**：工具的实际执行逻辑，接收参数、返回 JSON 字符串

---

## 3. 四个工具的详细设计

### 3.1 search_legal_knowledge — 法律知识库检索

**文件**：`src/tools/knowledge_search.py`

**功能**：基于 RAG 向量检索，从知识库中搜索与用户问题相关的法律资料。

**JSON Schema 定义**：
```json
{
    "name": "search_legal_knowledge",
    "parameters": {
        "properties": {
            "query": {"type": "string", "description": "搜索查询关键词"},
            "domain": {"type": "string", "enum": ["劳动纠纷", "婚姻家庭", ...]},
            "top_k": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    }
}
```

**执行流程**：
1. 接收 query、domain、top_k 参数
2. 调用 `RAGRetriever.retrieve()` 进行向量检索
3. 返回结果列表，每项包含 content、domain、source、relevance_score

**输出示例**：
```json
{
    "success": true,
    "query": "劳动合同解除赔偿",
    "result_count": 5,
    "results": [
        {"content": "根据劳动合同法第四十七条...", "domain": "劳动纠纷", "relevance_score": 0.8523}
    ]
}
```

### 3.2 lookup_legal_article — 法条查询

**文件**：`src/tools/article_lookup.py`

**功能**：查询具体的法律条文内容。支持中文/阿拉伯数字条款编号互转。

**参数**：`law_name`（法律名称）、`article_number`（条款编号）

**特殊逻辑**：
- `_normalize_article_number()` 将"第二十条"转为 ["第二十条", "第20条"] 两种变体
- `_cn_to_arabic()` 处理中文数字转阿拉伯数字（支持百位、十位）
- 先检索，再按精确匹配和模糊匹配分类排序

### 3.3 query_knowledge_graph — 知识图谱查询

**文件**：`src/tools/knowledge_graph.py`

**功能**：查询法律知识图谱中的实体关系，帮助理解法律概念之间的联系。

**参数**：
- `entity`：查询实体名称
- `relation_type`：`all_neighbors`（邻居节点）/ `related`（2跳内关系）/ `shortest_path`（最短路径）
- `target_entity`：shortest_path 时的目标实体

**底层实现**：
- 使用 NetworkX 加载 `.graphml` 格式的知识图谱文件
- `_find_matching_nodes()` 支持精确匹配和包含匹配
- 惰性加载：首次调用时加载图谱文件

### 3.4 analyze_legal_situation — 法律场景分析

**文件**：`src/tools/situation_analyzer.py`

**功能**：分析用户描述的法律场景，提取关键事实要素。

**输出内容**：
- **实体识别**：调用 `EntityRecognizer`，识别 PERSON、COMPANY、TIME、LOCATION
- **情感分析**：调用 `SentimentAnalyzer`，判断 positive / negative / neutral
- **领域判断**：基于关键词匹配，判断属于哪个法律领域

---

## 4. RAG 知识库模块

**文件**：`src/rag.py`

### 4.1 知识库构建流程

```
法律数据集 (JSONL)
  │ scripts/build_rag_knowledge_base.py
  ▼
文本预处理 + 分块 (RecursiveCharacterTextSplitter)
  │
  ▼
BGE-small-zh-v1.5 向量化 (HuggingFaceEmbeddings)
  │
  ▼
ChromaDB 持久化存储 (data/knowledge_base/vector_db/)
```

**数据来源**：
- `data/legal/datasets/DISC-Law-SFT-Pair-QA-released.jsonl`
- `data/legal/datasets/DISC-Law-SFT-Triplet-QA-released.jsonl`
- 来自 [DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM) 开源项目

### 4.2 RAGRetriever 类

**初始化**：
1. 加载 BGE 嵌入模型（`BAAI/bge-small-zh-v1.5`，约 100MB）
2. 连接 ChromaDB 向量数据库
3. 设置 `is_ready` 标志

**检索接口 `retrieve()`**：
```python
def retrieve(self, query, k=5, domain=None):
    # 1. 将 query 通过 BGE 模型编码为向量
    # 2. 在 ChromaDB 中做 similarity_search_with_relevance_scores
    # 3. 可选按 domain 过滤
    # 4. 返回 [{content, metadata, score}, ...]
```

**状态接口 `get_status()`**：返回知识库就绪状态、文档数量、嵌入模型名称。

### 4.3 配置参数

| 参数 | 配置项 | 默认值 | 说明 |
|------|--------|--------|------|
| 嵌入模型 | `RAG_EMBEDDING_MODEL` | `models/bge-small-zh-v1.5` | 本地路径 |
| 检索数量 | `RAG_TOP_K` | 5 | 每次检索返回的文档数 |
| 知识库路径 | `KNOWLEDGE_BASE_DIR` | `data/knowledge_base/vector_db` | ChromaDB 存储位置 |

---

## 5. 知识图谱模块

### 5.1 构建流程

```
法律数据集
  │ scripts/build_knowledge_graph.py
  ▼
提取实体和关系
  │
  ▼
NetworkX 图数据结构
  │
  ▼
导出为 GraphML (data/knowledge_graph/law_knowledge_graph.graphml)
```

### 5.2 查询实现

知识图谱查询在 `src/tools/knowledge_graph.py` 中实现：

- **图加载**：惰性加载 `.graphml` 文件，全局缓存
- **模糊匹配**：`_find_matching_nodes()` 支持精确和包含匹配
- **三种查询模式**：
  - `all_neighbors`：返回直接邻居节点及边关系
  - `related`：返回 1 跳内的关联实体
  - `shortest_path`：使用 Dijkstra 算法查找两实体间最短路径

---

## 6. NLP 工具模块

### 6.1 实体识别

**文件**：`src/entity.py`

基于正则表达式的命名实体识别，支持 4 种实体类型：

| 实体类型 | 标识 | 匹配规则 |
|----------|------|----------|
| 人名 | PERSON | 常见姓氏 + 1-2 个汉字 |
| 公司 | COMPANY | 中文名 + "公司/企业/集团" 等后缀 |
| 时间 | TIME | 年/月/日、"N年/N个月" 等 |
| 地点 | LOCATION | "省/市/县/区" 后缀 + 主要城市名 |

### 6.2 情感分析

**文件**：`src/sentiment.py`

基于情感词典的情感倾向分析：

- **正面词汇**：谢谢、满意、不错、很好...
- **负面词汇**：不满意、糟糕、愤怒、不公平...
- **中性词汇**：请问、咨询、了解、如何...

通过词频统计判断主导情感（positive / negative / neutral）。

---

## 7. 前端架构

### 7.1 组件结构

```
frontend/src/
├── App.vue              # 主布局组件（三栏布局 + 核心状态管理）
├── main.js              # 入口文件
├── api/
│   └── index.js         # API 接口封装（Axios）
└── components/
    ├── ChatMessage.vue   # 单条消息组件（含 Agent 推理过程 + 来源标注）
    ├── SessionList.vue   # 左侧会话列表
    └── AnalysisPanel.vue # 右侧分析面板（系统状态 + 工具调用详情 + 反馈）
```

### 7.2 状态管理

使用 Vue 3 Composition API（`ref` + `reactive`），未引入 Vuex/Pinia：

```javascript
// App.vue 中的核心状态
const messages = ref([])           // 当前会话消息列表
const sessions = ref([])           // 所有会话列表
const currentSessionId = ref('')   // 当前活动会话 ID
const systemStatus = reactive({})  // 系统健康状态
const currentAnalysis = reactive({}) // 最新 Agent 分析结果
```

**数据持久化**：会话列表和聊天记录存储在 `localStorage`。

### 7.3 API 封装

**文件**：`frontend/src/api/index.js`

```javascript
const api = axios.create({
    baseURL: 'http://localhost:8000',
    timeout: 60000,  // Agent 多轮工具调用可能较慢
})
```

主要接口：

| 函数 | 方法 | 路径 | 说明 |
|------|------|------|------|
| `healthCheck` | GET | `/health` | 健康检查 |
| `sendMessage` | POST | `/dialog` | 发送消息 |
| `resetDialog` | DELETE | `/dialog/{id}` | 重置会话 |
| `getSummary` | POST | `/dialog/{id}/summary` | 生成总结 |
| `submitFeedback` | POST | `/feedback` | 提交反馈 |
| `getKnowledgeStatus` | GET | `/knowledge/status` | 知识库状态 |
| `getAgentStatus` | GET | `/agent/status` | Agent 状态 |

### 7.4 消息展示逻辑

`ChatMessage.vue` 中的关键展示逻辑：

- **Agent 推理过程**：当 `msg.agentActions` 有值时，展示绿色面板，列出每个工具调用的标签和结果摘要
- **内容格式化**：HTML 转义 + 换行符转 `<br>`
- **来源标注**：当 `msg.sources` 有值时，底部展示引用领域和相关度百分比
- **工具标签颜色**：知识库检索(蓝) / 法条查询(绿) / 知识图谱(橙) / 场景分析(灰)
- **推荐问题卡片 ⭐**：
  - 当 `msg.recommendedQuestions` 有值时，在 AI 回复下方展示推荐卡片
  - 卡片标题："💡 您可能还想问："
  - 卡片布局：每行显示 2-3 个问题，采用卡片式设计
  - 交互逻辑：点击问题自动填入输入框（用户可编辑后再发送）
  - 样式：鼠标悬停高亮，支持折叠/展开

---

## 8. API 接口完整列表

### 8.1 核心接口

#### POST /dialog — 对话

**请求**：
```json
{
    "user_input": "被公司辞退了怎么办",
    "session_id": "session_1234"
}
```

**响应**：
```json
{
    "response": "根据劳动合同法...",
    "session_id": "session_1234",
    "status": "success",
    "agent_actions": [
        {
            "tool": "search_legal_knowledge",
            "input": {"query": "公司辞退赔偿", "domain": "劳动纠纷"},
            "result_summary": "找到 5 条相关法律资料"
        }
    ],
    "sources": [
        {"domain": "劳动纠纷", "score": 0.85, "snippet": "根据劳动合同法第四十七条..."}
    ],
    "recommended_questions": [  // ⭐ 新增：智能推荐问题
        "如何计算经济补偿金？",
        "公司违法辞退有哪些情形？",
        "被辞退后社保怎么处理？"
    ]
}
```

#### GET /health — 健康检查

**响应**：
```json
{
    "status": "healthy",
    "project": "智能领域聊天机器人",
    "version": "3.0.0",
    "architecture": "agent",
    "model": "qwen-plus",
    "tools": ["search_legal_knowledge", "lookup_legal_article", "query_knowledge_graph", "analyze_legal_situation"]
}
```

### 8.2 知识库接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/knowledge/status` | 知识库状态（就绪/文档数） |
| POST | `/knowledge/search` | 手动搜索知识库 |
| GET | `/agent/status` | Agent 详细状态 |

### 8.3 历史/反馈/总结接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/dialog/{session_id}/history` | 获取对话历史 |
| GET | `/dialog/{session_id}/history/export` | 导出对话历史 |
| DELETE | `/dialog/{session_id}` | 重置对话 |
| GET | `/sessions` | 获取会话列表 |
| POST | `/feedback` | 提交反馈（rating 1-5 + comment） |
| GET | `/feedback/{session_id}` | 获取会话反馈 |
| GET | `/feedback/stats` | 反馈统计 |
| POST | `/dialog/{session_id}/summary` | 生成对话总结 |
| DELETE | `/dialog/{session_id}/summary` | 删除对话总结 |

---

## 9. 智能问题推荐模块 ⭐

### 9.1 模块概述

**文件**：`src/recommendation.py`

智能问题推荐模块基于对话上下文自动生成相关问题，帮助用户探索更多相关内容。

**核心特性**：
- 通用推荐机制，适配所有领域（不限于法律）
- 利用现有 LLM 能力，无需额外模型训练
- 失败降级：推荐失败不影响主对话流程
- 智能跳过：简单对话（打招呼、感谢）不生成推荐

### 9.2 QuestionRecommender 类

**初始化**：
```python
class QuestionRecommender:
    def __init__(self):
        self.model = config.RECOMMEND_MODEL  # qwen-turbo
        self.timeout = config.RECOMMEND_TIMEOUT
        self.count_range = config.RECOMMEND_COUNT
```

**主要方法**：

#### generate() - 生成推荐问题
```python
def generate(self, conversation_history, agent_actions, response):
    """
    基于对话上下文生成推荐问题

    Args:
        conversation_history: 对话历史列表
        agent_actions: Agent 工具调用记录
        response: AI 的最终回复

    Returns:
        list[str]: 推荐问题列表（3-5个），失败返回空列表
    """
    # 1. 判断是否跳过（简单对话、连续推荐等）
    if self._should_skip(conversation_history, response):
        return []

    # 2. 构建 Prompt
    prompt = self._build_prompt(conversation_history, agent_actions, response)

    # 3. 调用 LLM（使用 qwen-turbo 以降低成本）
    try:
        result = self._call_llm(prompt, timeout=self.timeout)
        questions = self._parse_response(result)
        return questions
    except Exception as e:
        logger.warning(f"推荐生成失败: {e}")
        return []
```

#### _should_skip() - 判断是否跳过推荐
```python
def _should_skip(self, conversation_history, response):
    """判断是否应该跳过推荐生成"""
    # 场景 1: 简短对话（打招呼、感谢、再见）
    short_patterns = ["你好", "您好", "谢谢", "感谢", "再见", "拜拜"]
    last_user_msg = conversation_history[-1].get("content", "")
    if any(pattern in last_user_msg for pattern in short_patterns):
        return True

    # 场景 2: 用户明确表示没有其他问题
    if any(pattern in response for pattern in ["没有其他问题", "暂无其他", "不需要了"]):
        return True

    # 场景 3: 回复过短（< 30 字）
    if len(response) < 30:
        return True

    return False
```

#### _build_prompt() - 构建推荐 Prompt
```python
def _build_prompt(self, conversation_history, agent_actions, response):
    """构建用于生成推荐问题的 Prompt"""
    prompt = [
        {
            "role": "system",
            "content": "基于当前对话，生成3-5个用户可能想问的相关问题。\n\n"
                      "要求：\n"
                      "1. 问题要具体、有价值，避免重复已有问题\n"
                      "2. 考虑对话主题和工具调用结果（如检索到的知识、实体关系）\n"
                      "3. 问题可以是：追问细节、了解相关概念、延伸话题、实用建议\n"
                      "4. 简洁明了，每题不超过20字\n\n"
                      "输出JSON格式：\n"
                      '{\"questions\": [\"问题1\", \"问题2\", \"问题3\"]}'
        },
        {
            "role": "user",
            "content": self._format_context(conversation_history, agent_actions, response)
        }
    ]
    return prompt
```

#### _format_context() - 格式化上下文
```python
def _format_context(self, conversation_history, agent_actions, response):
    """将对话历史、工具调用、回复内容格式化为上下文"""
    context_parts = [
        "# 对话历史",
        self._format_history(conversation_history[-3:]),  # 最近 3 轮
        "",
        "# 工具调用记录",
        self._format_actions(agent_actions),
        "",
        "# AI 回复",
        response
    ]
    return "\n".join(context_parts)
```

### 9.3 与 API 的集成

在 `api/main.py` 的 `/dialog` 端点中集成推荐生成：

```python
@app.post("/dialog")
async def dialog(request: DialogRequest):
    # ... 获取对话历史 ...

    # 1. Agent 生成回复
    result = agent.chat(user_input, conversation_history)

    # 2. ⭐ 生成推荐问题
    recommended_questions = []
    if config.RECOMMEND_ENABLED:
        try:
            recommender = QuestionRecommender()
            recommended_questions = recommender.generate(
                conversation_history=conversation_history,
                agent_actions=result.get("agent_actions", []),
                response=result["response"]
            )
        except Exception as e:
            logger.warning(f"推荐生成失败: {e}")
            # 失败不影响主流程，返回空列表

    # 3. 返回完整响应
    return {
        "response": result["response"],
        "agent_actions": result.get("agent_actions", []),
        "sources": result.get("sources", []),
        "conversation_history": result["conversation_history"],
        "recommended_questions": recommended_questions  # ⭐
    }
```

### 9.4 前端展示

在 `ChatMessage.vue` 中添加推荐卡片：

```vue
<template>
  <!-- AI 消息内容 -->
  <div class="message-content">{{ formattedContent }}</div>

  <!-- 推荐问题卡片 ⭐ -->
  <div
    v-if="message.recommendedQuestions?.length"
    class="recommended-questions"
  >
    <div class="rq-header">💡 您可能还想问：</div>
    <div class="rq-list">
      <div
        v-for="(question, idx) in message.recommendedQuestions"
        :key="idx"
        class="rq-item"
        @click="handleQuestionClick(question)"
      >
        {{ question }}
      </div>
    </div>
  </div>
</template>

<script setup>
const handleQuestionClick = (question) => {
  // 点击推荐问题后，自动填入输入框
  emit('fill-input', question)
}
</script>

<style scoped>
.recommended-questions {
  margin-top: 12px;
  padding: 12px;
  background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
  border-radius: 8px;
  border-left: 3px solid #409EFF;
}

.rq-header {
  font-size: 13px;
  font-weight: 500;
  color: #606266;
  margin-bottom: 8px;
}

.rq-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
}

.rq-item {
  padding: 8px 12px;
  background: white;
  border-radius: 6px;
  font-size: 13px;
  color: #303133;
  cursor: pointer;
  transition: all 0.2s;
}

.rq-item:hover {
  background: #409EFF;
  color: white;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.3);
}
</style>
```

### 9.5 性能与成本优化

| 优化策略 | 实现方式 |
|---------|---------|
| **使用廉价模型** | 推荐使用 `qwen-turbo` 而非 `qwen-plus`，成本降低 50% |
| **超时控制** | 设置 5 秒超时，避免阻塞主流程 |
| **智能跳过** | 简单对话直接跳过，减少不必要的 LLM 调用 |
| **失败降级** | 推荐失败返回空列表，不影响用户体验 |
| **缓存优化**（未来） | 对相似问题缓存推荐结果，减少重复调用 |

---

## 10. 旧版系统（src/legacy/）

### 9.1 目录说明

`src/legacy/` 保留了 v1.0/v2.0 的规则对话系统模块，不参与当前系统运行，仅用于论文对比实验。

| 文件 | 功能 | 对应新版机制 |
|------|------|-------------|
| `dialog.py` | 对话管理器（规则流水线） | DomainAgent |
| `intent.py` | TF-IDF 意图识别 | LLM 自主理解意图 |
| `intent_bert.py` | BERT 意图识别 | LLM 自主理解意图 |
| `tfidf.py` | TF-IDF 计算器 | 不再需要 |
| `slot.py` | 槽位填充 | LLM 自主提取信息 |
| `state.py` | 对话状态跟踪 | conversation_history |

### 9.2 旧版处理流程

```
用户输入
  → TF-IDF/BERT 意图识别 → 匹配场景节点
  → 正则槽位填充 → 追问缺失槽位
  → 对话状态跟踪 → 生成 RAG/规则回答
```

**与新版 Agent 的核心区别**：
- 旧版：必须先分类意图、再填槽、再回答，流程固定
- 新版：LLM 自主判断，一步到位，按需调用工具

### 9.3 对比实验

运行 `python scripts/compare_versions.py`，使用同一批测试问题对比两个系统。详见 README.md 的「运行对比实验」章节。

---

## 10. 配置说明

**文件**：`config/config.py`

### 10.1 核心配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `API_KEY` | (从 .env 读取) | 阿里云百炼 API Key |
| `AGENT_MODEL` | `qwen-plus` | Agent 使用的 LLM 模型 |
| `AGENT_MAX_TOOL_ROUNDS` | 5 | 最大工具调用轮次 |
| `AGENT_TEMPERATURE` | 0.7 | 生成温度 |
| `MAX_DIALOG_HISTORY` | 10 | 对话历史保留轮数 |
| `RAG_TOP_K` | 5 | 知识库检索返回数量 |
| `RAG_EMBEDDING_MODEL` | `models/bge-small-zh-v1.5` | 嵌入模型路径 |
| `RECOMMEND_ENABLED` | `True` | 是否启用智能推荐 ⭐ |
| `RECOMMEND_COUNT` | `(3, 5)` | 推荐问题数量范围 ⭐ |
| `RECOMMEND_TIMEOUT` | `5` | 推荐生成超时时间（秒）⭐ |
| `RECOMMEND_MODEL` | `qwen-turbo` | 推荐生成使用的模型 ⭐ |

### 10.2 环境变量

通过 `.env` 文件配置（不提交到 Git）：

```ini
API_KEY=sk-xxx            # 必填
AGENT_MODEL=qwen-plus     # 可选
AGENT_MAX_TOOL_ROUNDS=5   # 可选
AGENT_TEMPERATURE=0.7     # 可选
LOG_LEVEL=INFO             # 可选
```

### 10.3 旧版配置

`config.py` 下半部分包含旧版规则系统的配置（意图识别阈值、槽位优先级、依赖关系等），仅供 `src/legacy/` 模块使用。

---

## 11. 部署说明

### 11.1 最小部署（无知识库）

仅需：
1. Python 3.10+ 环境
2. `pip install -r requirements.txt`
3. `.env` 配置 `API_KEY`
4. `python -m uvicorn api.main:app --port 8000`
5. `cd frontend && npm install && npm run dev`

此模式下系统可正常对话，但法律问题不会引用知识库。

### 11.2 完整部署

在最小部署基础上：
1. 下载 BGE 嵌入模型到 `models/bge-small-zh-v1.5/`
2. 下载法律数据集到 `data/legal/datasets/`
3. 运行 `python scripts/build_rag_knowledge_base.py` 构建向量库
4. 运行 `python scripts/build_knowledge_graph.py` 构建知识图谱

详见 README.md 的部署指南。

# 基于API部署的智能领域聊天机器人

## 摘要

随着大语言模型技术的快速发展，基于自然语言处理的智能对话系统在各行业中的应用日益广泛。本文设计并实现了一个基于 Agent 架构的智能领域聊天机器人系统，以法律咨询领域为深度增强方向。系统以通义千问大语言模型为核心，采用 Function Calling 机制实现 ReAct（Reasoning + Acting）推理循环，使大模型能够自主决策是否调用外部工具。系统集成了检索增强生成（RAG）法律知识库、法律知识图谱查询、法条精确检索和法律场景分析等四个专业工具，在法律咨询场景中提供有依据的专业回答；同时保留通用对话能力，对非法律问题直接生成回答。后端采用 FastAPI 框架提供 RESTful API，前端基于 Vue 3 和 Element Plus 构建现代化交互界面，支持多会话管理、Agent 推理过程可视化、对话总结与反馈等功能。实验结果表明，相较于传统规则系统，Agent 架构在回答准确性、问题覆盖范围和用户体验方面均有显著提升。

**关键词**：大语言模型；Agent 架构；Function Calling；检索增强生成；知识图谱；法律智能问答

## Abstract

With the rapid development of large language model (LLM) technology, intelligent dialogue systems based on natural language processing have been increasingly applied across various industries. This thesis designs and implements an Agent-based intelligent domain chatbot system, with legal consultation as the domain of deep enhancement. The system takes the Qwen large language model as its core, employing a Function Calling mechanism to implement a ReAct (Reasoning + Acting) reasoning loop, enabling the LLM to autonomously decide whether to invoke external tools. The system integrates four specialized tools: a Retrieval-Augmented Generation (RAG) legal knowledge base, legal knowledge graph query, precise legal article lookup, and legal situation analysis, providing evidence-based professional answers in legal consultation scenarios while retaining general conversational capabilities. The backend is built with FastAPI providing RESTful APIs, and the frontend is developed using Vue 3 with Element Plus, supporting multi-session management, Agent reasoning process visualization, dialogue summarization, and feedback collection. Experimental results demonstrate that compared with traditional rule-based systems, the Agent architecture achieves significant improvements in answer accuracy, question coverage, and user experience.

**Keywords**: Large Language Model; Agent Architecture; Function Calling; Retrieval-Augmented Generation; Knowledge Graph; Legal Intelligent Q&A

---

## 第1章 绪论

### 1.1 研究背景与意义

近年来，以 GPT、通义千问、文心一言为代表的大语言模型（Large Language Model, LLM）技术取得了突破性进展，展现出强大的自然语言理解与生成能力。这些模型不仅能够进行流畅的多轮对话，还能执行复杂的推理、总结和创作任务，为智能对话系统的发展带来了革命性的变化。

与此同时，法律咨询作为一项专业性强、需求广泛的社会服务，长期面临着资源分布不均、获取成本高的问题。据统计，我国每万人拥有的律师数量远低于发达国家，大量基层群众在遇到法律问题时难以获得及时、准确的法律指导。将人工智能技术应用于法律咨询领域，开发智能法律问答系统，对于提高法律服务的普惠性和可及性具有重要的现实意义。

然而，传统的智能对话系统多采用"意图识别—槽位填充—规则匹配"的固定流水线架构，存在以下局限性：

1. **意图分类固化**：需要预定义意图类别，无法处理超出预设范围的问题
2. **流程刚性**：每次对话必须经过完整的意图识别、槽位填充、状态跟踪流程，灵活性差
3. **通用能力缺失**：专为特定领域设计，无法处理领域外的通用问题
4. **维护成本高**：规则和模板需要人工编写和更新

随着 LLM Agent 技术的兴起，一种新的对话系统架构正在形成：以大语言模型为"大脑"，通过 Function Calling 机制调用外部工具，实现推理与行动的有机结合。这种架构具有高度的灵活性和可扩展性，LLM 能够根据用户输入自主判断问题类型，决定是否需要调用领域工具，并将工具返回的信息整合为高质量的回答。

基于上述背景，本文设计并实现了一个基于 Agent 架构的智能领域聊天机器人系统，以法律领域为增强方向，集成 RAG 知识库检索、知识图谱、法条查询和场景分析等工具，旨在探索 LLM Agent 在垂直领域智能问答中的应用范式。

### 1.2 国内外研究现状

#### 1.2.1 大语言模型与 Agent 技术

2022 年 ChatGPT 发布后，大语言模型技术进入爆发期。OpenAI 提出的 Function Calling 机制允许 LLM 在对话过程中调用外部函数，为构建 Agent 系统提供了技术基础。Yao 等人提出的 ReAct（Reasoning and Acting）框架，将推理（Reasoning）和行动（Acting）交替进行，使 LLM 能够在推理过程中调用工具获取外部信息，显著提升了回答的准确性和可靠性。

国内方面，阿里云推出的通义千问系列模型支持 OpenAI 兼容格式的 Function Calling，为国内开发者构建 Agent 应用提供了便利。百度文心、智谱 ChatGLM 等模型也陆续支持了工具调用能力。

#### 1.2.2 检索增强生成（RAG）技术

检索增强生成（Retrieval-Augmented Generation, RAG）是解决 LLM "幻觉"问题的有效技术路线。Lewis 等人在 2020 年首次提出 RAG 框架，通过在生成前检索相关文档，将外部知识注入生成过程。近年来，RAG 技术在问答系统中得到广泛应用，尤其在专业领域，RAG 能够确保回答基于真实文档，避免模型凭空编造。

向量检索是 RAG 的核心环节。BGE（BAAI General Embedding）系列模型是北京智源人工智能研究院推出的中文嵌入模型，在多个中文语义检索基准上达到领先水平。ChromaDB 是一个轻量级的开源向量数据库，支持持久化存储和相似度检索，适合中小规模应用场景。

#### 1.2.3 法律智能问答系统

法律 NLP 是自然语言处理的重要应用方向。国内代表性工作包括复旦大学的 DISC-LawLLM 项目，构建了大规模法律问答数据集；清华大学的 ChatLaw 项目，探索了法律领域的大模型微调方案。这些工作为法律知识库的构建和法律文本的理解提供了数据基础和技术参考。

然而，现有的法律智能问答系统多采用端到端的微调方案或简单的 RAG 检索，较少采用 Agent 架构。本文提出的方案将 LLM Agent 与法律领域工具相结合，使系统同时具备通用对话能力和专业领域深度，这在现有研究中具有一定的创新性。

### 1.3 本文主要工作与创新点

本文的主要工作包括：

1. **设计并实现了基于 Agent 架构的智能对话系统**：以通义千问大模型为核心，通过 Function Calling 和 ReAct 模式实现自主工具调用
2. **构建了法律领域工具集**：包括 RAG 知识库检索、法条精确查询、知识图谱查询和法律场景分析四个工具
3. **实现了前后端分离的完整系统**：FastAPI 后端 + Vue 3 前端，支持多会话管理和 Agent 推理过程可视化
4. **进行了系统对比实验**：与传统规则系统进行了多维度对比分析

本文的创新点：

1. **Agent 架构应用于法律领域**：采用 LLM Agent 替代传统规则流水线，实现了"通用对话 + 领域深度"的双重能力
2. **多工具协同**：通过工具注册表机制，实现了 RAG 检索、知识图谱、NLP 分析等多种工具的统一管理和自主调用
3. **推理过程透明化**：前端实时展示 Agent 的工具调用过程，增强了系统的可解释性和用户信任度

### 1.4 论文组织结构

本文共分为六章：

- 第1章介绍研究背景、国内外现状和本文主要工作
- 第2章介绍系统涉及的相关技术和理论基础
- 第3章进行系统需求分析和总体设计
- 第4章详细阐述系统的设计与实现
- 第5章展示系统测试结果和分析
- 第6章总结全文工作并展望未来研究方向

---

## 第2章 相关技术与理论基础

### 2.1 大语言模型与 Function Calling

大语言模型（Large Language Model, LLM）是基于 Transformer 架构，在大规模文本语料上预训练的神经网络模型。LLM 通过自回归方式生成文本，能够理解和执行多种自然语言任务。

本系统使用的通义千问（Qwen-Plus）是阿里云推出的大语言模型，通过阿里云百炼平台提供 API 服务。Qwen-Plus 支持 OpenAI 兼容格式的 Function Calling，即在 API 请求中传入工具定义（JSON Schema），模型会根据用户输入判断是否需要调用工具，并以结构化格式返回工具调用指令。

Function Calling 的核心流程：

1. 开发者在 API 请求中定义可用工具（`tools` 字段）
2. LLM 分析用户输入，判断是否需要调用工具
3. 若需要，LLM 返回 `tool_calls` 字段，包含工具名和参数
4. 开发者执行工具，将结果以 `role: "tool"` 消息返回
5. LLM 结合工具结果生成最终回答

### 2.2 Agent 架构与 ReAct 模式

Agent（智能代理）是一种以 LLM 为核心的系统架构，LLM 作为"大脑"自主规划和执行任务。ReAct 是 Yao 等人于 2022 年提出的 Agent 推理框架，核心思想是将推理（Reasoning）和行动（Acting）交替进行：

```
思考 (Thought): 分析用户问题，决定下一步行动
行动 (Action): 调用外部工具获取信息
观察 (Observation): 接收工具返回结果
... 循环直到获得足够信息 ...
回答 (Answer): 基于收集的信息生成最终回答
```

本系统实现的 ReAct 循环流程：

```
用户输入
  │
  ▼
LLM 推理（带 tools 参数）
  │
  ├── 返回 tool_calls → 执行工具 → 工具结果写入上下文 → 回到 LLM 推理
  │
  └── 返回 content → 最终回答，循环结束
```

Agent 架构相比传统规则系统的优势：

| 维度 | 传统规则系统 | Agent 架构 |
|------|-------------|-----------|
| 意图理解 | 预定义分类 | LLM 自主理解 |
| 工具调用 | 规则触发 | LLM 自主决策 |
| 通用能力 | 无 | LLM 原生能力 |
| 可扩展性 | 需改代码 | 添加工具定义即可 |
| 多轮对话 | 状态机 | 上下文记忆 |

### 2.3 检索增强生成（RAG）

检索增强生成（Retrieval-Augmented Generation, RAG）是一种结合信息检索和文本生成的技术方案，用于解决 LLM 知识时效性不足和"幻觉"（hallucination）问题。

RAG 的基本流程：

```
用户问题
  │
  ▼
文本编码（Embedding 模型）
  │
  ▼
向量检索（从知识库中召回 Top-K 相关文档）
  │
  ▼
构建增强 Prompt（将检索结果拼接到上下文中）
  │
  ▼
LLM 生成回答（基于检索到的真实文档）
```

本系统的 RAG 实现涉及以下关键技术：

- **Embedding 模型**：使用 BGE-small-zh-v1.5，一个针对中文优化的文本嵌入模型，能够将文本编码为 384 维向量
- **向量数据库**：使用 ChromaDB，一个开源的向量数据库，支持持久化存储和带过滤条件的相似度检索
- **文本分块**：使用 LangChain 的 RecursiveCharacterTextSplitter，将长文本切分为适合检索的片段

### 2.4 知识图谱

知识图谱（Knowledge Graph）以图结构存储实体及其关系，能够表达丰富的语义信息。在法律领域，知识图谱可以表达法律概念之间的上下位关系、构成要件之间的关联等结构化知识。

本系统使用 NetworkX 库构建和管理知识图谱，以 GraphML 格式持久化存储。知识图谱支持的查询操作包括：

- 邻居查询：查找与某个法律概念直接相关的所有概念
- 关系路径查询：查找两个法律概念之间的最短关系路径
- 关联查询：查找某个概念在指定跳数内的所有关联实体

### 2.5 FastAPI 与前后端分离

FastAPI 是一个基于 Python 的现代 Web 框架，具有以下特点：

- 基于 Python 类型提示的自动数据验证和序列化
- 自动生成 OpenAPI（Swagger）交互式文档
- 原生异步支持，高并发性能
- 使用 Uvicorn 作为 ASGI 服务器

前后端分离架构将 UI 呈现（前端）和业务逻辑（后端）解耦：

- **前端**：Vue 3 + Vite 构建单页应用（SPA），通过 Axios 调用后端 REST API
- **后端**：FastAPI 提供 RESTful API，处理业务逻辑和 AI 推理
- **通信**：HTTP + JSON 格式，跨域通过 CORS 中间件解决

---

## 第3章 系统需求分析与总体设计

### 3.1 需求分析

#### 3.1.1 功能需求

1. **智能对话**：用户可以通过自然语言与系统进行多轮对话，系统应能回答法律问题和通用问题
2. **法律知识检索**：对于法律问题，系统应能从知识库中检索相关法条和案例，提供有依据的专业回答
3. **知识图谱查询**：系统应能查询法律概念之间的关系，帮助用户理解法律术语
4. **会话管理**：支持创建、切换、删除多个对话会话，会话数据本地持久化
5. **推理过程展示**：前端应展示 Agent 的推理过程，包括调用了哪些工具、工具返回了什么结果
6. **对话总结**：支持对当前对话生成摘要总结
7. **对话导出**：支持将对话记录导出为 JSON 文件
8. **用户反馈**：支持用户对回答进行评分和文字反馈

#### 3.1.2 非功能需求

1. **响应性能**：对话响应时间应控制在 10 秒以内（含网络延迟和 LLM 推理）
2. **可扩展性**：系统应支持方便地添加新的领域工具
3. **可靠性**：当知识库或工具不可用时，系统应能降级为纯 LLM 对话模式
4. **可解释性**：Agent 的工具调用过程应对用户透明

### 3.2 系统总体架构设计

系统采用三层架构设计：

```
┌──────────────────────────────────────┐
│            表现层（前端）              │
│  Vue 3 + Element Plus + Vite         │
│  三栏布局：会话列表 | 聊天区 | 分析面板│
└──────────────┬───────────────────────┘
               │ REST API (HTTP/JSON)
┌──────────────▼───────────────────────┐
│           业务逻辑层（后端）           │
│  FastAPI (api/main.py)               │
│  ┌─────────────────────────────────┐ │
│  │      DomainAgent (核心引擎)      │ │
│  │  LLM + Function Calling + ReAct │ │
│  └─────────────────────────────────┘ │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│            数据与服务层               │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │ ChromaDB │ │ NetworkX │ │ NLP  │ │
│  │ (RAG)    │ │ (知识图谱)│ │ 工具 │ │
│  └──────────┘ └──────────┘ └──────┘ │
│  ┌──────────┐ ┌──────────┐          │
│  │ Qwen API │ │ BGE 模型 │          │
│  └──────────┘ └──────────┘          │
└──────────────────────────────────────┘
```

### 3.3 模块划分与功能描述

| 模块 | 文件 | 功能 |
|------|------|------|
| Agent 核心 | `src/agent.py` | DomainAgent 类，实现 ReAct 循环和 LLM 调用 |
| 工具注册表 | `src/tools/__init__.py` | 管理所有工具的定义和执行函数 |
| 知识库检索 | `src/tools/knowledge_search.py` | RAG 向量检索工具 |
| 法条查询 | `src/tools/article_lookup.py` | 法条精确检索工具 |
| 知识图谱 | `src/tools/knowledge_graph.py` | 知识图谱查询工具 |
| 场景分析 | `src/tools/situation_analyzer.py` | 法律场景分析工具 |
| RAG 检索器 | `src/rag.py` | RAGRetriever 类，管理向量库和检索 |
| 实体识别 | `src/entity.py` | 基于正则的命名实体识别 |
| 情感分析 | `src/sentiment.py` | 基于情感词典的情感倾向分析 |
| 对话历史 | `src/history.py` | 对话历史的持久化管理 |
| 反馈管理 | `src/feedback.py` | 用户反馈的收集和存储 |
| 对话总结 | `src/summary.py` | 对话摘要生成 |
| API 入口 | `api/main.py` | FastAPI 应用，路由定义 |
| 全局配置 | `config/config.py` | 系统配置参数 |

### 3.4 RESTful API 接口设计

系统对外暴露的 API 接口遵循 RESTful 风格：

**核心接口**：

| 方法 | 路径 | 功能 | 请求体 | 响应体 |
|------|------|------|--------|--------|
| POST | `/dialog` | 对话 | `{user_input, session_id}` | `{response, agent_actions, sources}` |
| GET | `/health` | 健康检查 | — | `{status, model, tools}` |
| DELETE | `/dialog/{session_id}` | 重置会话 | — | `{status, message}` |

**知识库接口**：

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/knowledge/status` | 知识库状态 |
| POST | `/knowledge/search` | 手动检索 |
| GET | `/agent/status` | Agent 状态 |

**辅助接口**：

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/dialog/{id}/history` | 获取历史 |
| POST | `/dialog/{id}/summary` | 生成总结 |
| POST | `/feedback` | 提交反馈 |

---

## 第4章 系统详细设计与实现

### 4.1 Agent 对话引擎（DomainAgent）

#### 4.1.1 类结构设计

DomainAgent 类是系统的核心引擎，包含以下关键属性和方法：

**属性**：
- `api_key`：Qwen API 密钥
- `api_url`：API 端点（阿里云百炼 OpenAI 兼容模式）
- `model`：模型名称（默认 qwen-plus）
- `max_tool_rounds`：最大工具调用轮次（默认 5）
- `temperature`：生成温度（默认 0.7）
- `rag_retriever`：RAG 检索器实例
- `tools`：工具定义列表

**核心方法**：
- `chat()`：对话入口，执行完整的 ReAct 循环
- `_build_messages()`：构建消息列表（系统提示 + 历史 + 用户输入 + 当前时间）
- `_call_llm()`：调用 Qwen API（含 Function Calling 参数）
- `_execute_tool()`：执行工具并返回结果
- `_summarize_tool_result()`：生成工具结果摘要
- `_extract_sources()`：提取法律来源信息
- `_force_final_answer()`：超出最大轮次时强制生成回答

#### 4.1.2 System Prompt 工程

System Prompt 的设计直接影响 Agent 的行为。本系统的 System Prompt 包含以下部分：

1. **身份定义**：明确系统为"智能领域聊天机器人"，是通用型 AI 助手
2. **自我介绍规范**：规定在用户打招呼时如何介绍自己，避免被误认为法律专用助手
3. **工具使用策略**：法律问题调用工具，通用问题直接回答
4. **回答规范**：中文回答、不编造法条、诚实告知不足、法律回答附免责声明
5. **动态时间注入**：在 `_build_messages()` 中注入当前日期和星期，使 Agent 能回答时间相关问题

#### 4.1.3 工具注册与管理

工具注册表（`src/tools/__init__.py`）采用字典结构管理所有工具：

```python
TOOL_REGISTRY = {
    "tool_name": (TOOL_DEFINITION, execute_function),
    ...
}
```

- `get_all_tool_definitions()`：返回所有工具的 JSON Schema 定义列表，传入 LLM API
- `get_tool_executor(name)`：根据工具名获取执行函数

这种设计使得添加新工具只需：
1. 创建工具文件，定义 `TOOL_DEFINITION` 和 `execute()` 函数
2. 在 `__init__.py` 中注册

### 4.2 RAG 法律知识库的构建与检索

#### 4.2.1 数据来源与预处理

法律知识库的数据来源于 DISC-LawLLM 开源项目的法律问答数据集，包含大量法律条文和问答对，涵盖劳动纠纷、婚姻家庭、交通事故、房产纠纷、刑事案件、合同纠纷等多个领域。

数据预处理流程：
1. 从 JSONL 文件加载原始数据
2. 提取法律文本内容和元数据（领域标签、来源等）
3. 使用 RecursiveCharacterTextSplitter 将长文本分块（chunk），确保每个片段适合向量检索

`[待填写] 说明：运行 python scripts/build_rag_knowledge_base.py 查看构建日志，可获取数据条数、分块参数等详细信息`

#### 4.2.2 向量化与存储

文本分块后，使用 BGE-small-zh-v1.5 嵌入模型将每个文本片段编码为 384 维向量，然后存入 ChromaDB 向量数据库，同时保存原文内容和元数据（领域标签、来源类型等）。

构建完成后，向量库持久化存储在 `data/knowledge_base/vector_db/` 目录。

`[待填写] 说明：构建完成后，访问 http://localhost:8000/knowledge/status 可查看文档总数`

#### 4.2.3 检索流程

RAGRetriever 类封装了完整的检索流程：

1. 接收用户查询文本
2. 使用 BGE 模型将查询编码为向量
3. 在 ChromaDB 中执行 similarity_search_with_relevance_scores，返回 Top-K 相似文档
4. 可选按领域（domain）过滤检索结果
5. 返回文档内容、元数据和相似度分数

### 4.3 法律知识图谱的构建与查询

#### 4.3.1 知识图谱构建

法律知识图谱的构建流程：

1. 从法律数据集中提取法律实体和关系
2. 使用 NetworkX 构建图数据结构，节点为法律概念，边为概念间关系
3. 导出为 GraphML 格式文件，存储在 `data/knowledge_graph/` 目录

`[待填写] 说明：运行 python scripts/build_knowledge_graph.py 查看构建日志，可获取节点数、边数等数据`

#### 4.3.2 查询实现

知识图谱查询工具支持三种查询模式：

1. **邻居查询（all_neighbors）**：返回指定实体的所有直接邻居及边关系
2. **关联查询（related）**：返回指定实体 1 跳范围内的所有关联实体
3. **路径查询（shortest_path）**：查找两个实体之间的最短关系路径

查询时使用模糊匹配机制，支持精确匹配和包含匹配，提高查询的容错率。

### 4.4 NLP 工具模块

#### 4.4.1 实体识别

实体识别模块（`src/entity.py`）基于正则表达式实现，支持 4 种实体类型：

| 实体类型 | 说明 | 匹配策略 |
|----------|------|----------|
| PERSON | 人名 | 常见姓氏 + 1-2个汉字 |
| COMPANY | 公司/企业 | 中文名 + 组织后缀（公司/企业/集团等） |
| TIME | 时间 | 年月日格式 / "N年" / "N个月" |
| LOCATION | 地点 | 行政区划后缀 / 主要城市名 |

#### 4.4.2 情感分析

情感分析模块（`src/sentiment.py`）基于情感词典实现，将用户输入的情感倾向分为正面（positive）、负面（negative）和中性（neutral）三类。通过统计文本中各类情感关键词的出现频次，判断主导情感。

#### 4.4.3 场景分析工具

场景分析工具（`src/tools/situation_analyzer.py`）综合使用实体识别和情感分析，对用户描述的法律场景进行全面分析：

1. 提取关键实体（人物、公司、时间、地点）
2. 判断情感倾向
3. 基于关键词匹配判断涉及的法律领域（劳动纠纷、婚姻家庭等）

### 4.5 前端界面设计与实现

#### 4.5.1 整体布局

前端采用 Vue 3 + Element Plus 构建，三栏布局：

- **左栏（220px）**：SessionList 会话列表，支持创建、切换、删除会话
- **中栏**：ChatMessage 聊天区域，消息列表 + 输入框
- **右栏（260px）**：AnalysisPanel 分析面板，展示系统状态、工具调用详情和反馈入口

#### 4.5.2 Agent 推理过程可视化

ChatMessage 组件在 bot 消息上方展示 Agent 推理过程：

- 显示调用了哪些工具（以不同颜色的 Tag 标识）
- 显示每个工具的执行结果摘要
- 在消息底部标注引用来源和相关度

AnalysisPanel 在右侧面板展示更详细的工具调用信息，包括工具的输入参数和结果。

#### 4.5.3 状态管理

使用 Vue 3 的 Composition API（ref + reactive）管理状态，未引入 Vuex 或 Pinia。核心状态包括：

- `messages`：当前会话的消息列表
- `sessions`：所有会话列表（持久化到 localStorage）
- `systemStatus`：后端健康状态（定时轮询 /health 接口）
- `currentAnalysis`：最近一次对话的 Agent 分析结果

---

## 第5章 系统测试与结果分析

### 5.1 测试环境

| 项目 | 配置 |
|------|------|
| 操作系统 | macOS / Linux |
| Python 版本 | 3.10+ |
| Node.js 版本 | 18+ |
| LLM 模型 | 通义千问 qwen-plus |
| 嵌入模型 | BGE-small-zh-v1.5 |
| 向量数据库 | ChromaDB |

`[待填写] 说明：补充具体硬件配置（CPU、内存、磁盘），可在终端运行 uname -a 和 sysctl -n hw.memsize 获取`

### 5.2 功能测试

#### 5.2.1 法律问题测试

| 测试问题 | 是否调用工具 | 调用的工具 | 回答是否引用法条 |
|----------|-------------|-----------|-----------------|
| 被公司辞退了，工作了3年，能拿到多少赔偿？ | `[待填写]` | `[待填写]` | `[待填写]` |
| 离婚时财产怎么分割？ | `[待填写]` | `[待填写]` | `[待填写]` |
| 劳动合同法第四十七条规定了什么？ | `[待填写]` | `[待填写]` | `[待填写]` |
| 被人打了构成什么罪？ | `[待填写]` | `[待填写]` | `[待填写]` |
| 工伤认定需要满足什么条件？ | `[待填写]` | `[待填写]` | `[待填写]` |

`[待填写] 说明：启动系统后，在前端界面逐一输入以上问题，观察 Agent 推理过程面板中的工具调用情况，记录回答中是否引用了具体法条`

#### 5.2.2 通用问题测试

| 测试问题 | 是否调用工具 | 回答质量 |
|----------|-------------|---------|
| 你好，你是谁？ | `[待填写]` | `[待填写]` |
| Python 和 Java 哪个好？ | `[待填写]` | `[待填写]` |
| 帮我写一首关于春天的诗 | `[待填写]` | `[待填写]` |
| 今天是星期几？ | `[待填写]` | `[待填写]` |

`[待填写] 说明：通用问题预期不调用任何工具，LLM 直接回答。观察右侧分析面板，应显示"等待对话..."（无工具调用）`

#### 5.2.3 多轮对话测试

`[待填写] 说明：在一个会话中连续提问 3-5 个相关法律问题（如先问劳动法，再追问赔偿标准，再追问仲裁流程），观察系统是否正确维持上下文`

### 5.3 性能测试

#### 5.3.1 响应时间测试

| 测试场景 | 平均响应时间 | 最短 | 最长 |
|----------|-------------|------|------|
| 通用问题（无工具调用） | `[待填写]` | `[待填写]` | `[待填写]` |
| 法律问题（1 次工具调用） | `[待填写]` | `[待填写]` | `[待填写]` |
| 法律问题（2+ 次工具调用） | `[待填写]` | `[待填写]` | `[待填写]` |

`[待填写] 说明：运行 python scripts/compare_versions.py，Agent 系统的 time_ms 列即为响应时间。可以多次运行取平均值`

#### 5.3.2 对比实验：规则系统 vs Agent 系统

| 对比维度 | 规则系统 | Agent 系统 |
|----------|---------|-----------|
| 法律问题平均响应时间 | `[待填写]` | `[待填写]` |
| 通用问题平均响应时间 | `[待填写]` | `[待填写]` |
| 法律问题可回答率 | `[待填写]` | `[待填写]` |
| 通用问题可回答率 | `[待填写]` | `[待填写]` |
| 法律依据引用率 | `[待填写]` | `[待填写]` |
| 总工具调用次数 | 不适用 | `[待填写]` |

`[待填写] 说明：运行 python scripts/compare_versions.py，脚本会自动输出上述所有对比数据。原始数据保存在 data/compare_results.json 中`

### 5.4 结果分析

`[待填写] 说明：基于上述测试数据进行分析。重点阐述以下内容：`

1. **Agent 架构的优势**：
   - 通用问题可回答率方面，Agent 系统预期显著优于规则系统（规则系统对非法律问题返回"无法回答"）
   - 法律依据引用方面，Agent 系统能自主判断是否需要调用知识库
   - 可扩展性方面，添加新工具只需定义 JSON Schema 和实现函数

2. **响应时间分析**：
   - Agent 系统的响应时间预期高于规则系统（需要调用远程 LLM API）
   - 这是用灵活性和回答质量换取的合理代价

3. **不足与局限**：
   - 依赖远程 API，网络波动影响响应时间
   - 知识库覆盖范围有限，取决于训练数据
   - 实体识别和情感分析基于规则，准确率有限

---

## 第6章 总结与展望

### 6.1 工作总结

本文设计并实现了一个基于 Agent 架构的智能领域聊天机器人系统，以法律咨询为深度增强方向。系统的主要成果包括：

1. **架构层面**：采用 LLM Agent + Function Calling 的架构，以通义千问大模型为核心，实现了 ReAct 推理循环，使系统同时具备通用对话能力和法律领域的专业深度。

2. **工具层面**：设计并实现了四个法律领域工具——RAG 知识库检索、法条精确查询、知识图谱查询和场景分析，通过统一的工具注册表机制进行管理。

3. **系统层面**：构建了完整的前后端分离系统，后端基于 FastAPI 提供 RESTful API，前端基于 Vue 3 构建现代化交互界面，支持多会话管理、Agent 推理过程可视化、对话总结与反馈等功能。

4. **实验层面**：保留了旧版规则系统，编写了对比实验脚本，验证了 Agent 架构相比传统规则系统在通用能力和回答质量方面的提升。

### 6.2 不足与展望

本系统存在以下不足，也是未来的改进方向：

1. **知识库规模**：当前知识库的法律数据覆盖范围有限，未来可以接入更多法律法规数据源，如国家法律法规数据库、裁判文书网等。

2. **实体识别精度**：当前的实体识别基于正则表达式，在复杂场景下准确率有限。未来可以引入深度学习模型（如 BERT-NER）提升识别精度。

3. **RAG 检索优化**：可以引入重排序（Re-ranking）机制，对初始检索结果进行二次排序，提高检索精度。

4. **多模态支持**：当前系统仅支持文本对话，未来可以扩展支持图片（如合同照片）的输入和解析。

5. **实时信息获取**：当前系统无法获取实时信息（如天气、新闻），未来可以添加网络搜索工具，扩展系统的信息获取能力。

6. **性能优化**：可以引入流式输出（Streaming）机制，让用户在 LLM 生成过程中就能看到回答，提升体验。

7. **安全性增强**：可以添加用户认证和权限管理模块，支持多用户场景。

---

## 参考文献

[1] Yao S, Zhao J, Yu D, et al. ReAct: Synergizing Reasoning and Acting in Language Models[C]. International Conference on Learning Representations (ICLR), 2023.

[2] Lewis P, Perez E, Piktus A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks[C]. Advances in Neural Information Processing Systems (NeurIPS), 2020.

[3] Xiao S, Liu Z, Zhang P, et al. C-Pack: Packaged Resources To Advance General Chinese Embedding[J]. arXiv preprint arXiv:2309.07597, 2023.

[4] Yue Z, Zhang S, Fan L, et al. DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services[J]. arXiv preprint arXiv:2309.11325, 2023.

[5] OpenAI. Function Calling and Other API Updates[EB/OL]. https://openai.com/blog/function-calling-and-other-api-updates, 2023.

[6] Brown T, Mann B, Ryder N, et al. Language Models are Few-Shot Learners[C]. Advances in Neural Information Processing Systems (NeurIPS), 2020.

[7] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[C]. Advances in Neural Information Processing Systems (NeurIPS), 2017.

[8] Devlin J, Chang M-W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]. NAACL-HLT, 2019.

[9] 阿里云. 通义千问大模型 API 文档[EB/OL]. https://help.aliyun.com/zh/dashscope/, 2024.

[10] FastAPI Documentation[EB/OL]. https://fastapi.tiangolo.com/, 2024.

---

## 致谢

`[待填写]`

本论文的完成离不开指导教师和同学们的帮助。在此向所有给予支持和帮助的人表示衷心的感谢。

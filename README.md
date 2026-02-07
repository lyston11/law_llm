# 智能领域聊天机器人

> 基于 Agent 架构的智能对话系统，精通法律领域

一个基于大语言模型（LLM）的智能聊天机器人系统，采用 **Agent 架构**，通过 Function Calling 机制让 LLM 自主决策是否调用领域工具。系统以**法律领域**作为深度增强方向，配备了 RAG 知识库检索、法条查询、知识图谱、场景分析等专业工具；同时也能处理日常对话、常识问答等通用问题。

---

## 目录

- [重要文档](#重要文档)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [从零部署（完整步骤）](#从零部署完整步骤)
  - [前置条件](#前置条件)
  - [第 1 步：克隆项目](#第-1-步克隆项目)
  - [第 2 步：还原数据文件](#第-2-步还原数据文件重要)
  - [第 3 步：创建 Python 虚拟环境](#第-3-步创建-python-虚拟环境)
  - [第 4 步：安装 Python 依赖](#第-4-步安装-python-依赖)
  - [第 5 步：安装前端依赖](#第-5-步安装前端依赖)
  - [第 6 步：配置 API Key](#第-6-步配置-api-key)
  - [第 7 步：启动系统](#第-7-步启动系统)
  - [第 8 步：访问系统](#第-8-步访问系统)
- [手动还原数据（备选方案）](#手动还原数据备选方案)
- [一键启动脚本详解](#一键启动脚本详解-startsh)
- [手动启动（不用脚本）](#手动启动不用脚本)
- [环境变量配置说明](#环境变量配置说明)
- [API 接口说明](#api-接口说明)
- [开发者指南](#开发者指南)
  - [运行对比实验](#运行对比实验规则系统-vs-agent-系统)
  - [运行单元测试](#运行单元测试)
  - [构建知识库](#从源数据重新构建知识库)
- [常见问题排查](#常见问题排查)
- [版本历史](#版本历史)

---

## 重要文档

| 文档 | 路径 | 说明 |
|------|------|------|
| **用户使用手册** | [`docs/user_manual.md`](docs/user_manual.md) | 系统功能说明、界面操作指南、使用场景示例 |
| **项目技术文档** | [`docs/technical_doc.md`](docs/technical_doc.md) | 架构设计、模块详解、API 接口、配置说明（开发者必读） |
| **毕业论文初稿** | [`docs/thesis_draft.md`](docs/thesis_draft.md) | 完整论文结构，`[待填写]` 处标注了数据获取方法 |
| **对比实验脚本** | [`scripts/compare_versions.py`](scripts/compare_versions.py) | 规则系统 vs Agent 系统的自动对比，输出可直接用于论文 |
| **快速启动指南** | [`docs/start_manual.md`](docs/start_manual.md) | 精简版启动指南 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                   前端 (Vue 3 + Element Plus)              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ 会话管理  │  │   聊天界面    │  │  分析面板         │    │
│  │SessionList│  │ ChatMessage  │  │ AnalysisPanel    │    │
│  └──────────┘  └──────────────┘  └──────────────────┘    │
└─────────────────────────┬────────────────────────────────┘
                          │ HTTP REST API (Axios)
                          │ POST /dialog, GET /health, ...
┌─────────────────────────▼────────────────────────────────┐
│                FastAPI 后端 (api/main.py)                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           DomainAgent (src/agent.py)  ⭐核心         │ │
│  │  ┌────────────────────────────────────────────────┐ │ │
│  │  │     Qwen-Plus LLM (Function Calling / ReAct)   │ │ │
│  │  │  用户输入 → 推理 → 调用工具(0~N次) → 最终回答   │ │ │
│  │  └────────┬──────────┬──────────┬──────────┬──────┘ │ │
│  │           │          │          │          │         │ │
│  │  ┌────────▼──┐ ┌─────▼────┐ ┌──▼───────┐ ┌▼─────┐  │ │
│  │  │知识库检索  │ │法条查询   │ │知识图谱   │ │场景   │  │ │
│  │  │(RAG+BGE)  │ │          │ │(Graph)   │ │分析   │  │ │
│  │  └─────┬─────┘ └────┬─────┘ └────┬─────┘ └──┬────┘  │ │
│  └────────┼────────────┼────────────┼───────────┼───────┘ │
│           │            │            │           │          │
│  ┌────────▼────────────▼──┐ ┌───────▼───┐ ┌────▼───────┐ │
│  │  RAGRetriever           │ │ NetworkX  │ │ Entity +   │ │
│  │  ChromaDB + BGE嵌入模型  │ │ GraphML   │ │ Sentiment  │ │
│  └────────────────────────┘ └───────────┘ └────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**工作流程**：用户在前端输入问题 → Axios 发送 POST 请求到后端 `/dialog` 接口 → `DomainAgent` 调用 Qwen LLM → LLM 判断是否需要调用工具 → 如果是法律问题则调用知识库/法条/知识图谱等工具获取专业知识 → LLM 结合工具结果生成最终回答 → 返回前端展示。

---

## 技术栈

| 层级 | 技术 | 版本要求 | 说明 |
|------|------|---------|------|
| **前端** | Vue 3 + Vite | Vue 3.5+, Vite 7+ | 现代前端框架，响应式 UI |
| | Element Plus | 2.13+ | UI 组件库（按钮、输入框、对话框等） |
| | Axios | 1.13+ | HTTP 请求库，与后端 API 通信 |
| **后端** | FastAPI | 0.100+ | 高性能异步 Python Web 框架 |
| | Uvicorn | 0.23+ | ASGI 服务器，运行 FastAPI |
| | Pydantic | 2.0+ | 数据校验和序列化 |
| **AI 核心** | 通义千问 (Qwen-Plus) | — | 阿里云大语言模型，支持 Function Calling |
| | Agent (ReAct 模式) | — | 推理-行动循环，LLM 自主决策工具调用 |
| **知识增强** | LangChain | 0.1+ | RAG 流程编排 |
| | ChromaDB | 0.4+ | 向量数据库，存储法律知识向量 |
| | BGE-small-zh-v1.5 | — | 中文文本嵌入模型（将文本转为向量） |
| | NetworkX | 3.0+ | 知识图谱，法律概念关系查询 |
| **NLP** | jieba | 0.42+ | 中文分词 |
| | 实体识别 | 自研 | 正则 + 规则，识别人名/公司/时间/地点 |
| | 情感分析 | 自研 | 基于情感词典的情感倾向判断 |

---

## 项目结构

```
law_llm/                               # 项目根目录
│
├── api/                                # ===== API 层 =====
│   └── main.py                         #   FastAPI 应用入口，所有 REST 接口定义
│
├── config/                             # ===== 配置 =====
│   ├── __init__.py                     #   配置包初始化
│   └── config.py                       #   全局配置（API Key、模型名、路径等）
│
├── src/                                # ===== 核心源码 =====
│   ├── __init__.py                     #   包初始化
│   ├── agent.py                        #   ⭐ DomainAgent（Agent 核心：ReAct + Function Calling）
│   ├── rag.py                          #   RAG 检索增强生成（ChromaDB + BGE 嵌入）
│   ├── entity.py                       #   命名实体识别（人名/公司/时间/地点）
│   ├── sentiment.py                    #   情感分析（正面/负面/中性）
│   ├── history.py                      #   对话历史管理（记录/导出/查询）
│   ├── feedback.py                     #   用户反馈管理（评分/评论）
│   ├── summary.py                      #   对话总结生成
│   │
│   ├── tools/                          #   ===== Agent 工具集 =====
│   │   ├── __init__.py                 #     工具注册表（统一管理4个工具的定义和执行函数）
│   │   ├── knowledge_search.py         #     工具1: 法律知识库检索（基于 RAG 向量检索）
│   │   ├── article_lookup.py           #     工具2: 法条精确查询（按法律名+条款号查询）
│   │   ├── knowledge_graph.py          #     工具3: 知识图谱查询（法律概念关系）
│   │   └── situation_analyzer.py       #     工具4: 法律场景分析（实体识别+情感分析+领域判断）
│   │
│   └── legacy/                         #   ===== 旧版规则系统（仅用于论文对比实验）=====
│       ├── __init__.py                 #     包说明
│       ├── dialog.py                   #     旧版对话管理器（规则流水线）
│       ├── intent.py                   #     TF-IDF 意图识别
│       ├── intent_bert.py              #     BERT 意图识别
│       ├── tfidf.py                    #     TF-IDF 计算器
│       ├── slot.py                     #     槽位填充
│       └── state.py                    #     对话状态跟踪
│
├── frontend/                           # ===== 前端 (Vue 3) =====
│   ├── index.html                      #   HTML 入口
│   ├── package.json                    #   前端依赖声明（vue, element-plus, axios 等）
│   ├── vite.config.js                  #   Vite 配置（开发服务器端口 3000）
│   └── src/
│       ├── main.js                     #   Vue 应用入口
│       ├── App.vue                     #   主组件（三栏布局 + 状态管理 + 消息发送逻辑）
│       ├── api/
│       │   └── index.js                #   API 接口封装（Axios，baseURL = localhost:8000）
│       └── components/
│           ├── ChatMessage.vue         #   聊天消息组件（含 Agent 推理过程展示 + 来源标注）
│           ├── SessionList.vue         #   左侧会话列表组件
│           └── AnalysisPanel.vue       #   右侧分析面板（系统状态 + 工具详情 + 反馈）
│
├── scripts/                            # ===== 脚本 =====
│   ├── compare_versions.py             #   ⭐ 版本对比实验脚本（规则系统 vs Agent 系统）
│   ├── build_rag_knowledge_base.py     #   从法律数据集构建 RAG 向量知识库
│   ├── build_knowledge_graph.py        #   从法律数据集构建知识图谱
│   ├── fine_tune_classifier.py         #   意图分类器训练脚本
│   └── fine_tune_legal_bert.py         #   BERT 微调脚本
│
├── data/                               # ===== 数据 =====
│   ├── knowledge_base.tar.gz.part_*    #   RAG 向量知识库压缩分片（共 10 片，约 877MB）
│   │                                   #   → 还原后：data/knowledge_base/
│   ├── knowledge_graph.tar.gz.part_*   #   法律知识图谱压缩分片（共 2 片，约 187MB）
│   │                                   #   → 还原后：data/knowledge_graph/
│   ├── legal/
│   │   ├── datasets.tar.gz             #   法律数据集压缩包（42MB）
│   │   │                               #   → 还原后：data/legal/datasets/
│   │   ├── corpus/                     #   语料库（意图识别训练数据）
│   │   └── scenario/                   #   场景配置（旧版规则系统用）
│   ├── annotation/                     #   意图标注数据
│   └── dataset/                        #   意图识别数据集
│
├── models/                             # ===== 模型 =====
│   ├── bge-small-zh-v1.5.tar.gz.part_* #   BGE 中文嵌入模型压缩分片（共 2 片，约 107MB）
│   │                                   #   → 还原后：models/bge-small-zh-v1.5/
│   └── intent_classifier/              #   TF-IDF 意图分类器（7MB，无需解压，直接可用）
│       ├── intent_classifier.joblib    #     训练好的分类器模型
│       └── label_map.json              #     意图标签映射
│
├── docs/                               # ===== 文档 =====
│   ├── user_manual.md                  #   ⭐ 用户使用手册
│   ├── technical_doc.md                #   ⭐ 项目技术文档（开发者必读）
│   ├── thesis_draft.md                 #   ⭐ 毕业论文初稿
│   ├── start_manual.md                 #   快速启动指南
│   ├── trigger_questions.md            #   测试问题集
│   └── thesis/                         #   论文相关原始文件（开题报告、任务书等）
│
├── tests/                              # ===== 测试 =====
│   ├── test_entity.py                  #   实体识别单元测试
│   ├── test_sentiment.py               #   情感分析单元测试
│   ├── test_feedback.py                #   反馈功能测试
│   ├── test_history.py                 #   历史记录测试
│   ├── test_dialog.py                  #   对话功能测试（使用 legacy 模块）
│   ├── test_tfidf.py                   #   TF-IDF 测试
│   ├── test_intent_comparison.py       #   意图识别对比测试
│   ├── test_intent_comprehensive.py    #   意图识别综合测试
│   └── manual/                         #   手动测试脚本（需要 API Key）
│       ├── test_qwen_api.py            #     测试 Qwen API 连通性
│       ├── test_dialog.py              #     测试完整对话流程
│       └── ...                         #     其他场景测试
│
├── restore_data.sh                     # ⭐ 数据还原脚本（克隆后必须第一步运行）
├── start.sh                            # ⭐ 一键启动脚本（自动检查环境 + 启动前后端）
├── .env.example                        # 环境变量模板（复制为 .env 后填入 API Key）
├── .env                                # 环境变量（已被 .gitignore 忽略，不会提交）
├── .gitignore                          # Git 忽略规则
├── requirements.txt                    # Python 依赖列表
├── pytest.ini                          # Pytest 配置
└── README.md                           # 本文件
```

---

## 从零部署（完整步骤）

> **目标**：按照以下步骤操作，任何人（包括零基础的同学）都能在本地跑起来。
>
> **预计耗时**：首次部署约 15-30 分钟（取决于网速）。

### 前置条件

需要在电脑上安装以下 3 个工具：

| 工具 | 最低版本 | 用途 | 安装方式 |
|------|---------|------|---------|
| **Python** | 3.10 | 后端运行环境 | [Miniconda（推荐）](https://docs.conda.io/en/latest/miniconda.html) 或 [Python 官网](https://www.python.org/) |
| **Node.js** | 18 | 前端运行环境 | [Node.js 官网](https://nodejs.org/)，下载 LTS 版本 |
| **Git** | 任意 | 克隆代码 | [Git 官网](https://git-scm.com/) |

**验证安装**（打开终端，逐条执行）：

```bash
python3 --version
# 期望输出: Python 3.10.x 或更高 (如 Python 3.11.5)

node --version
# 期望输出: v18.x.x 或更高 (如 v20.11.0)

npm --version
# 期望输出: 9.x.x 或更高 (如 10.2.4)

git --version
# 期望输出: git version 2.x.x (如 git version 2.39.2)
```

> **注意**：如果使用 Conda，后续所有 `python3` 命令可能需要换成 `python`（取决于你的 Conda 配置）。

---

### 第 1 步：克隆项目

```bash
git clone git@github.com:lyston11/law_llm.git
cd law_llm
```

> **如果 SSH 克隆失败**（提示 `Permission denied`），改用 HTTPS：
> ```bash
> git clone https://github.com/lyston11/law_llm.git
> cd law_llm
> ```

克隆完成后，确认目录结构：

```bash
ls
# 期望看到: README.md  api/  config/  data/  docs/  frontend/
#           models/  requirements.txt  restore_data.sh  scripts/
#           src/  start.sh  tests/  ...
```

---

### 第 2 步：还原数据文件（重要）

> **为什么需要这一步？**
> 因为 GitHub 对单个文件有 100MB 的大小限制，项目中的模型文件和向量数据库（总计约 1.2GB）被压缩成多个小文件存储在仓库中。克隆后需要将它们还原为原始目录。

**运行还原脚本**：

```bash
bash restore_data.sh
```

**期望输出**：

```
============================================================
  智能领域聊天机器人 — 数据还原脚本
============================================================

[INFO]  项目目录: /path/to/law_llm

[INFO]  ========== 还原: BGE 嵌入模型 (models/bge-small-zh-v1.5/) ==========
[INFO]  合并分片文件...
[OK]    合并完成: models/bge-small-zh-v1.5.tar.gz (107M)
[INFO]  解压到 models/ ...
[OK]    解压完成: models/bge-small-zh-v1.5
[OK]    验证通过: models/bge-small-zh-v1.5 (XX 个文件)

[INFO]  ========== 还原: RAG 向量知识库 (data/knowledge_base/) ==========
[INFO]  合并分片文件...
[OK]    合并完成: data/knowledge_base.tar.gz (877M)
[INFO]  解压到 data/ ...
[OK]    解压完成: data/knowledge_base
[OK]    验证通过: data/knowledge_base (XX 个文件)

[INFO]  ========== 还原: 法律知识图谱 (data/knowledge_graph/) ==========
...

[INFO]  ========== 还原: 法律数据集 (data/legal/datasets/) ==========
...

============================================================
  数据还原完成！
============================================================
```

**还原了什么？4 个压缩包对应 4 个目录：**

| 压缩包文件 | 大小 | 还原后的目录 | 用途 |
|-----------|------|-------------|------|
| `models/bge-small-zh-v1.5.tar.gz.part_aa`<br>`models/bge-small-zh-v1.5.tar.gz.part_ab` | 共 107MB（2片） | `models/bge-small-zh-v1.5/` | BGE 中文嵌入模型，将文本转为向量用于知识库检索 |
| `data/knowledge_base.tar.gz.part_aa`<br>到 `part_aj` | 共 877MB（10片） | `data/knowledge_base/` | ChromaDB 向量数据库，存储已向量化的法律知识 |
| `data/knowledge_graph.tar.gz.part_aa`<br>`data/knowledge_graph.tar.gz.part_ab` | 共 187MB（2片） | `data/knowledge_graph/` | 法律知识图谱（GraphML 格式），存储法律概念关系 |
| `data/legal/datasets.tar.gz` | 42MB（1个文件） | `data/legal/datasets/` | 原始法律数据集（JSONL 格式），用于构建知识库的源数据 |

> **脚本特性**：
> - 如果目标目录已存在（说明已经还原过），会自动跳过
> - 可以重复运行，不会重复解压
> - 合并后的临时 `.tar.gz` 文件会自动清理

如果 `restore_data.sh` 无法运行（例如 Windows 没有 Bash），请参考下方 [手动还原数据](#手动还原数据备选方案) 章节。

---

### 第 3 步：创建 Python 虚拟环境

> **为什么需要虚拟环境？** 避免不同项目的 Python 包版本冲突。

**方式 A：使用 Conda（推荐）**

```bash
conda create -n bishe python=3.10 -y
conda activate bishe
```

验证：

```bash
python --version
# 期望输出: Python 3.10.x

which python
# 期望输出: /path/to/miniconda3/envs/bishe/bin/python（说明在虚拟环境中）
```

**方式 B：使用 venv**

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

# Windows CMD
# .\.venv\Scripts\activate.bat
```

验证：

```bash
which python3
# 期望输出: /path/to/law_llm/.venv/bin/python3（说明在虚拟环境中）
```

> **后续每次使用都需要先激活虚拟环境**，否则会找不到已安装的包。

---

### 第 4 步：安装 Python 依赖

> 确保虚拟环境已激活（终端提示符前有 `(bishe)` 或 `(.venv)`）

```bash
pip install -r requirements.txt
```

**期望结果**：安装约 30 个 Python 包（包括 FastAPI、PyTorch、LangChain、ChromaDB 等），首次安装需要 3-10 分钟。

验证安装成功：

```bash
python -c "import fastapi, uvicorn, torch, chromadb; print('所有核心依赖安装成功')"
# 期望输出: 所有核心依赖安装成功
```

> **国内网络加速**（如果 pip 下载很慢）：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

> **常见错误**：如果安装 `torch` 时报错或下载太慢，可以先单独安装 CPU 版本：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

### 第 5 步：安装前端依赖

```bash
cd frontend
npm install
cd ..
```

**期望结果**：`frontend/` 目录下会出现 `node_modules/` 文件夹（约 100MB）。

验证：

```bash
ls frontend/node_modules/ | head -5
# 期望看到一些包名，说明 node_modules 已创建
```

> **npm 加速**（国内推荐）：
> ```bash
> npm config set registry https://registry.npmmirror.com
> cd frontend && npm install && cd ..
> ```

---

### 第 6 步：配置 API Key

本项目调用**阿里云百炼（通义千问）** 的大模型 API，需要一个 API Key。

**6.1 获取 API Key**

1. 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 注册/登录阿里云账号
3. 进入百炼控制台 → 左侧菜单「API-KEY 管理」→ 创建新的 API Key
4. 复制 API Key（格式类似 `sk-xxxxxxxxxxxxxxxxxxxxxxxx`）

> **费用说明**：新用户通常有免费试用额度，足够测试使用。

**6.2 配置到项目中**

```bash
# 复制环境变量模板
cp .env.example .env
```

然后用任意文本编辑器打开 `.env` 文件，将 `sk-your-api-key-here` 替换为你的真实 API Key：

```ini
# .env 文件内容（修改第 4 行的 API_KEY）

# 阿里云百炼（DashScope）API 密钥
API_KEY=sk-你的真实API密钥粘贴到这里

# API 地址（不需要修改）
API_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation

# Agent 配置（可选，一般不需要修改）
AGENT_MODEL=qwen-plus
AGENT_MAX_TOOL_ROUNDS=5
AGENT_TEMPERATURE=0.7

# 日志配置
LOG_LEVEL=INFO
```

> **安全提醒**：`.env` 文件已在 `.gitignore` 中排除，**不会**被提交到 Git 仓库，你的 API Key 是安全的。

---

### 第 7 步：启动系统

有两种启动方式：

#### 方式 A：一键启动（推荐）

```bash
bash start.sh
```

脚本会自动检测 Conda 环境、完成环境检查、端口清理、启动后端和前端。详见 [一键启动脚本详解](#一键启动脚本详解-startsh)。

> **注意**：如果你使用 Conda，脚本会自动检测并激活名为 `bishe` 的环境，**不需要手动 `conda activate`**。

#### 方式 B：手动启动

需要打开 **两个终端窗口**，分别启动后端和前端。详见 [手动启动](#手动启动不用脚本)。

---

### 第 8 步：访问系统

启动成功后，打开浏览器访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| **前端界面** | http://localhost:3000 | 主要交互界面，在这里和机器人对话 |
| **后端 API** | http://localhost:8000 | REST API 服务（一般不需要直接访问） |
| **API 文档** | http://localhost:8000/docs | Swagger 交互式 API 文档（开发者可用来测试接口） |
| **ReDoc 文档** | http://localhost:8000/redoc | 另一种风格的 API 文档 |

**验证系统正常运行**：

1. 打开 http://localhost:3000，应该看到聊天界面
2. 右侧分析面板的「服务状态」应该显示「正常」
3. 在输入框输入「你好」，点击发送，应该收到回复
4. 输入一个法律问题如「被公司辞退了怎么办」，应该能看到 Agent 调用了知识库检索工具

**停止服务**：在终端按 `Ctrl + C`。

---

## 手动还原数据（备选方案）

> 如果 `restore_data.sh` 无法运行（如 Windows 无 Bash 环境），可按以下步骤手动操作。
> Windows 用户也可以在 **Git Bash** 或 **WSL** 中运行 `restore_data.sh`。

**1. 还原 BGE 嵌入模型**（2 个分片 → `models/bge-small-zh-v1.5/`）

```bash
# 合并 2 个分片为一个 tar.gz
cat models/bge-small-zh-v1.5.tar.gz.part_aa models/bge-small-zh-v1.5.tar.gz.part_ab > models/bge-small-zh-v1.5.tar.gz

# 解压到 models/ 目录
tar -xzf models/bge-small-zh-v1.5.tar.gz -C models/

# 删除临时的 tar.gz（节省空间，可选）
rm models/bge-small-zh-v1.5.tar.gz

# 验证
ls models/bge-small-zh-v1.5/
# 期望看到: config.json  model.safetensors  tokenizer.json  等文件
```

**2. 还原 RAG 向量知识库**（10 个分片 → `data/knowledge_base/`）

```bash
# 合并 10 个分片（aa 到 aj）
cat data/knowledge_base.tar.gz.part_* > data/knowledge_base.tar.gz

# 解压
tar -xzf data/knowledge_base.tar.gz -C data/

# 清理
rm data/knowledge_base.tar.gz

# 验证
ls data/knowledge_base/
# 期望看到: vector_db/ 目录
```

**3. 还原法律知识图谱**（2 个分片 → `data/knowledge_graph/`）

```bash
cat data/knowledge_graph.tar.gz.part_* > data/knowledge_graph.tar.gz
tar -xzf data/knowledge_graph.tar.gz -C data/
rm data/knowledge_graph.tar.gz

# 验证
ls data/knowledge_graph/
# 期望看到: law_knowledge_graph.graphml 文件
```

**4. 还原法律数据集**（单个文件 → `data/legal/datasets/`）

```bash
# 这个不需要合并，直接解压
tar -xzf data/legal/datasets.tar.gz -C data/legal/

# 验证
ls data/legal/datasets/
# 期望看到: DISC-Law-SFT-Pair-QA-released.jsonl 等文件
```

---

## 一键启动脚本详解 (`start.sh`)

`start.sh` 是一个 Bash 脚本，自动完成从环境检查到启动服务的全过程。

**使用方法**：

```bash
bash start.sh
```

**脚本执行流程**：

| 步骤 | 操作 | 检查什么 | 失败会怎样 |
|------|------|---------|-----------|
| 预处理 | **自动激活 Conda 环境** | 检测是否安装了 Conda，且存在名为 `bishe` 的环境 | 如果没有 Conda 或没有 `bishe` 环境，跳过（使用当前 Python） |
| 1 | 检查环境变量 | `.env` 文件是否存在、`API_KEY` 是否已填写 | 如果没有 `.env`，自动从模板创建并提示填写 API Key 后退出 |
| 2 | 检查 Python | Python 是否安装、版本是否 >= 3.10 | 报错并退出，提示安装 Python |
| 3 | 检查 Node.js | Node.js 和 npm 是否安装、版本是否 >= 18 | 报错并退出，提示安装 Node.js |
| 4 | 安装 Python 依赖 | `import fastapi, uvicorn, requests, dotenv` 是否成功 | 自动执行 `pip install -r requirements.txt` |
| 5 | 安装前端依赖 | `frontend/node_modules/` 是否存在 | 自动执行 `npm install` |
| 6 | 检查模型文件 | `models/bge-small-zh-v1.5/` 目录是否存在 | **仅警告**，不退出。系统仍可启动，但知识库检索功能不可用 |
| 7 | 检查知识库 | `data/knowledge_base/vector_db/` 和 `data/knowledge_graph/` 是否存在 | **仅警告**，不退出。系统仍可启动，但法律工具功能不可用 |
| 8 | 检查端口 | 8000 和 3000 端口是否被占用 | 自动 kill 占用进程，释放端口 |
| 9 | 启动后端 | 运行 `uvicorn api.main:app --port 8000`，等待健康检查通过 | 如果 30 秒内健康检查不通过，报错退出 |
| 10 | 启动前端 | 在 `frontend/` 目录运行 `npm run dev` | 启动 Vite 开发服务器，监听 3000 端口 |

> **Conda 用户**：脚本会自动执行 `conda activate bishe`，不需要手动激活。如果你的 Conda 环境名不是 `bishe`，可以修改 `start.sh` 顶部的 `CONDA_ENV_NAME` 变量。

**成功启动后的输出**：

```
[INFO]  项目目录: /path/to/law_llm
[INFO]  检测到 Conda 环境 'bishe'，正在激活...
[OK]    Conda 环境 'bishe' 已激活

[INFO]  ========== 第 1 步: 检查环境变量 ==========
[OK]    .env 配置已就绪

[INFO]  ========== 第 2 步: 检查 Python 环境 ==========
[OK]    Python 3.10.19 (python)

[INFO]  ========== 第 3 步: 检查 Node.js 环境 ==========
[OK]    Node.js v20.11.0 + npm 10.2.4

[INFO]  ========== 第 4 步: 安装 Python 依赖 ==========
[OK]    Python 核心依赖已安装

[INFO]  ========== 第 5 步: 安装前端依赖 ==========
[OK]    前端依赖已安装

[INFO]  ========== 第 6 步: 检查模型文件 ==========
[OK]    BGE 嵌入模型已存在

[INFO]  ========== 第 7 步: 检查知识库 ==========
[OK]    RAG 向量知识库已构建
[OK]    法律知识图谱已构建

[INFO]  ========== 第 8 步: 检查端口 ==========
[OK]    端口 8000 可用
[OK]    端口 3000 可用

[INFO]  ========== 第 9 步: 启动后端服务 ==========
[OK]    后端服务已就绪: http://127.0.0.1:8000

[INFO]  ========== 第 10 步: 启动前端服务 ==========
============================================================
  智能领域聊天机器人 — 启动成功！
============================================================

  前端界面:       http://localhost:3000
  后端 API:       http://localhost:8000
  API 文档:       http://localhost:8000/docs

  后端 PID: 12345
  前端 PID: 12346

  按 Ctrl+C 停止所有服务
```

**注意事项**：
- 按 `Ctrl+C` 会同时停止前后端服务（脚本注册了 trap 信号处理）
- 如果检查 API Key 时退出，编辑好 `.env` 后重新运行 `bash start.sh` 即可
- 步骤 6 和 7 不会阻止启动——即使没有知识库，系统也能正常运行通用对话，只是法律问题的回答不会引用知识库

---

## 手动启动（不用脚本）

如果不想用 `start.sh`，也可以手动启动。需要打开 **两个终端窗口**。

### 终端 1：启动后端

```bash
# 1. 进入项目目录
cd /path/to/law_llm

# 2. 激活虚拟环境
conda activate bishe          # 如果用 Conda
# source .venv/bin/activate   # 如果用 venv

# 3. 启动后端
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**期望输出**：

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     🚀 智能领域聊天机器人 v3.0.0 启动成功
INFO:        架构: Agent (LLM Function Calling)
INFO:        模型: qwen-plus
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**验证后端正常**：在浏览器打开 http://localhost:8000/health ，应该看到类似：

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

### 终端 2：启动前端

```bash
# 1. 进入前端目录
cd /path/to/law_llm/frontend

# 2. 启动前端开发服务器
npm run dev
```

**期望输出**：

```
  VITE v7.x.x  ready in XXX ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: http://192.168.x.x:3000/
  ➜  press h + enter to show help
```

现在打开浏览器访问 http://localhost:3000 即可使用。

---

## 环境变量配置说明

所有配置项都在 `.env` 文件中（从 `.env.example` 复制）。

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `API_KEY` | **是** | — | 阿里云百炼 API Key，在 [百炼控制台](https://bailian.console.aliyun.com/) 创建 |
| `API_URL` | 否 | `https://dashscope.aliyuncs.com/...` | DashScope API 地址，一般不需要修改 |
| `AGENT_MODEL` | 否 | `qwen-plus` | Agent 使用的模型名称。可选：`qwen-turbo`（快但弱）、`qwen-plus`（推荐）、`qwen-max`（强但贵） |
| `AGENT_MAX_TOOL_ROUNDS` | 否 | `5` | Agent 最大工具调用轮次，防止 LLM 无限循环调用工具 |
| `AGENT_TEMPERATURE` | 否 | `0.7` | 生成温度（0~1），越低回答越确定，越高越有创意 |
| `LOG_LEVEL` | 否 | `INFO` | 日志级别：`DEBUG`（最详细）、`INFO`、`WARNING`、`ERROR` |

> **最简配置**：只需要填 `API_KEY` 一项，其他全部使用默认值即可。

---

## API 接口说明

后端提供以下 REST API 接口（完整文档可在 http://localhost:8000/docs 查看）：

### 核心接口

| 方法 | 路径 | 说明 | 请求体示例 |
|------|------|------|-----------|
| `POST` | `/dialog` | **对话接口**（最核心） | `{"user_input": "被辞退怎么办", "session_id": "s1"}` |
| `GET` | `/health` | 健康检查 | — |

**POST /dialog 响应示例**：

```json
{
  "response": "根据《劳动合同法》第四十七条的规定...",
  "session_id": "s1",
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
  ]
}
```

### 辅助接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/knowledge/status` | 知识库状态（是否就绪、文档数量） |
| `POST` | `/knowledge/search` | 手动搜索知识库 |
| `GET` | `/agent/status` | Agent 详细状态（模型名、工具列表、知识库状态） |
| `GET` | `/dialog/{session_id}/history` | 获取某个会话的对话历史 |
| `GET` | `/dialog/{session_id}/history/export` | 导出某个会话的对话历史 |
| `DELETE` | `/dialog/{session_id}` | 重置（清空）某个会话 |
| `GET` | `/sessions` | 获取所有会话列表 |
| `POST` | `/feedback` | 提交用户反馈（1-5 星评分 + 评论） |
| `GET` | `/feedback/{session_id}` | 获取某个会话的反馈 |
| `GET` | `/feedback/stats` | 反馈统计 |
| `POST` | `/dialog/{session_id}/summary` | 生成某个会话的对话总结 |
| `DELETE` | `/dialog/{session_id}/summary` | 删除某个会话的总结 |

---

## 开发者指南

### 运行对比实验（规则系统 vs Agent 系统）

项目保留了 v1.0/v2.0 的规则对话系统（`src/legacy/` 目录），可以与当前 v3.0 Agent 系统做对比实验。这个实验的结果可以直接用于毕业论文第五章。

**前提条件**：
- `.env` 中已配置 `API_KEY`
- 虚拟环境已激活
- 依赖已安装
- 数据已还原（`restore_data.sh`）

**运行命令**：

```bash
python scripts/compare_versions.py
```

**脚本做了什么？**

1. 准备 12 个测试问题（7 个法律问题 + 5 个通用问题）
2. 用**旧版规则系统**（`src/legacy/dialog.py`）逐个回答这些问题，记录响应时间和结果
3. 用**新版 Agent 系统**（`src/agent.py`）逐个回答同样的问题，记录响应时间、结果和工具调用
4. 输出逐题对比表格（编号、问题类别、响应时间、是否可答、工具调用情况）
5. 输出分类统计（法律问题和通用问题分别的平均响应时间、可回答率、法律依据引用率）
6. 输出总体统计
7. 展示 3 个问题的回答内容对比示例
8. 将原始数据保存到 `data/compare_results.json`

**期望输出**（示例）：

```
============================================================
  智能领域聊天机器人 — 版本对比实验
  规则系统 (v1.0) vs Agent 系统 (v3.0)
============================================================

测试问题数: 12
  法律问题: 7
  通用问题: 5

============================================================
  测试旧版规则系统 (src/legacy/dialog.py)
============================================================
✅ 旧版 DialogManager 初始化成功

  [1/12] 法律-劳动纠纷: 我被公司辞退了...        ✓ 1234ms
  [2/12] 法律-婚姻家庭: 离婚时财产怎么分割？...    ✓ 890ms
  ...

============================================================
  测试新版 Agent 系统 (src/agent.py)
============================================================
✅ DomainAgent 初始化成功

  [1/12] 法律-劳动纠纷: 我被公司辞退了...        ✓ 3456ms | 工具: search_legal_knowledge
  [2/12] 法律-婚姻家庭: 离婚时财产怎么分割？...    ✓ 2890ms | 工具: search_legal_knowledge
  ...

============================================================
  对 比 结 果
============================================================

### 逐题对比
编号 | 问题类别      | 规则系统(ms) | Agent(ms) | 规则可答 | Agent可答 | Agent工具调用
...

### 分类统计
  【法律问题】（共 7 题）
    指标               | 规则系统    | Agent系统
    -----------------+-----------+-----------
    平均响应时间       |   XXX.Xms |   XXX.Xms
    可回答率           |    XX.X%  |   100.0%
    法律依据引用率     |    XX.X%  |    XX.X%

  【通用问题】（共 5 题）
    ...

📄 原始数据已保存到: data/compare_results.json
   可用于论文中的表格和图表数据填充
```

**论文中如何使用这些数据**：
- `data/compare_results.json` 包含每个问题的详细结果（响应时间、回答内容、工具调用等）
- 论文初稿 `docs/thesis_draft.md` 的第五章中标注了 `[待填写]` 的位置，将对比数据填入即可
- 对比表格可以直接从终端输出复制到论文

> **注意**：对比实验需要调用 Qwen API，12 个问题每个系统各调用一次（旧版也用 Qwen 生成最终回答），总共约 24 次 API 调用。

---

### 运行单元测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_entity.py -v        # 实体识别
pytest tests/test_sentiment.py -v     # 情感分析
pytest tests/test_feedback.py -v      # 反馈功能
pytest tests/test_history.py -v       # 历史记录

# 运行手动测试（需要 API Key）
python tests/manual/test_qwen_api.py  # 测试 Qwen API 是否正常连通
```

---

### 从源数据重新构建知识库

如果想从零重新构建 RAG 知识库或知识图谱（而不是用还原的数据），可以运行以下脚本：

```bash
# 构建 RAG 向量知识库（需要 BGE 模型和法律数据集）
python scripts/build_rag_knowledge_base.py
# 耗时约 30-60 分钟，输出到 data/knowledge_base/vector_db/

# 构建法律知识图谱
python scripts/build_knowledge_graph.py
# 耗时约 5-10 分钟，输出到 data/knowledge_graph/
```

---

## 常见问题排查

### Q: `restore_data.sh` 报错怎么办？

**方案 1**：Windows 用户请在 **Git Bash** 或 **WSL（Windows Subsystem for Linux）** 中运行。

**方案 2**：按照 [手动还原数据](#手动还原数据备选方案) 章节逐步手动操作。

**方案 3**：如果报 `Permission denied`：
```bash
chmod +x restore_data.sh
bash restore_data.sh
```

---

### Q: `start.sh` 提示 "请先在 .env 文件中填入真实的 API_KEY"

说明 `.env` 文件中的 `API_KEY` 还是模板值。解决方法：

```bash
# 用编辑器打开 .env
nano .env      # 或 vim .env 或用 VS Code 打开

# 将 API_KEY=sk-your-api-key-here 改为：
# API_KEY=sk-你的真实密钥
```

API Key 在 [阿里云百炼平台](https://bailian.console.aliyun.com/) 创建。

---

### Q: 启动后端时报 `ModuleNotFoundError: No module named 'xxx'`

**原因**：虚拟环境没有激活，或者依赖没有安装完整。

```bash
# 1. 确认虚拟环境已激活
conda activate bishe
# 或 source .venv/bin/activate

# 2. 确认在项目根目录
pwd
# 应该是 /path/to/law_llm

# 3. 重新安装依赖
pip install -r requirements.txt
```

---

### Q: 前端页面显示"服务状态: 离线"

说明前端无法连接到后端 API。检查步骤：

1. 确认后端正在运行（另一个终端能看到 uvicorn 的日志）
2. 在浏览器访问 http://localhost:8000/health ，确认返回 JSON
3. 如果后端运行在其他端口，需要修改 `frontend/src/api/index.js` 中的 `baseURL`

---

### Q: 前端白屏或报错

```bash
# 1. 确认 node_modules 已安装
ls frontend/node_modules/
# 如果不存在，运行：
cd frontend && npm install && cd ..

# 2. 确认 Vite 配置正确
# 前端默认监听 3000 端口（在 frontend/vite.config.js 中配置）
```

---

### Q: 知识库显示"未加载"

说明模型文件或向量数据库不存在。解决方法：

```bash
# 运行数据还原脚本
bash restore_data.sh

# 验证目录存在
ls models/bge-small-zh-v1.5/        # 应该有文件
ls data/knowledge_base/vector_db/   # 应该有文件
ls data/knowledge_graph/            # 应该有 .graphml 文件
```

> **注意**：没有知识库时系统仍可正常运行。LLM 通用问答功能不受影响，只是法律问题不会引用知识库中的法条。

---

### Q: API Key 在哪里获取？费用多少？

访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)，注册并创建 API Key。

- **新用户**通常有免费试用额度（百万 token 级别）
- `qwen-plus` 模型的价格很便宜，日常测试花费极低
- 可以在百炼控制台查看用量和费用

---

### Q: `pip install` 时下载太慢

使用国内镜像源：

```bash
# 清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或 阿里源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

---

### Q: `npm install` 时下载太慢

```bash
# 设置 npm 镜像（全局配置，只需要一次）
npm config set registry https://registry.npmmirror.com

# 然后重新安装
cd frontend && npm install && cd ..
```

---

### Q: 端口 8000 或 3000 被占用

```bash
# 查看占用端口的进程
lsof -i :8000
lsof -i :3000

# 杀掉占用进程
kill -9 <PID>

# 或者用 start.sh，脚本会自动清理端口
```

---

### Q: macOS 上 `lsof` 命令报权限错误

可能需要用 `sudo`：

```bash
sudo lsof -i :8000
```

或者直接用 `start.sh`，它会自动处理。

---

### Q: 对话回答很慢（超过 10 秒）

回答速度取决于：
1. **Qwen API 响应速度**：通常 2-5 秒，高峰期可能更慢
2. **工具调用次数**：如果 Agent 调用了多个工具，每次工具调用会额外花费时间
3. **网络状况**：API 服务器在阿里云，国内访问通常较快

这是正常现象，不需要担心。

---

## 版本历史

| 版本 | 日期 | 架构 | 主要变化 |
|------|------|------|---------|
| **v3.0.0** | 2026-02 | Agent (ReAct + Function Calling) | 全新 Agent 架构，LLM 自主决策工具调用，支持通用对话 |
| v2.0.0 | 2026-01 | 规则 + RAG | 集成 RAG 知识库检索、知识图谱 |
| v1.0.0 | 2025-12 | 纯规则 | TF-IDF 意图识别、槽位填充、规则匹配 |

---

## 许可证

本项目仅用于学术研究和毕业设计。

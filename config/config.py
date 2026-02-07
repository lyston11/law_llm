import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """主配置类"""

    # ==================== 项目信息 ====================
    PROJECT_NAME = "智能领域聊天机器人"
    PROJECT_VERSION = "3.0.0"

    # ==================== 路径配置 ====================
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    # ==================== API 配置 ====================
    API_KEY = os.getenv("API_KEY", "")
    API_URL = os.getenv("API_URL", "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")

    # ==================== 日志配置 ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # ==================== Agent 配置（v3.0 核心）====================
    # Agent 使用的 LLM 模型（qwen-plus 对 function calling 支持更好）
    AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen-plus")
    # Agent 最大工具调用轮次（防止无限循环）
    AGENT_MAX_TOOL_ROUNDS = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "5"))
    # Agent 生成温度（0-1，越低越确定性）
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
    # 对话历史保留轮数
    MAX_DIALOG_HISTORY = 10

    # ==================== RAG 知识库配置 ====================
    KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "knowledge_base", "vector_db")
    KNOWLEDGE_GRAPH_DIR = os.path.join(DATA_DIR, "knowledge_graph")
    RAG_EMBEDDING_MODEL = os.path.join(MODELS_DIR, "bge-small-zh-v1.5")
    RAG_TOP_K = 5
    USE_RAG = True

    # ==================== 数据目录 ====================
    LEGAL_DATASETS_DIR = os.path.join(DATA_DIR, "legal", "datasets")
    LEGAL_CORPUS_DIR = os.path.join(DATA_DIR, "legal", "corpus")
    SCENARIO_DIR = os.path.join(DATA_DIR, "legal", "scenario")
    ANNOTATION_DIR = os.path.join(DATA_DIR, "annotation")
    DATASET_DIR = os.path.join(DATA_DIR, "dataset")

    # ====================================================================
    # 以下为旧版规则系统（v1.0/v2.0）的配置，仅供 src/legacy/ 模块使用
    # 新版 Agent 架构不依赖这些配置
    # ====================================================================

    # 意图识别配置（旧版）
    INTENT_SWITCH_THRESHOLD = 0.2
    USE_BERT_INTENT = True
    BERT_MODEL_NAME = os.path.join(MODELS_DIR, "bge-large-zh-v1.5")
    INTENT_CONFIDENCE_THRESHOLD = 0.2
    HIERARCHICAL_INTENT_RECOGNITION = True

    # 槽位优先级（旧版）
    SLOT_PRIORITY = {
        '#法律类型#': 1,
        '#劳动问题类型#': 2, '#婚姻问题类型#': 2, '#事故类型#': 2,
        '#房产问题类型#': 2, '#知识产权类型#': 2, '#刑事罪名#': 2,
        '#行政案件类型#': 2, '#合同类型#': 2,
        '#用人单位#': 3, '#婚姻时长#': 3, '#责任方#': 3,
        '#工作时长#': 4, '#房屋位置#': 4, '#合同标的#': 4,
    }

    # 法律类型与子类型映射（旧版）
    LEGAL_SUBTYPE_MAP = {
        '劳动': '#劳动问题类型#', '婚姻': '#婚姻问题类型#',
        '交通': '#事故类型#', '房产': '#房产问题类型#',
        '知识产权': '#知识产权类型#', '刑事': '#刑事罪名#',
        '行政': '#行政案件类型#', '合同': '#合同类型#',
    }

    # 槽位依赖关系（旧版）
    SLOT_DEPENDENCIES = {
        '#劳动问题类型#': ['#法律类型#'],
        '#用人单位#': ['#法律类型#'],
        '#工作时长#': ['#法律类型#', '#劳动问题类型#'],
        '#婚姻问题类型#': ['#法律类型#'],
        '#婚姻时长#': ['#法律类型#', '#婚姻问题类型#'],
        '#事故类型#': ['#法律类型#'],
        '#责任方#': ['#法律类型#', '#事故类型#'],
        '#房产问题类型#': ['#法律类型#'],
        '#房屋位置#': ['#法律类型#', '#房产问题类型#'],
        '#知识产权类型#': ['#法律类型#'],
        '#刑事罪名#': ['#法律类型#'],
        '#行政案件类型#': ['#法律类型#'],
        '#合同类型#': ['#法律类型#'],
        '#合同标的#': ['#法律类型#', '#合同类型#'],
    }

    # 法律类型与槽位映射（旧版）
    LEGAL_TYPE_SLOTS = {
        '劳动': ['#劳动问题类型#', '#工作时长#', '#用人单位#'],
        '婚姻': ['#婚姻问题类型#', '#婚姻时长#'],
        '交通': ['#事故类型#', '#责任方#'],
        '房产': ['#房产问题类型#', '#房屋位置#'],
        '知识产权': ['#知识产权类型#'],
        '刑事': ['#刑事罪名#'],
        '行政': ['#行政案件类型#'],
        '合同': ['#合同类型#', '#合同标的#'],
    }

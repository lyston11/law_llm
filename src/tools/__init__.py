"""Agent 工具注册表

定义所有可供 Agent 使用的工具（Function Calling Tools）。
每个工具包含：
  - definition: OpenAI 兼容的工具定义（JSON Schema）
  - implementation: 工具的实际执行函数
"""

from .knowledge_search import (
    TOOL_DEFINITION as KNOWLEDGE_SEARCH_DEF,
    execute as execute_knowledge_search,
)
from .article_lookup import (
    TOOL_DEFINITION as ARTICLE_LOOKUP_DEF,
    execute as execute_article_lookup,
)
from .knowledge_graph import (
    TOOL_DEFINITION as KNOWLEDGE_GRAPH_DEF,
    execute as execute_knowledge_graph,
)
from .situation_analyzer import (
    TOOL_DEFINITION as SITUATION_ANALYZER_DEF,
    execute as execute_situation_analyzer,
)

# 工具注册表：name -> (definition, implementation)
TOOL_REGISTRY = {
    "search_legal_knowledge": (KNOWLEDGE_SEARCH_DEF, execute_knowledge_search),
    "lookup_legal_article": (ARTICLE_LOOKUP_DEF, execute_article_lookup),
    "query_knowledge_graph": (KNOWLEDGE_GRAPH_DEF, execute_knowledge_graph),
    "analyze_legal_situation": (SITUATION_ANALYZER_DEF, execute_situation_analyzer),
}


def get_all_tool_definitions():
    """获取所有工具的 OpenAI 兼容定义列表"""
    return [defn for defn, _ in TOOL_REGISTRY.values()]


def get_tool_executor(tool_name):
    """根据工具名称获取对应的执行函数"""
    if tool_name in TOOL_REGISTRY:
        return TOOL_REGISTRY[tool_name][1]
    return None

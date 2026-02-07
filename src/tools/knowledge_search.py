"""工具：法律知识库搜索

基于 RAG 向量检索，从知识库中搜索与用户问题相关的法律资料。
底层复用 src/rag.py 的 RAGRetriever。
"""
import json

# ==================== 工具定义（OpenAI Function Calling 格式）====================

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_legal_knowledge",
        "description": (
            "从法律知识库中检索与用户问题相关的法律条文、案例和解释。"
            "当用户询问法律问题、需要法律依据、或者需要参考相关法律资料时，应该调用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询关键词，应该是用户问题的核心内容，例如 '劳动合同解除赔偿' 或 '交通事故责任认定'",
                },
                "domain": {
                    "type": "string",
                    "description": "法律领域过滤，可选值：劳动纠纷、婚姻家庭、交通事故、房产纠纷、刑事案件、合同纠纷、知识产权、行政纠纷",
                    "enum": [
                        "劳动纠纷", "婚姻家庭", "交通事故", "房产纠纷",
                        "刑事案件", "合同纠纷", "知识产权", "行政纠纷",
                    ],
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的结果数量，默认为 5",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ==================== 工具实现 ====================

# 全局 RAG 检索器引用（由 Agent 初始化时注入）
_rag_retriever = None


def set_rag_retriever(retriever):
    """注入 RAG 检索器实例"""
    global _rag_retriever
    _rag_retriever = retriever


def execute(query: str, domain: str = None, top_k: int = 5) -> str:
    """
    执行知识库搜索

    Args:
        query: 搜索关键词
        domain: 法律领域过滤
        top_k: 返回结果数量

    Returns:
        str: JSON 格式的搜索结果
    """
    if _rag_retriever is None or not _rag_retriever.is_ready:
        return json.dumps({
            "success": False,
            "error": "法律知识库未就绪",
            "results": [],
        }, ensure_ascii=False)

    try:
        results = _rag_retriever.retrieve(query=query, k=top_k, domain=domain)

        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc["content"][:500],
                "domain": doc["metadata"].get("domain", "未知"),
                "source": doc["metadata"].get("source", "未知"),
                "relevance_score": doc["score"],
            })

        return json.dumps({
            "success": True,
            "query": query,
            "result_count": len(formatted_results),
            "results": formatted_results,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"检索失败: {str(e)}",
            "results": [],
        }, ensure_ascii=False)

"""工具：法条查询

查询具体的法律条文内容，如"劳动法第二十条"、"民法典第1087条"等。
底层使用向量检索 + 关键词精确匹配。
"""
import json
import re

# 全局 RAG 检索器引用
_rag_retriever = None


# ==================== 工具定义 ====================

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "lookup_legal_article",
        "description": (
            "查询具体的法律条文内容。当用户提到某部法律的具体条款时使用，"
            "例如 '劳动法第二十条'、'民法典第1087条'、'刑法第234条' 等。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "law_name": {
                    "type": "string",
                    "description": "法律名称，例如 '劳动法'、'劳动合同法'、'民法典'、'刑法'、'婚姻法'",
                },
                "article_number": {
                    "type": "string",
                    "description": "条款编号，例如 '第二十条'、'第1087条'、'第234条'",
                },
            },
            "required": ["law_name", "article_number"],
        },
    },
}


# ==================== 工具实现 ====================

def set_rag_retriever(retriever):
    """注入 RAG 检索器实例"""
    global _rag_retriever
    _rag_retriever = retriever


def _normalize_article_number(article_number: str) -> list[str]:
    """
    将条款编号标准化为多种可能的格式，用于模糊匹配。
    例如 "第二十条" → ["第二十条", "第20条", "第二十条"]
    """
    variants = [article_number]

    # 中文数字 → 阿拉伯数字
    cn_num_map = {
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        '十': '10', '百': '100', '千': '1000',
    }

    # 尝试提取数字部分
    num_match = re.search(r'第?([零一二三四五六七八九十百千万\d]+)条?', article_number)
    if num_match:
        num_str = num_match.group(1)

        # 如果已经是阿拉伯数字
        if num_str.isdigit():
            variants.append(f"第{num_str}条")
            # 尝试转为中文（简单情况）
        else:
            # 简单中文数字转换
            arabic = _cn_to_arabic(num_str)
            if arabic:
                variants.append(f"第{arabic}条")
                variants.append(f"第{num_str}条")

    return list(set(variants))


def _cn_to_arabic(cn_str: str) -> str:
    """简单的中文数字转阿拉伯数字"""
    cn_map = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    }
    try:
        if '百' in cn_str:
            parts = cn_str.split('百')
            hundreds = cn_map.get(parts[0], 0) * 100
            remainder = parts[1] if len(parts) > 1 else ''
            if not remainder:
                return str(hundreds)
            if '十' in remainder:
                return str(hundreds + _parse_tens(remainder, cn_map))
            elif remainder in cn_map:
                return str(hundreds + cn_map[remainder])
            return str(hundreds)
        elif '十' in cn_str:
            return str(_parse_tens(cn_str, cn_map))
        elif cn_str in cn_map:
            return str(cn_map[cn_str])
        return None
    except Exception:
        return None


def _parse_tens(s: str, cn_map: dict) -> int:
    """解析十位数"""
    parts = s.split('十')
    if parts[0] == '':
        tens = 10
    else:
        tens = cn_map.get(parts[0], 0) * 10
    ones = cn_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
    return tens + ones


def execute(law_name: str, article_number: str) -> str:
    """
    执行法条查询

    Args:
        law_name: 法律名称
        article_number: 条款编号

    Returns:
        str: JSON 格式的查询结果
    """
    if _rag_retriever is None or not _rag_retriever.is_ready:
        return json.dumps({
            "success": False,
            "error": "法律知识库未就绪",
            "results": [],
        }, ensure_ascii=False)

    try:
        # 构建精确查询
        search_query = f"{law_name}{article_number}"

        # 使用 RAG 检索
        results = _rag_retriever.retrieve(query=search_query, k=8)

        # 获取条款编号的所有变体
        article_variants = _normalize_article_number(article_number)

        # 优先返回包含精确条款号的结果
        exact_matches = []
        fuzzy_matches = []

        for doc in results:
            content = doc["content"]
            is_exact = any(variant in content for variant in article_variants)

            entry = {
                "content": content[:600],
                "domain": doc["metadata"].get("domain", "未知"),
                "source": doc["metadata"].get("source", "未知"),
                "relevance_score": doc["score"],
                "exact_match": is_exact,
            }

            if is_exact:
                exact_matches.append(entry)
            else:
                fuzzy_matches.append(entry)

        # 精确匹配优先，不足时补充模糊匹配
        all_results = exact_matches + fuzzy_matches[:3]

        return json.dumps({
            "success": True,
            "law_name": law_name,
            "article_number": article_number,
            "exact_match_count": len(exact_matches),
            "total_results": len(all_results),
            "results": all_results,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"法条查询失败: {str(e)}",
            "results": [],
        }, ensure_ascii=False)

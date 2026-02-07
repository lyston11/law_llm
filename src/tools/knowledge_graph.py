"""工具：知识图谱查询

查询法律知识图谱中的实体关系，帮助理解法律概念之间的联系。
底层使用 NetworkX 加载 .graphml 格式的知识图谱。
"""
import json
import os

from config import Config

config = Config()

# 全局知识图谱
_graph = None
_graph_loaded = False


# ==================== 工具定义 ====================

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "query_knowledge_graph",
        "description": (
            "查询法律知识图谱，了解法律概念之间的关系。"
            "例如查询某个法律概念涉及哪些相关条文、哪些实体之间存在法律关系等。"
            "适用于需要理解法律概念的上下位关系、关联关系时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "要查询的法律实体或概念名称，例如 '劳动合同'、'工伤赔偿'、'离婚财产分割'",
                },
                "relation_type": {
                    "type": "string",
                    "description": "查询的关系类型",
                    "enum": ["related", "all_neighbors", "shortest_path"],
                    "default": "all_neighbors",
                },
                "target_entity": {
                    "type": "string",
                    "description": "目标实体（仅在 relation_type 为 shortest_path 时需要）",
                },
            },
            "required": ["entity"],
        },
    },
}


# ==================== 工具实现 ====================

def _ensure_graph_loaded():
    """确保知识图谱已加载"""
    global _graph, _graph_loaded

    if _graph_loaded:
        return _graph is not None

    _graph_loaded = True

    graph_path = os.path.join(config.KNOWLEDGE_GRAPH_DIR, "law_knowledge_graph.graphml")
    if not os.path.exists(graph_path):
        print(f"⚠️  知识图谱文件不存在: {graph_path}")
        return False

    try:
        import networkx as nx
        _graph = nx.read_graphml(graph_path)
        print(f"✅ 知识图谱加载完成: {_graph.number_of_nodes()} 个节点, {_graph.number_of_edges()} 条边")
        return True
    except Exception as e:
        print(f"⚠️  知识图谱加载失败: {e}")
        return False


def _find_matching_nodes(entity: str) -> list:
    """在图中查找匹配的节点（支持模糊匹配）"""
    if _graph is None:
        return []

    matches = []
    entity_lower = entity.lower()

    for node in _graph.nodes():
        node_str = str(node).lower()
        # 精确匹配
        if entity_lower == node_str:
            matches.insert(0, node)
        # 包含匹配
        elif entity_lower in node_str or node_str in entity_lower:
            matches.append(node)

    return matches[:10]  # 最多返回 10 个


def execute(entity: str, relation_type: str = "all_neighbors", target_entity: str = None) -> str:
    """
    执行知识图谱查询

    Args:
        entity: 要查询的实体名称
        relation_type: 查询类型
        target_entity: 目标实体（shortest_path 时使用）

    Returns:
        str: JSON 格式的查询结果
    """
    if not _ensure_graph_loaded():
        return json.dumps({
            "success": False,
            "error": "知识图谱未加载，暂时无法查询法律概念关系",
            "results": [],
        }, ensure_ascii=False)

    try:
        import networkx as nx

        # 查找匹配节点
        matching_nodes = _find_matching_nodes(entity)
        if not matching_nodes:
            return json.dumps({
                "success": True,
                "entity": entity,
                "message": f"未找到与 '{entity}' 相关的知识图谱节点",
                "results": [],
            }, ensure_ascii=False)

        primary_node = matching_nodes[0]
        results = []

        if relation_type == "all_neighbors":
            # 查询所有邻居节点
            neighbors = list(_graph.neighbors(primary_node))
            for neighbor in neighbors[:20]:
                edge_data = _graph.get_edge_data(primary_node, neighbor) or {}
                results.append({
                    "source": str(primary_node),
                    "target": str(neighbor),
                    "relation": edge_data.get("relation", edge_data.get("label", "相关")),
                })

        elif relation_type == "shortest_path":
            if not target_entity:
                return json.dumps({
                    "success": False,
                    "error": "shortest_path 查询需要提供 target_entity",
                    "results": [],
                }, ensure_ascii=False)

            target_nodes = _find_matching_nodes(target_entity)
            if not target_nodes:
                return json.dumps({
                    "success": True,
                    "message": f"未找到目标实体 '{target_entity}'",
                    "results": [],
                }, ensure_ascii=False)

            try:
                path = nx.shortest_path(_graph, primary_node, target_nodes[0])
                results.append({
                    "path": [str(n) for n in path],
                    "length": len(path) - 1,
                })
            except nx.NetworkXNoPath:
                results.append({
                    "message": f"'{entity}' 和 '{target_entity}' 之间没有直接路径",
                })

        else:  # "related"
            # 查询相关实体（2 跳以内）
            neighbors_1 = set(_graph.neighbors(primary_node))
            for n in neighbors_1:
                edge_data = _graph.get_edge_data(primary_node, n) or {}
                results.append({
                    "entity": str(n),
                    "relation": edge_data.get("relation", edge_data.get("label", "相关")),
                    "hops": 1,
                })

        # 获取节点属性
        node_attrs = dict(_graph.nodes[primary_node]) if primary_node in _graph.nodes else {}

        return json.dumps({
            "success": True,
            "entity": entity,
            "matched_node": str(primary_node),
            "node_attributes": {k: str(v) for k, v in node_attrs.items()},
            "relation_count": len(results),
            "results": results[:20],
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"知识图谱查询失败: {str(e)}",
            "results": [],
        }, ensure_ascii=False)

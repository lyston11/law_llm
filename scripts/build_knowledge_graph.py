import jsonlines
import os
import time
import networkx as nx
from pyvis.network import Network
from tqdm import tqdm
import argparse

import re

def extract_legal_entities(text):
    """从文本中提取法律实体"""
    # 简化的实体提取，基于关键词匹配
    entities = {
        "罪名": [],
        "法条": [],
        "法律概念": [],
        "法律程序": []
    }
    
    # 罪名关键词
    crime_keywords = ["罪", "罪名", "刑事责任", "有期徒刑", "无期徒刑", "死刑"]
    # 法条关键词
    law_keywords = ["法", "条例", "规定", "办法", "细则", "条款"]
    # 法律概念关键词
    concept_keywords = ["权利", "义务", "责任", "效力", "时效", "管辖"]
    # 法律程序关键词
    procedure_keywords = ["起诉", "上诉", "申诉", "调解", "仲裁", "执行"]
    
    # 提取罪名
    for keyword in crime_keywords:
        if keyword in text:
            # 简单提取包含关键词的短语
            matches = re.findall(r'[\u4e00-\u9fa5]{2,}'+keyword, text)
            entities["罪名"].extend(matches)
    
    # 提取法条
    for keyword in law_keywords:
        if keyword in text:
            matches = re.findall(r'[\u4e00-\u9fa5]+第[0-9]+条', text)
            entities["法条"].extend(matches)
    
    # 去重
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

def build_law_knowledge_graph(version="1"):
    """构建法律知识图谱"""
    start_time = time.time()
    print("=== 开始构建法律知识图谱 ===")
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 使用统一的项目目录结构
    knowledge_graph_dir = os.path.join(project_root, "data", "knowledge_graph")
    os.makedirs(knowledge_graph_dir, exist_ok=True)
    
    # 创建知识图谱
    datasets_dir = os.path.join(project_root, "data", "legal", "datasets")
    
    G = nx.DiGraph()
    
    # 添加实体类型属性（使用字符串类型，避免GraphML格式问题）
    G.graph["name"] = "法律知识图谱"
    G.graph["description"] = "基于DISC-Law-SFT数据集构建的法律知识图谱"
    G.graph["entity_types"] = "问题,法条,答案,罪名,法律概念,法律程序,法院,当事人,案由,判决结果"
    G.graph["relation_types"] = "解答,引用,涉及,适用,包含,定义,属于"
    G.graph["version"] = version
    
    # 处理DISC-Law-SFT-Triplet-QA数据集
    print("\n1. 正在处理DISC-Law-SFT-Triplet-QA-released.jsonl...")
    triplet_path = os.path.join(datasets_dir, "DISC-Law-SFT-Triplet-QA-released.jsonl")
    
    if os.path.exists(triplet_path):
        with jsonlines.open(triplet_path) as reader:
            for item in tqdm(reader, desc="处理Triplet数据"):
                # 提取问题作为实体
                question = item["input"].strip()
                if question:
                    G.add_node(question, type="问题")
                
                # 提取答案作为实体
                answer = item["output"].strip()
                if answer:
                    G.add_node(answer, type="答案")
                
                # 建立"解答"关系
                if question and answer:
                    G.add_edge(question, answer, type="解答")
                
                # 处理参考法条
                for ref in item["reference"]:
                    ref = ref.strip()
                    if ref:
                        G.add_node(ref, type="法条")
                        # 建立"引用"关系
                        if question:
                            G.add_edge(question, ref, type="引用")
                        # 建立"适用"关系
                        if answer:
                            G.add_edge(answer, ref, type="适用")
                
                # 从问题和答案中提取额外实体
                for text in [question, answer]:
                    if text:
                        entities = extract_legal_entities(text)
                        # 添加罪名实体
                        for crime in entities["罪名"]:
                            G.add_node(crime, type="罪名")
                            G.add_edge(text, crime, type="涉及")
                        # 添加法条实体
                        for law in entities["法条"]:
                            G.add_node(law, type="法条")
                            G.add_edge(text, law, type="涉及")
    else:
        print(f"✗ 未找到文件: {triplet_path}")
    
    # 处理DISC-Law-SFT-Pair-QA数据集
    print("\n2. 正在处理DISC-Law-SFT-Pair-QA-released.jsonl...")
    pair_path = os.path.join(datasets_dir, "DISC-Law-SFT-Pair-QA-released.jsonl")
    
    if os.path.exists(pair_path):
        with jsonlines.open(pair_path) as reader:
            for item in tqdm(reader, desc="处理Pair数据"):
                # 提取问题作为实体
                question = item["input"].strip()
                if question:
                    G.add_node(question, type="问题")
                
                # 提取答案作为实体
                answer = item["output"].strip()
                if answer:
                    G.add_node(answer, type="答案")
                
                # 建立"解答"关系
                if question and answer:
                    G.add_edge(question, answer, type="解答")
                
                # 从问题和答案中提取额外实体
                for text in [question, answer]:
                    if text:
                        entities = extract_legal_entities(text)
                        # 添加罪名实体
                        for crime in entities["罪名"]:
                            G.add_node(crime, type="罪名")
                            G.add_edge(text, crime, type="涉及")
                        # 添加法条实体
                        for law in entities["法条"]:
                            G.add_node(law, type="法条")
                            G.add_edge(text, law, type="涉及")
    else:
        print(f"✗ 未找到文件: {pair_path}")
    
    # 统计知识图谱信息
    print("\n3. 知识图谱统计信息:")
    print(f"总节点数: {G.number_of_nodes()}")
    print(f"总边数: {G.number_of_edges()}")
    
    # 按类型统计节点
    node_types = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "未知")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\n节点类型分布:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")
    
    # 按类型统计边
    edge_types = {}
    for u, v, attrs in G.edges(data=True):
        edge_type = attrs.get("type", "未知")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("\n边类型分布:")
    for edge_type, count in edge_types.items():
        print(f"  {edge_type}: {count}")
    
    # 保存知识图谱
    print("\n4. 正在保存知识图谱...")
    
    # 保存为GraphML格式，便于后续分析和可视化
    graphml_path = os.path.join(knowledge_graph_dir, "law_knowledge_graph.graphml")
    nx.write_graphml(G, graphml_path)
    print(f"✓ GraphML格式保存成功: {graphml_path}")
    
    # 保存为GEXF格式，适合Gephi等工具可视化
    gexf_path = os.path.join(knowledge_graph_dir, "law_knowledge_graph.gexf")
    nx.write_gexf(G, gexf_path)
    print(f"✓ GEXF格式保存成功: {gexf_path}")
    
    # 创建交互式HTML可视化
    print("\n5. 正在生成交互式HTML可视化...")
    
    # 限制节点数量，避免可视化过于复杂
    max_nodes = 1000
    if G.number_of_nodes() > max_nodes:
        print(f"注意: 知识图谱节点数过多 ({G.number_of_nodes()}), 仅可视化前 {max_nodes} 个节点")
        # 采用中心性算法选择最重要的节点
        degree_centrality = nx.degree_centrality(G)
        sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
        # 创建子图
        G_visual = G.subgraph(top_nodes)
    else:
        G_visual = G
    
    # 创建Pyvis网络
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.set_options("""{
      "nodes": {
        "font": {"size": 12},
        "shape": "ellipse",
        "scaling": {
          "min": 10,
          "max": 30
        }
      },
      "edges": {
        "color": {"inherit": true},
        "smooth": {"type": "continuous"}
      },
      "layout": {
        "hierarchical": {
          "enabled": false,
          "sortMethod": "directed"
        }
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -80000,
          "springConstant": 0.001,
          "springLength": 200
        }
      }
    }""")
    
    # 为不同类型的节点设置不同颜色
    color_map = {
        "问题": "#FF6B6B",  # 红色
        "法条": "#4ECDC4",  # 青色
        "答案": "#45B7D1",  # 蓝色
        "罪名": "#FFA07A",  # 浅橙色
        "法律概念": "#98D8C8",  # 浅绿色
        "法律程序": "#F7DC6F",  # 黄色
        "法院": "#BB8FCE",  # 紫色
        "当事人": "#85C1E2",  # 浅蓝色
        "案由": "#F8C471",  # 橙色
        "判决结果": "#82E0AA",  # 绿色
        "未知": "#999999"   # 灰色
    }
    
    # 为不同类型的边设置不同颜色
    edge_color_map = {
        "解答": "#3498DB",  # 蓝色
        "引用": "#E74C3C",  # 红色
        "涉及": "#2ECC71",  # 绿色
        "适用": "#9B59B6",  # 紫色
        "包含": "#F39C12",  # 橙色
        "定义": "#1ABC9C",  # 青色
        "属于": "#34495E",  # 深灰色
        "未知": "#999999"   # 灰色
    }
    
    # 添加节点和边到Pyvis网络
    for node, attrs in G_visual.nodes(data=True):
        node_type = attrs.get("type", "未知")
        color = color_map.get(node_type, color_map["未知"])
        net.add_node(
            node, 
            label=node[:50] + "..." if len(node) > 50 else node,  # 限制标签长度
            title=f"{node_type}: {node}",  # 悬停显示完整内容
            color=color,
            group=node_type,
            size=15 + len(node) / 10  # 根据节点长度调整大小
        )
    
    for u, v, attrs in G_visual.edges(data=True):
        edge_type = attrs.get("type", "未知")
        edge_color = edge_color_map.get(edge_type, edge_color_map["未知"])
        net.add_edge(u, v, title=edge_type, label=edge_type[:2], color=edge_color, width=2)
    
    # 保存HTML可视化文件
    html_path = os.path.join(knowledge_graph_dir, "law_knowledge_graph.html")
    net.save_graph(html_path)
    print(f"✓ 交互式HTML可视化生成成功: {html_path}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== 法律知识图谱构建完成 ===")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    print(f"版本: {version}")
    print(f"知识图谱文件已保存至 {knowledge_graph_dir} 目录")
    
    return G

def main():
    parser = argparse.ArgumentParser(description="构建法律知识图谱")
    parser.add_argument("--version", type=str, default="1", help="版本号，默认为1")
    args = parser.parse_args()
    
    build_law_knowledge_graph(args.version)

if __name__ == "__main__":
    main()
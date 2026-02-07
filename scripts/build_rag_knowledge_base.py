import jsonlines
import os
import time
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import torch
import argparse

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

def classify_domain(text):
    """根据文本内容分类到不同法律领域"""
    # 定义领域关键词
    domain_keywords = {
        "劳动纠纷": ["工资", "社保", "加班", "辞退", "劳动合同", "劳动者", "用人单位", "工伤"],
        "婚姻家庭": ["离婚", "结婚", "家暴", "抚养权", "财产分割", "配偶", "孩子", "家庭"],
        "交通事故": ["车祸", "追尾", "责任", "赔偿", "撞车", "闯红灯", "交通事故", "肇事者"],
        "房产问题": ["买房", "租房", "拆迁", "装修", "房产", "房屋", "物业", "业主"],
        "知识产权": ["专利", "商标", "著作权", "侵权", "抄袭", "知识产权", "版权", "原创"],
        "刑事案件": ["盗窃", "诈骗", "抢劫", "故意伤害", "犯罪", "刑事", "坐牢", "判刑"],
        "行政案件": ["行政处罚", "行政复议", "行政诉讼", "政府", "行政机关", "执法"],
        "合同纠纷": ["合同", "违约", "履行", "纠纷", "协议", "签订", "解除合同"]
    }
    
    # 统计每个领域的关键词出现次数
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            domain_scores[domain] = score
    
    # 返回得分最高的领域
    if domain_scores:
        return max(domain_scores, key=domain_scores.get)
    else:
        return "其他法律问题"

def classify_question_type(text):
    """根据文本内容分类问题类型"""
    question_types = {
        "处罚标准": ["判刑", "处罚", "量刑", "罚金", "刑期", "怎么判", "判多久"],
        "法律概念": ["是什么", "概念", "定义", "含义", "意思"],
        "权利义务": ["义务", "权利", "可以", "应该", "必须", "不能"],
        "程序流程": ["如何", "怎么", "流程", "步骤", "申请", "办理"],
        "赔偿标准": ["赔偿", "补偿", "损失", "费用", "多少钱", "经济补偿"],
        "认定标准": ["认定", "条件", "标准", "符合", "满足", "构成"],
        "纠纷解决": ["纠纷", "解决", "处理", "怎么办", "如何解决", "调解"]
    }
    
    # 统计每个问题类型的关键词出现次数
    type_scores = {}
    for qtype, keywords in question_types.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            type_scores[qtype] = score
    
    # 返回得分最高的问题类型
    if type_scores:
        return max(type_scores, key=type_scores.get)
    else:
        return "其他问题类型"

def process_law_datasets(version="1"):
    """处理法律数据集，构建RAG检索向量知识库"""
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 检查GPU是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")
    
    # 加载预训练的中文embedding模型
    print("正在加载中文embedding模型...")
    os.environ["TRANSFORMERS_SAFE_SERIALIZATION"] = "true"
    
    model_path = os.path.join(project_root, "models", "bge-small-zh-v1.5")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 128},
    )
    
    # 使用统一的项目目录结构
    vector_db_dir = os.path.join(project_root, "data", "knowledge_base", "vector_db")
    datasets_dir = os.path.join(project_root, "data", "legal", "datasets")
    os.makedirs(vector_db_dir, exist_ok=True)
    
    all_docs = []
    doc_metadata = []
    
    # 处理DISC-Law-SFT-Pair-QA数据集
    print("\n正在处理DISC-Law-SFT-Pair-QA-released.jsonl...")
    pair_start_time = time.time()
    with jsonlines.open(os.path.join(datasets_dir, "DISC-Law-SFT-Pair-QA-released.jsonl")) as reader:
        for item in tqdm(reader, desc="Pair数据集"):
            content = f"问题: {item['input']}\n回答: {item['output']}"
            all_docs.append(content)
            
            # 收集原始元数据
            doc_metadata.append({
                "dataset": "Pair-QA",
                "id": item.get("id", f"pair_{len(all_docs)-1}"),
                "input": item["input"],
                "output": item["output"]
            })
    pair_end_time = time.time()
    print(f"Pair数据集处理时间: {pair_end_time - pair_start_time:.2f}秒")
    
    # 处理DISC-Law-SFT-Triplet-QA数据集
    print("\n正在处理DISC-Law-SFT-Triplet-QA-released.jsonl...")
    triplet_start_time = time.time()
    with jsonlines.open(os.path.join(datasets_dir, "DISC-Law-SFT-Triplet-QA-released.jsonl")) as reader:
        for item in tqdm(reader, desc="Triplet数据集"):
            content = f"参考法条: {'; '.join(item['reference'])}\n问题: {item['input']}\n回答: {item['output']}"
            all_docs.append(content)
            
            # 收集原始元数据
            doc_metadata.append({
                "dataset": "Triplet-QA",
                "id": item.get("id", f"triplet_{len(all_docs)-1}"),
                "reference": item["reference"],
                "input": item["input"],
                "output": item["output"]
            })
    triplet_end_time = time.time()
    print(f"Triplet数据集处理时间: {triplet_end_time - triplet_start_time:.2f}秒")
    
    print(f"\n总文档数: {len(all_docs)}")
    
    # 文本分割优化：使用更细粒度的分割策略
    print("\n正在进行文本分割...")
    split_start_time = time.time()
    
    # 优化：使用更小的chunk_size和适当的overlap，提高检索精度
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # 减小chunk_size，提高检索精度
        chunk_overlap=100,  # 调整overlap，保持上下文连贯性
        length_function=len,
        separators=["\n", "。", "！", "？", ";", "，", "、"]  # 增加更多分隔符
    )
    
    # 将文本转换为LangChain文档对象
    langchain_docs = []
    for i, (content, metadata) in enumerate(tqdm(zip(all_docs, doc_metadata), desc="文本分割", total=len(all_docs))):
        chunks = text_splitter.split_text(content)
        
        # 为每个chunk添加丰富的元数据
        for j, chunk in enumerate(chunks):
            # 分类领域和问题类型
            domain = classify_domain(content)
            question_type = classify_question_type(metadata["input"])
            
            # 提取法律实体
            entities = extract_legal_entities(content)
            
            # 构建GraphRAG元数据（确保所有值都是基本类型）
            graph_rag_metadata = {
                "source": f"law_doc_{i}_chunk_{j}",
                "version": version,
                "dataset": metadata["dataset"],
                "doc_id": metadata["id"],
                "domain": domain,  # 领域标签
                "question_type": question_type,  # 问题类型标签
                "original_input": metadata["input"][:100],  # 原始问题摘要
                "crime_count": len(entities["罪名"]),  # GraphRAG: 罪名数量
                "law_count": len(entities["法条"]),  # GraphRAG: 法条数量
                "concept_count": len(entities["法律概念"]),  # GraphRAG: 法律概念数量
                "procedure_count": len(entities["法律程序"]),  # GraphRAG: 法律程序数量
                "has_crimes": len(entities["罪名"]) > 0,  # GraphRAG: 是否包含罪名
                "has_laws": len(entities["法条"]) > 0,  # GraphRAG: 是否包含法条
                "has_concepts": len(entities["法律概念"]) > 0,  # GraphRAG: 是否包含法律概念
                "has_procedures": len(entities["法律程序"]) > 0  # GraphRAG: 是否包含法律程序
            }
            
            # 只添加非空的实体列表（转换为字符串）
            if entities["罪名"]:
                graph_rag_metadata["crimes"] = ", ".join(entities["罪名"])[:200]  # 限制长度
            if entities["法条"]:
                graph_rag_metadata["laws"] = ", ".join(entities["法条"])[:200]  # 限制长度
            if entities["法律概念"]:
                graph_rag_metadata["concepts"] = ", ".join(entities["法律概念"])[:200]  # 限制长度
            if entities["法律程序"]:
                graph_rag_metadata["procedures"] = ", ".join(entities["法律程序"])[:200]  # 限制长度
            
            langchain_docs.append({
                "page_content": chunk,
                "metadata": graph_rag_metadata
            })
    
    split_end_time = time.time()
    print(f"文本分割时间: {split_end_time - split_start_time:.2f}秒")
    
    print(f"\n分割后总chunk数: {len(langchain_docs)}")
    
    # 提取page_content和metadata
    documents = [doc["page_content"] for doc in langchain_docs]
    metadatas = [doc["metadata"] for doc in langchain_docs]
    
    # 构建向量数据库
    print("\n正在构建向量数据库...")
    db_start_time = time.time()
    
    # 优化：增大batch_size，提高构建效率
    batch_size = 2000  # 增大batch_size，提高处理速度
    total_batches = len(documents) // batch_size + 1
    
    # 初始化向量数据库
    vector_db = Chroma.from_texts(
        texts=documents[:batch_size],
        embedding=embeddings,
        metadatas=metadatas[:batch_size],
        persist_directory=vector_db_dir
    )
    
    # 批量添加剩余文档
    for batch_idx in tqdm(range(1, total_batches), desc="构建向量库"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(documents))
        
        batch_docs = documents[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]
        
        vector_db.add_texts(
            texts=batch_docs,
            metadatas=batch_metadatas
        )
    
    db_end_time = time.time()
    print(f"向量数据库构建时间: {db_end_time - db_start_time:.2f}秒")
    
    print(f"\n向量数据库构建完成！")
    print(f"向量数量: {vector_db._collection.count()}")
    print(f"版本: {version}")
    print(f"总耗时: {db_end_time - pair_start_time:.2f}秒")
    
    # 优化：增加更多测试用例，覆盖所有领域
    print("\n正在测试检索功能...")
    test_queries = [
        "诈骗罪如何判刑",
        "被管制罪犯的义务是什么",
        "劳动者解除劳动合同需要提前多少天通知用人单位",
        "离婚时三岁孩子的抚养权归谁",
        "交通事故对方全责怎么赔偿",
        "专利侵权怎么处理",
        "租房被坑了怎么办",
        "合同违约怎么赔偿"
    ]
    
    for query in test_queries:
        results = vector_db.similarity_search(query, k=3)  # 增加检索结果数量
        print(f"\n测试查询: {query}")
        print(f"检索结果 ({len(results)} 条):")
        for i, result in enumerate(results):
            print(f"结果 {i+1}:")
            print(f"  内容: {result.page_content[:150]}...")
            print(f"  领域: {result.metadata.get('domain', '未知')}")
            print(f"  问题类型: {result.metadata.get('question_type', '未知')}")
    
    return vector_db

def main():
    parser = argparse.ArgumentParser(description="构建RAG检索向量知识库")
    parser.add_argument("--version", type=str, default="1", help="版本号，默认为1")
    args = parser.parse_args()
    
    vector_db = process_law_datasets(args.version)
    print("\n法律RAG检索向量知识库构建完成！")
    print(f"向量数据库已保存至 data/knowledge_base/vector_db/ 目录")

if __name__ == "__main__":
    main()
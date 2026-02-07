#!/usr/bin/env python3
"""
意图分类模型微调脚本
使用BGE模型生成句子嵌入，然后训练一个简单的分类器
"""

import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import joblib
import os
import logging
from config import Config

# 创建配置实例
config = Config()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型配置
MODEL_NAME = "BAAI/bge-large-zh-v1.5"  # 2025年最强中文嵌入模型
RANDOM_SEED = 42
OUTPUT_DIR = "models/intent_classifier"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_annotation_data(file_path):
    """
    加载标注数据
    
    Args:
        file_path (str): 标注数据文件路径
        
    Returns:
        list: 标注数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_embeddings(model, texts):
    """
    使用BGE模型生成句子嵌入
    
    Args:
        model (SentenceTransformer): BGE模型
        texts (list): 文本列表
        
    Returns:
        np.array: 句子嵌入数组
    """
    logger.info(f"生成 {len(texts)} 条文本的嵌入...")
    embeddings = model.encode(texts, convert_to_tensor=False, batch_size=32, show_progress_bar=True)
    return embeddings

def train_classifier(X_train, y_train):
    """
    训练意图分类器
    
    Args:
        X_train (np.array): 训练数据嵌入
        y_train (list): 训练数据标签
        
    Returns:
        XGBClassifier: 训练好的分类器
    """
    logger.info("训练意图分类器...")
    classifier = XGBClassifier(
        random_state=RANDOM_SEED,
        n_estimators=1000,  # 增加树的数量
        max_depth=8,         # 增加树的深度
        learning_rate=0.05,  # 降低学习率，配合更多树
        subsample=0.9,       # 增加子采样比例
        colsample_bytree=0.9, # 增加列采样比例
        reg_lambda=0.1,      # 增加L2正则化，防止过拟合
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_classifier(classifier, X_test, y_test, label_map):
    """
    评估分类器性能
    
    Args:
        classifier: 训练好的分类器
        X_test (np.array): 测试数据嵌入
        y_test (list): 测试数据标签
        label_map (dict): 标签映射
        
    Returns:
        dict: 评估结果
    """
    logger.info("评估分类器性能...")
    
    # 预测
    y_pred = classifier.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 生成分类报告
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=list(label_map.keys()),
        zero_division=0
    )
    
    logger.info(f"准确率: {accuracy:.4f}")
    logger.info(f"分类报告:\n{report}")
    
    return {
        "accuracy": accuracy,
        "classification_report": report
    }

def save_model(model, classifier, label_map, output_dir):
    """
    保存模型和分类器
    
    Args:
        model (SentenceTransformer): BGE模型
        classifier: 训练好的分类器
        label_map (dict): 标签映射
        output_dir (str): 输出目录
    """
    logger.info(f"保存模型到: {output_dir}")
    
    # 保存分类器
    classifier_path = os.path.join(output_dir, "intent_classifier.joblib")
    joblib.dump(classifier, classifier_path)
    
    # 保存标签映射
    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    logger.info("模型保存完成！")

def main():
    """
    主函数，执行微调流程
    """
    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 1. 加载标注数据
    logger.info("加载合并标注数据...")
    annotation_data = load_annotation_data(os.path.join(config.ANNOTATION_DIR, "intent_annotation_merged.json"))
    
    # 2. 准备数据
    logger.info("准备训练数据...")
    texts = [item['user_input'] for item in annotation_data]
    labels = [item['intent_label'] for item in annotation_data]
    
    # 创建标签映射，将场景前缀添加到标签中
    # 场景节点ID格式：scenario-法律问答_node1
    unique_labels = list(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # 转换标签为数字
    numeric_labels = [label_map[label] for label in labels]
    
    # 3. 加载BGE模型
    logger.info(f"加载预训练模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. 生成句子嵌入
    embeddings = generate_embeddings(model, texts)
    
    # 5. 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, 
        numeric_labels, 
        test_size=0.2, 
        random_state=RANDOM_SEED,
        stratify=numeric_labels
    )
    
    logger.info(f"训练集大小: {len(X_train)}")
    logger.info(f"测试集大小: {len(X_test)}")
    logger.info(f"标签数量: {len(unique_labels)}")
    
    # 6. 训练分类器
    classifier = train_classifier(X_train, y_train)
    
    # 7. 评估分类器
    evaluate_classifier(classifier, X_test, y_test, label_map)
    
    # 8. 保存模型
    save_model(model, classifier, label_map, OUTPUT_DIR)
    
    logger.info("意图分类器微调流程完成！")

if __name__ == "__main__":
    main()

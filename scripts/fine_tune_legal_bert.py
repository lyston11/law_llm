#!/usr/bin/env python3
"""
法律BERT模型微调脚本
在标注的法律意图数据上微调预训练的法律BERT模型
"""

import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import logging
import os
from config import Config

# 创建配置实例
config = Config()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型配置
MODEL_NAME = "BAAI/bge-large-zh-v1.5"  # 2025年最强中文嵌入模型
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
RANDOM_SEED = 42

# 输出目录
OUTPUT_DIR = "models/legal-bert-finetuned"
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

def prepare_data(annotation_data):
    """
    准备训练数据和测试数据
    
    Args:
        annotation_data (list): 标注数据列表
        
    Returns:
        tuple: (train_examples, test_examples, label_map)
    """
    # 提取所有标签
    labels = [item['intent_label'] for item in annotation_data]
    unique_labels = list(set(labels))
    
    # 创建标签映射
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # 转换为InputExample格式
    examples = []
    for item in annotation_data:
        examples.append(InputExample(
            texts=[item['user_input']],
            label=label_map[item['intent_label']]
        ))
    
    # 分割训练集和测试集
    train_examples, test_examples = train_test_split(
        examples, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=[ex.label for ex in examples]
    )
    
    logger.info(f"训练集大小: {len(train_examples)}")
    logger.info(f"测试集大小: {len(test_examples)}")
    logger.info(f"标签数量: {len(unique_labels)}")
    
    return train_examples, test_examples, label_map

def fine_tune_model(train_examples, model_name, output_dir):
    """
    微调BERT模型
    
    Args:
        train_examples (list): 训练数据
        model_name (str): 预训练模型名称
        output_dir (str): 微调后模型保存目录
        
    Returns:
        SentenceTransformer: 微调后的模型
    """
    # 加载预训练模型
    logger.info(f"加载预训练模型: {model_name}")
    model = SentenceTransformer(model_name)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=BATCH_SIZE
    )
    
    # 定义损失函数
    # 对于分类任务，使用SoftmaxLoss更合适
    from sentence_transformers.losses import SoftmaxLoss
    
    # 获取标签数量
    num_labels = len(set(item['intent_label'] for item in load_annotation_data(os.path.join(config.ANNOTATION_DIR, "intent_annotation.json"))))
    
    # 创建SoftmaxLoss
    train_loss = SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=num_labels
    )
    
    # 微调模型
    logger.info("开始微调模型...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=100,
        show_progress_bar=True,
        output_path=output_dir
    )
    
    logger.info(f"模型微调完成，已保存到: {output_dir}")
    return model

def evaluate_model(model, test_examples, label_map):
    """
    评估模型性能
    
    Args:
        model (SentenceTransformer): 微调后的模型
        test_examples (list): 测试数据
        label_map (dict): 标签映射
        
    Returns:
        dict: 评估结果
    """
    logger.info("开始评估模型...")
    
    # 提取文本和真实标签
    texts = [ex.texts[0] for ex in test_examples]
    true_labels = [ex.label for ex in test_examples]
    
    # 计算文本嵌入
    embeddings = model.encode(texts)
    
    # 计算相似度矩阵
    similarity_matrix = embeddings @ embeddings.T
    
    # 评估性能（这里使用简单的最近邻分类作为示例）
    # 实际应用中，应该使用更复杂的分类器
    predicted_labels = []
    for i in range(len(similarity_matrix)):
        # 找到最相似的训练样本
        similarity_scores = similarity_matrix[i]
        # 排除自身
        similarity_scores[i] = -float('inf')
        # 找到最大相似度的索引
        max_idx = np.argmax(similarity_scores)
        predicted_labels.append(true_labels[max_idx])
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=list(label_map.keys())
    )
    
    logger.info(f"模型准确率: {accuracy:.4f}")
    logger.info(f"分类报告:\n{report}")
    
    return {
        "accuracy": accuracy,
        "classification_report": report
    }

def main():
    """
    主函数，执行模型微调流程
    """
    # 设置随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # 1. 加载标注数据
    logger.info("加载标注数据...")
    annotation_data = load_annotation_data(os.path.join(config.ANNOTATION_DIR, "intent_annotation.json"))
    
    # 2. 准备数据
    logger.info("准备训练数据...")
    train_examples, test_examples, label_map = prepare_data(annotation_data)
    
    # 3. 微调模型
    logger.info("微调法律BERT模型...")
    fine_tuned_model = fine_tune_model(train_examples, MODEL_NAME, OUTPUT_DIR)
    
    # 4. 评估模型
    logger.info("评估微调后的模型...")
    evaluate_model(fine_tuned_model, test_examples, label_map)
    
    logger.info("模型微调流程完成！")

if __name__ == "__main__":
    main()

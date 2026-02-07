#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试实体识别模块的优化效果
确保非法律问题被识别为"其他类型"，而不是法律诉讼
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.legacy.intent import IntentRecognizer
from src.legacy.intent_bert import BERTIntentRecognizer
from src.legacy.tfidf import TFIDFCalculator
from config import Config

config = Config()

def test_non_legal_questions():
    """
    测试非法律问题是否被正确分类为"其他类型"
    """
    print("=== 测试非法律问题分类 ===")
    
    # 创建简单的节点信息
    node_id2node_info = {
        'node1': {
            'intent': ['法律咨询', '法律问题'],
            'action': ['LEGAL_CONSULTATION']
        },
        'node2': {
            'intent': ['劳动纠纷', '劳动合同', '工资拖欠'],
            'action': ['LABOR_CONSULTATION']
        },
        'node3': {
            'intent': ['婚姻家庭', '离婚', '财产分割'],
            'action': ['FAMILY_CONSULTATION']
        }
    }
    
    # 测试问题列表
    test_questions = [
        # 法律问题（应该被正确分类）
        "我被公司拖欠工资了怎么办？",
        "离婚时财产如何分割？",
        "交通事故责任如何认定？",
        
        # 非法律问题（应该被分类为"其他类型"）
        "今天天气怎么样？",
        "如何学习Python编程？",
        "明天有什么电影上映？",
        "中国的首都是哪里？"
    ]
    
    # 测试TF-IDF意图识别器
    print("\n--- 测试TF-IDF意图识别器 ---")
    tfidf_calculator = TFIDFCalculator()
    intent_recognizer = IntentRecognizer(tfidf_calculator, node_id2node_info)
    
    # 初始化TF-IDF语料库
    for node in node_id2node_info.values():
        for intent in node.get('intent', []):
            tfidf_calculator.add_document(intent)
    tfidf_calculator.calculate_idf()
    
    for question in test_questions:
        memory = {
            'user_input': question,
            'avaiable_nodes': list(node_id2node_info.keys()),
            'hit_intent': None,
            'hit_intent_score': 0
        }
        result = intent_recognizer.intent_recognize(memory)
        print(f"问题: {question}")
        print(f"  识别结果: {result['legal_field']}")
        print(f"  意图分数: {result['hit_intent_score']}")
    
    # 测试BERT意图识别器
    print("\n--- 测试BERT意图识别器 ---")
    try:
        bert_intent_recognizer = BERTIntentRecognizer(node_id2node_info)
        
        for question in test_questions:
            memory = {
                'user_input': question,
                'avaiable_nodes': list(node_id2node_info.keys()),
                'hit_intent': None,
                'hit_intent_score': 0
            }
            result = bert_intent_recognizer.intent_recognize(memory)
            print(f"问题: {question}")
            print(f"  识别结果: {result['legal_field']}")
            print(f"  意图分数: {result['hit_intent_score']}")
    except Exception as e:
        print(f"BERT模型加载失败，跳过BERT意图识别器测试: {e}")

if __name__ == "__main__":
    test_non_legal_questions()

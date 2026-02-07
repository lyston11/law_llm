#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试法律领域识别效果
确保用户进入法律咨询场景时，法律问题被识别为具体的法律领域
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

def load_scene_data():
    """
    加载真实的场景数据，模拟dialog.py中的load_scenario方法
    """
    import json
    scene_file = os.path.join(config.SCENARIO_DIR, 'scenario-法律问答.json')
    with open(scene_file, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)
    
    # 构建节点ID到节点信息的映射，添加场景前缀
    node_id2node_info = {}
    scenario_name = os.path.basename(scene_file).split('.')[0]
    
    for node in scene_data:
        original_node_id = node['id']
        # 添加场景前缀，与dialog.py中的load_scenario方法保持一致
        node_id = f"{scenario_name}_{original_node_id}"
        # 创建节点副本，避免修改原始数据
        node_copy = node.copy()
        # 构建带有场景前缀的子节点ID
        if 'childnode' in node_copy:
            new_child = []
            for child in node_copy['childnode']:
                child_node_id = f"{scenario_name}_{child}"
                new_child.append(child_node_id)
            node_copy['childnode'] = new_child
        # 保存到映射中
        node_id2node_info[node_id] = node_copy
        # 同时保存原始节点ID的映射，便于测试
        node_id2node_info[original_node_id] = node_copy
    
    return node_id2node_info

def test_legal_field_recognition():
    """
    测试法律问题是否被正确识别为具体的法律领域
    """
    print("=== 测试法律领域识别 ===")
    
    # 加载真实的节点信息
    node_id2node_info = load_scene_data()
    
    # 测试问题列表 - 更针对性的法律问题
    test_questions = [
        # 劳动纠纷相关
        "我被公司拖欠工资了怎么办？",
        "公司无故解除劳动合同，我能获得赔偿吗？",
        "试用期内被辞退，有经济补偿吗？",
        
        # 婚姻家庭相关
        "离婚时财产如何分割？",
        "子女抚养权如何判定？",
        "婚前财产在离婚时会被分割吗？",
        
        # 交通事故相关
        "交通事故责任如何认定？",
        "车祸后对方全责，我能获得哪些赔偿？",
        "交通事故致人死亡，如何赔偿？",
        
        # 房产纠纷相关
        "房屋买卖合同无效怎么办？",
        "租房押金不退，如何维权？",
        "房产证加名需要什么手续？",
        
        # 合同纠纷相关
        "对方违约，我能解除合同吗？",
        "合同没有签字，有效吗？",
        "违约责任如何承担？",
        
        # 非法律问题
        "今天天气怎么样？",
        "如何学习Python编程？",
        "明天有什么电影上映？"
    ]
    
    # 测试BERT意图识别器（默认使用）
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
            print(f"  节点ID: {result['hit_intent']}")
            print(f"  识别结果: {result['legal_field']}")
            print(f"  是否法律问题: {result.get('is_legal_question', False)}")
            print(f"  意图分数: {result['hit_intent_score']}")
            print(f"  置信度: {result['confidence_level']}")
    except Exception as e:
        print(f"BERT模型加载失败，跳过BERT意图识别器测试: {e}")

if __name__ == "__main__":
    test_legal_field_recognition()

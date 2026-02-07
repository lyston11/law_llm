#!/usr/bin/env python3
"""
意图识别性能对比测试脚本
比较传统TF-IDF意图识别和基于BERT的意图识别的准确性和响应时间
"""

import time
import json
from src.legacy.dialog import DialogManager

# 测试用例
TEST_CASES = [
    "我有法律问题",
    "我想咨询劳动合同问题",
    "公司拖欠工资怎么办",
    "我被公司辞退了",
    "离婚财产怎么分割",
    "交通事故赔偿标准",
    "房产继承纠纷",
    "专利申请流程",
    "合同纠纷怎么解决",
    "行政复议程序"
]

def _test_intent_recognition(use_bert, test_cases):
    """
    测试意图识别性能
    
    Args:
        use_bert (bool): 是否使用BERT意图识别
        test_cases (list): 测试用例列表
        
    Returns:
        tuple: (平均响应时间, 识别结果)
    """
    # 初始化DialogManager
    dialog_manager = DialogManager(use_bert_intent=use_bert)
    
    # 准备对话记忆
    memory = {
        'avaiable_nodes': list(dialog_manager.node_id2node_info.keys()),
        'user_input': "",
        'entities': {},
        'slot_filled': {},
        'state': {},
        'hit_intent': None,
        'hit_intent_score': 0
    }
    
    results = []
    total_time = 0
    
    for test_input in test_cases:
        memory['user_input'] = test_input
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行意图识别
        result = dialog_manager.intent_recognizer.intent_recognize(memory.copy())
        
        # 记录结束时间
        end_time = time.time()
        
        # 计算响应时间
        response_time = end_time - start_time
        total_time += response_time
        
        results.append({
            'input': test_input,
            'intent': result['hit_intent'],
            'score': result['hit_intent_score'],
            'response_time': response_time
        })
    
    # 计算平均响应时间
    avg_time = total_time / len(test_cases)
    
    return avg_time, results


def test_intent_recognition_tfidf():
    """
    测试TF-IDF意图识别
    """
    avg_time, results = _test_intent_recognition(use_bert=False, test_cases=TEST_CASES)
    assert avg_time < 1.0  # 确保响应时间合理
    assert len(results) == len(TEST_CASES)
    
    # 检查每个结果都有意图识别结果
    for result in results:
        assert 'intent' in result
        assert 'score' in result


def test_intent_recognition_bert():
    """
    测试BERT意图识别
    """
    avg_time, results = _test_intent_recognition(use_bert=True, test_cases=TEST_CASES)
    assert avg_time < 2.0  # BERT模型可能慢一些，但应在合理范围内
    assert len(results) == len(TEST_CASES)
    
    # 检查每个结果都有意图识别结果
    for result in results:
        assert 'intent' in result
        assert 'score' in result

def main():
    """
    主函数，执行性能对比测试
    """
    print("意图识别性能对比测试")
    print("=" * 50)
    
    # 测试传统TF-IDF意图识别
    print("\n1. 测试传统TF-IDF意图识别:")
    avg_time_tfidf, results_tfidf = _test_intent_recognition(use_bert=False, test_cases=TEST_CASES)
    
    # 测试基于BERT的意图识别
    print("\n2. 测试基于BERT的意图识别:")
    avg_time_bert, results_bert = _test_intent_recognition(use_bert=True, test_cases=TEST_CASES)
    
    # 输出对比结果
    print("\n3. 性能对比结果:")
    print("=" * 50)
    print(f"平均响应时间 (TF-IDF): {avg_time_tfidf:.4f} 秒")
    print(f"平均响应时间 (BERT): {avg_time_bert:.4f} 秒")
    print(f"响应时间差异: {(avg_time_bert - avg_time_tfidf):.4f} 秒")
    
    # 输出详细结果对比
    print("\n4. 详细识别结果对比:")
    print("=" * 50)
    print(f"{'输入':<20} {'TF-IDF意图':<15} {'TF-IDF分数':<15} {'BERT意图':<15} {'BERT分数':<15} {'意图是否一致':<10}")
    print("-" * 95)
    
    for tfidf_result, bert_result in zip(results_tfidf, results_bert):
        input_text = tfidf_result['input']
        tfidf_intent = tfidf_result['intent'] or "None"
        tfidf_score = tfidf_result['score']
        bert_intent = bert_result['intent'] or "None"
        bert_score = bert_result['score']
        
        # 检查意图是否一致
        intent_match = "✓" if tfidf_intent == bert_intent else "✗"
        
        print(f"{input_text:<20} {tfidf_intent:<15} {tfidf_score:<15.4f} {bert_intent:<15} {bert_score:<15.4f} {intent_match:<10}")
    
    # 计算准确率差异（简化版，实际应该有标注数据）
    intent_matches = sum(1 for tfidf_res, bert_res in zip(results_tfidf, results_bert) 
                        if tfidf_res['intent'] == bert_res['intent'])
    match_rate = intent_matches / len(TEST_CASES) * 100
    
    print(f"\n意图识别一致性: {match_rate:.1f}%")
    
    # 计算平均分数差异
    avg_score_tfidf = sum(res['score'] for res in results_tfidf) / len(results_tfidf)
    avg_score_bert = sum(res['score'] for res in results_bert) / len(results_bert)
    
    print(f"\n平均意图分数 (TF-IDF): {avg_score_tfidf:.4f}")
    print(f"平均意图分数 (BERT): {avg_score_bert:.4f}")
    print(f"平均分数提升: {(avg_score_bert - avg_score_tfidf):.4f}")

if __name__ == "__main__":
    main()

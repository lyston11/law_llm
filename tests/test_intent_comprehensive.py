#!/usr/bin/env python3
"""
全面的意图识别测试脚本
测试基于BERT的意图识别在各种场景下的性能
"""

import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import random
from src.legacy.dialog import DialogManager
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from config import Config

# 创建配置实例
config = Config()

class IntentTestSuite:
    """
    意图识别测试套件
    """
    
    def __init__(self):
        self.dialog_manager = DialogManager(use_bert_intent=True)
        self.test_results = {}
        
    def prepare_memory(self):
        """
        准备对话记忆
        """
        return {
            'avaiable_nodes': list(self.dialog_manager.node_id2node_info.keys()),
            'user_input': "",
            'entities': {},
            'slot_filled': {},
            'state': {},
            'hit_intent': None,
            'hit_intent_score': 0
        }
    
    def load_test_data(self, file_path=None):
        """
        加载测试数据
        """
        if file_path is None:
            file_path = os.path.join(config.ANNOTATION_DIR, "intent_annotation_merged.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_intent_labels(self):
        """
        获取所有意图标签
        """
        return list(self.dialog_manager.node_id2node_info.keys())
    
    def test_basic_intent_recognition(self, test_cases=None):
        """
        基础意图识别测试
        """
        print("1. 基础意图识别测试")
        print("=" * 60)
        
        if not test_cases:
            test_cases = [
                "我有法律问题",
                "我想咨询劳动合同问题",
                "公司拖欠工资怎么办",
                "我被公司辞退了",
                "离婚财产怎么分割",
                "交通事故赔偿标准",
                "房产继承纠纷",
                "专利申请流程",
                "合同纠纷怎么解决",
                "行政复议程序",
                "著作权保护期限",
                "遗产继承顺序",
                "医疗事故赔偿",
                "民间借贷利率",
                "刑事案件辩护"
            ]
        
        memory = self.prepare_memory()
        results = []
        total_time = 0
        
        for test_input in test_cases:
            memory['user_input'] = test_input
            
            start_time = time.time()
            result = self.dialog_manager.intent_recognizer.intent_recognize(memory.copy())
            end_time = time.time()
            
            response_time = end_time - start_time
            total_time += response_time
            
            results.append({
                'input': test_input,
                'intent': result['hit_intent'],
                'score': result['hit_intent_score'],
                'response_time': response_time
            })
        
        avg_time = total_time / len(test_cases)
        
        # 输出结果
        print(f"平均响应时间: {avg_time:.4f}秒")
        print("\n详细结果:")
        print(f"{'输入':<25} {'识别意图':<25} {'置信度':<10} {'响应时间(ms)':<15}")
        print("-" * 75)
        
        for res in results:
            print(f"{res['input']:<25} {res['intent']:<25} {res['score']:<10.4f} {res['response_time']*1000:<15.2f}")
        
        self.test_results['basic_test'] = {
            'avg_response_time': avg_time,
            'results': results
        }
        
        print()
    
    def test_classification_accuracy(self, test_size=0.2):
        """
        测试分类准确率
        """
        print("2. 分类准确率测试")
        print("=" * 60)
        
        # 加载数据
        all_data = self.load_test_data()
        
        # 随机打乱数据
        random.shuffle(all_data)
        
        # 分割训练集和测试集
        test_split = int(len(all_data) * test_size)
        test_data = all_data[:test_split]
        
        print(f"测试样本数: {len(test_data)}")
        
        y_true = []
        y_pred = []
        scores = []
        memory = self.prepare_memory()
        
        for item in test_data:
            user_input = item['user_input']
            true_label = item['intent_label']
            
            memory['user_input'] = user_input
            result = self.dialog_manager.intent_recognizer.intent_recognize(memory.copy())
            
            pred_label = result['hit_intent']
            confidence = result['hit_intent_score']
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            scores.append(confidence)
        
        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)
        
        # 计算精确率、召回率、F1值
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # 计算没有被识别的样本数（返回None的情况）
        unrecognized = sum(1 for pred in y_pred if pred is None)
        
        # 输出结果
        print(f"分类准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1值: {f1:.4f}")
        print(f"未识别样本数: {unrecognized} ({unrecognized/len(test_data):.2%})")
        
        # 输出详细分类报告
        print("\n详细分类报告:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        self.test_results['accuracy_test'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'unrecognized': unrecognized,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        print()
    
    def test_confidence_thresholds(self, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], test_size=0.2):
        """
        测试不同置信度阈值下的性能
        优化：先计算所有样本的得分，再对不同阈值进行过滤，减少重复计算
        """
        print("3. 置信度阈值测试")
        print("=" * 60)
        
        # 加载数据
        all_data = self.load_test_data()
        random.shuffle(all_data)
        test_split = int(len(all_data) * test_size)
        test_data = all_data[:test_split]
        
        original_threshold = self.dialog_manager.intent_recognizer.confidence_threshold
        
        print(f"测试样本数: {len(test_data)}")
        print(f"原始置信度阈值: {original_threshold}")
        print()
        
        print(f"{'阈值':<8} {'准确率':<10} {'召回率':<10} {'未识别率':<10} {'平均置信度':<15}")
        print("-" * 53)
        
        # 先计算所有测试样本的原始结果（不应用阈值过滤）
        all_results = []
        memory = self.prepare_memory()
        
        # 临时将阈值设为0，获取所有预测结果
        self.dialog_manager.intent_recognizer.confidence_threshold = 0.0
        
        for item in test_data:
            user_input = item['user_input']
            true_label = item['intent_label']
            
            memory['user_input'] = user_input
            result = self.dialog_manager.intent_recognizer.intent_recognize(memory.copy())
            
            all_results.append({
                'true_label': true_label,
                'pred_label': result['hit_intent'],
                'confidence': result['hit_intent_score']
            })
        
        # 恢复原始阈值
        self.dialog_manager.intent_recognizer.confidence_threshold = original_threshold
        
        threshold_results = []
        
        # 对每个阈值，基于预计算的结果进行分析
        for threshold in thresholds:
            y_true = []
            y_pred = []
            total_confidence = 0
            recognized_count = 0
            
            for result in all_results:
                true_label = result['true_label']
                pred_label = result['pred_label']
                confidence = result['confidence']
                
                # 应用阈值过滤
                if confidence >= threshold:
                    # 置信度高于阈值，保留预测结果
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    total_confidence += confidence
                    recognized_count += 1
                else:
                    # 置信度低于阈值，预测结果为None
                    y_true.append(true_label)
                    y_pred.append(None)
            
            # 计算指标
            # 过滤掉预测为None的样本，只计算被识别样本的准确率
            recognized_y_true = [true for true, pred in zip(y_true, y_pred) if pred is not None]
            recognized_y_pred = [pred for pred in y_pred if pred is not None]
            
            accuracy = accuracy_score(recognized_y_true, recognized_y_pred) if recognized_count > 0 else 0
            recall = accuracy if recognized_count > 0 else 0
            
            unrecognized_rate = (len(y_pred) - recognized_count) / len(y_pred)
            avg_confidence = total_confidence / recognized_count if recognized_count > 0 else 0
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'recall': recall,
                'unrecognized_rate': unrecognized_rate,
                'avg_confidence': avg_confidence
            })
            
            print(f"{threshold:<8} {accuracy:<10.4f} {recall:<10.4f} {unrecognized_rate:<10.4f} {avg_confidence:<15.4f}")
        
        self.test_results['threshold_test'] = threshold_results
        
        print()
    
    def test_robustness(self):
        """
        测试模型鲁棒性
        """
        print("4. 鲁棒性测试")
        print("=" * 60)
        
        # 鲁棒性测试用例：包含拼写错误、口语化表达、噪音等
        robustness_test_cases = [
            # 正常输入
            ("公司拖欠工资怎么办", "正常输入"),
            # 拼写错误
            ("公司拖欠工姿怎么办", "拼写错误"),
            # 口语化表达
            ("老板不给工资咋整", "口语化表达"),
            # 长句
            ("我在一家公司工作了三年，现在公司拖欠我三个月的工资，我该怎么办才能要回我的工资", "长句"),
            # 短句
            ("工资拖欠", "短句"),
            # 带标点符号
            ("公司拖欠工资怎么办？", "带标点符号"),
            # 带表情符号
            ("公司拖欠工资怎么办😡", "带表情符号"),
            # 方言
            ("公司拖起工资不给咋个办", "方言"),
            # 重复词语
            ("公司拖欠拖欠工资工资怎么办", "重复词语"),
            # 模糊表达
            ("我有个关于工作的法律问题", "模糊表达")
        ]
        
        memory = self.prepare_memory()
        results = []
        
        print(f"{'测试类型':<15} {'输入':<30} {'识别意图':<25} {'置信度':<10}")
        print("-" * 80)
        
        for test_input, test_type in robustness_test_cases:
            memory['user_input'] = test_input
            result = self.dialog_manager.intent_recognizer.intent_recognize(memory.copy())
            
            results.append({
                'test_type': test_type,
                'input': test_input,
                'intent': result['hit_intent'],
                'score': result['hit_intent_score']
            })
            
            print(f"{test_type:<15} {test_input:<30} {result['hit_intent']:<25} {result['hit_intent_score']:<10.4f}")
        
        # 计算鲁棒性得分：能正确识别的测试用例比例
        # 这里简化处理，认为只要返回了意图就成功
        successful = sum(1 for res in results if res['intent'] is not None)
        robustness_score = successful / len(results)
        
        print(f"\n鲁棒性得分: {robustness_score:.4f} ({successful}/{len(results)})")
        
        self.test_results['robustness_test'] = {
            'score': robustness_score,
            'results': results
        }
        
        print()
    
    def test_category_performance(self, test_size=0.2):
        """
        测试不同类别上的表现
        """
        print("5. 类别性能测试")
        print("=" * 60)
        
        # 加载数据
        all_data = self.load_test_data()
        random.shuffle(all_data)
        test_split = int(len(all_data) * test_size)
        test_data = all_data[:test_split]
        
        # 按类别分组
        data_by_category = {}
        for item in test_data:
            category = item['intent_label']
            if category not in data_by_category:
                data_by_category[category] = []
            data_by_category[category].append(item)
        
        memory = self.prepare_memory()
        category_results = {}
        
        print(f"{'类别':<30} {'样本数':<8} {'准确率':<10} {'平均置信度':<15} {'未识别率':<10}")
        print("-" * 73)
        
        for category, items in data_by_category.items():
            y_true = []
            y_pred = []
            total_confidence = 0
            recognized_count = 0
            
            for item in items:
                user_input = item['user_input']
                true_label = item['intent_label']
                
                memory['user_input'] = user_input
                result = self.dialog_manager.intent_recognizer.intent_recognize(memory.copy())
                
                pred_label = result['hit_intent']
                confidence = result['hit_intent_score']
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                
                if pred_label is not None:
                    total_confidence += confidence
                    recognized_count += 1
            
            # 计算指标
            accuracy = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0
            avg_confidence = total_confidence / recognized_count if recognized_count > 0 else 0
            unrecognized_rate = (len(y_pred) - recognized_count) / len(y_pred)
            
            category_results[category] = {
                'sample_count': len(items),
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'unrecognized_rate': unrecognized_rate
            }
            
            print(f"{category:<30} {len(items):<8} {accuracy:<10.4f} {avg_confidence:<15.4f} {unrecognized_rate:<10.4f}")
        
        self.test_results['category_performance'] = category_results
        
        print()
    
    def test_tfidf_comparison(self):
        """
        与TF-IDF方法的对比测试
        """
        print("6. 与TF-IDF方法对比测试")
        print("=" * 60)
        
        test_cases = [
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
        
        # 初始化两个DialogManager，一个使用BERT，一个不使用
        bert_dialog = DialogManager(use_bert_intent=True)
        tfidf_dialog = DialogManager(use_bert_intent=False)
        
        memory = self.prepare_memory()
        
        print(f"{'输入':<20} {'TF-IDF意图':<25} {'TF-IDF置信度':<15} {'BERT意图':<25} {'BERT置信度':<15} {'意图是否一致':<10}")
        print("-" * 105)
        
        results = []
        
        for test_input in test_cases:
            memory['user_input'] = test_input
            
            # TF-IDF结果
            tfidf_result = tfidf_dialog.intent_recognizer.intent_recognize(memory.copy())
            tfidf_intent = tfidf_result['hit_intent']
            tfidf_score = tfidf_result['hit_intent_score']
            
            # BERT结果
            bert_result = bert_dialog.intent_recognizer.intent_recognize(memory.copy())
            bert_intent = bert_result['hit_intent']
            bert_score = bert_result['hit_intent_score']
            
            # 检查是否一致
            match = "✓" if tfidf_intent == bert_intent else "✗"
            
            results.append({
                'input': test_input,
                'tfidf_intent': tfidf_intent,
                'tfidf_score': tfidf_score,
                'bert_intent': bert_intent,
                'bert_score': bert_score,
                'match': match
            })
            
            print(f"{test_input:<20} {tfidf_intent:<25} {tfidf_score:<15.4f} {bert_intent:<25} {bert_score:<15.4f} {match:<10}")
        
        # 计算一致率
        match_count = sum(1 for res in results if res['match'] == "✓")
        match_rate = match_count / len(results)
        
        print(f"\n意图一致率: {match_rate:.2%} ({match_count}/{len(results)})")
        
        self.test_results['tfidf_comparison'] = {
            'match_rate': match_rate,
            'results': results
        }
        
        print()
    
    def generate_test_report(self):
        """
        生成测试报告
        """
        print("7. 测试报告总结")
        print("=" * 60)
        
        print("测试结果总结:")
        
        if 'accuracy_test' in self.test_results:
            acc_test = self.test_results['accuracy_test']
            print(f"- 分类准确率: {acc_test['accuracy']:.4f}")
            print(f"- 精确率: {acc_test['precision']:.4f}")
            print(f"- 召回率: {acc_test['recall']:.4f}")
            print(f"- F1值: {acc_test['f1']:.4f}")
            print(f"- 未识别样本率: {acc_test['unrecognized']/len(acc_test['y_true']):.2%}")
        
        if 'basic_test' in self.test_results:
            basic_test = self.test_results['basic_test']
            print(f"- 平均响应时间: {basic_test['avg_response_time']:.4f}秒")
        
        if 'robustness_test' in self.test_results:
            robustness_test = self.test_results['robustness_test']
            print(f"- 鲁棒性得分: {robustness_test['score']:.4f}")
        
        if 'threshold_test' in self.test_results:
            threshold_test = self.test_results['threshold_test']
            best_threshold = max(threshold_test, key=lambda x: x['accuracy'])
            print(f"- 最佳置信度阈值: {best_threshold['threshold']} (准确率: {best_threshold['accuracy']:.4f})")
        
        if 'tfidf_comparison' in self.test_results:
            tfidf_comp = self.test_results['tfidf_comparison']
            print(f"- 与TF-IDF意图一致率: {tfidf_comp['match_rate']:.2%}")
        
        print("\n测试完成！")
        
        # 保存测试结果到文件，只保存关键统计结果，不保存原始预测结果
        # 这样可以避免JSON序列化问题
        simple_results = {
            "summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "BGE + XGBoost"
            }
        }
        
        # 保存分类准确率测试结果
        if 'accuracy_test' in self.test_results:
            acc_test = self.test_results['accuracy_test']
            simple_results["accuracy_test"] = {
                "accuracy": float(acc_test['accuracy']),
                "precision": float(acc_test['precision']),
                "recall": float(acc_test['recall']),
                "f1": float(acc_test['f1']),
                "unrecognized_count": acc_test['unrecognized'],
                "test_samples": len(acc_test['y_true'])
            }
        
        # 保存基础测试结果
        if 'basic_test' in self.test_results:
            basic_test = self.test_results['basic_test']
            simple_results["basic_test"] = {
                "avg_response_time": float(basic_test['avg_response_time']),
                "test_cases": len(basic_test['results'])
            }
        
        # 保存鲁棒性测试结果
        if 'robustness_test' in self.test_results:
            robustness_test = self.test_results['robustness_test']
            simple_results["robustness_test"] = {
                "score": float(robustness_test['score']),
                "test_cases": len(robustness_test['results']),
                "successful_cases": sum(1 for res in robustness_test['results'] if res['intent'] is not None)
            }
        
        # 保存置信度阈值测试结果
        if 'threshold_test' in self.test_results:
            threshold_test = self.test_results['threshold_test']
            simple_results["threshold_test"] = {
                "thresholds": [float(t['threshold']) for t in threshold_test],
                "best_threshold": float(max(threshold_test, key=lambda x: x['accuracy'])['threshold']),
                "best_accuracy": float(max(threshold_test, key=lambda x: x['accuracy'])['accuracy'])
            }
        
        # 保存与TF-IDF对比结果
        if 'tfidf_comparison' in self.test_results:
            tfidf_comp = self.test_results['tfidf_comparison']
            simple_results["tfidf_comparison"] = {
                "match_rate": float(tfidf_comp['match_rate']),
                "test_cases": len(tfidf_comp['results']),
                "matches": sum(1 for res in tfidf_comp['results'] if res['match'] == "✓")
            }
        
        # 保存类别性能测试结果
        if 'category_performance' in self.test_results:
            category_perf = self.test_results['category_performance']
            simple_results["category_performance"] = {
                "categories": [{
                    "name": category,
                    "samples": data['sample_count'],
                    "accuracy": float(data['accuracy']),
                    "avg_confidence": float(data['avg_confidence']),
                    "unrecognized_rate": float(data['unrecognized_rate'])
                } for category, data in category_perf.items()]
            }
        
        with open("test_results.json", 'w', encoding='utf-8') as f:
            json.dump(simple_results, f, ensure_ascii=False, indent=2)
        
        print("测试结果已保存到 test_results.json")

def main():
    """
    主函数
    """
    print("意图识别全面测试套件")
    print("=" * 60)
    print("测试基于BERT的法律意图识别在各种场景下的性能")
    print()
    
    # 初始化测试套件
    test_suite = IntentTestSuite()
    
    # 执行所有测试
    test_suite.test_basic_intent_recognition()
    test_suite.test_classification_accuracy()
    test_suite.test_confidence_thresholds()
    test_suite.test_robustness()
    test_suite.test_category_performance()
    test_suite.test_tfidf_comparison()
    
    # 生成测试报告
    test_suite.generate_test_report()

if __name__ == "__main__":
    main()

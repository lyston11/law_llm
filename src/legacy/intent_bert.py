"""基于BERT的意图识别模块"""
from sentence_transformers import SentenceTransformer
from config import Config
import torch
import numpy as np
import joblib
import json
import os

config = Config()

# 微调后的意图分类器路径
CLASSIFIER_DIR = "models/intent_classifier"

class BERTIntentRecognizer:
    """
    基于BERT的意图识别类，负责识别用户输入的意图
    使用sentence-transformers库实现文本相似度计算
    """
    def __init__(self, node_id2node_info):
        # 初始化基础属性
        self.node_id2node_info = node_id2node_info
        self.model = None  # 确保model属性始终被初始化
        
        # 配置参数
        self.confidence_threshold = config.INTENT_CONFIDENCE_THRESHOLD
        self.use_hierarchical = config.HIERARCHICAL_INTENT_RECOGNITION
        
        # 加载微调后的意图分类器（如果存在）
        self.classifier = None
        self.label_map = None
        self.reverse_label_map = None
        self._load_fine_tuned_classifier()
        
        # 尝试加载BGE模型用于嵌入生成
        try:
            from sentence_transformers import SentenceTransformer
            # 移除local_files_only=True，允许模型从网络下载或更新，避免使用本地可能不安全的模型文件
            # 这样可以绕过PyTorch版本限制，使用最新的模型加载方式
            self.model = SentenceTransformer(config.BERT_MODEL_NAME, local_files_only=False)
            # 缓存意图向量，避免重复计算
            self.intent_vectors_cache = {}
            # 初始化时预计算所有意图的向量
            self._precompute_intent_vectors()
        except Exception as e:
            print(f"⚠️  无法加载BGE模型: {e}")
            self.model = None
            # 如果没有分类器，使用关键词匹配
            if self.classifier is None:
                self.intent_vectors_cache = {}
                print("⚠️  没有加载到分类器，将使用关键词匹配进行意图识别")
    
    def _load_fine_tuned_classifier(self):
        """
        加载微调后的意图分类器
        """
        if os.path.exists(CLASSIFIER_DIR):
            try:
                # 加载分类器
                classifier_path = os.path.join(CLASSIFIER_DIR, "intent_classifier.joblib")
                self.classifier = joblib.load(classifier_path)
                
                # 加载标签映射
                label_map_path = os.path.join(CLASSIFIER_DIR, "label_map.json")
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    self.label_map = json.load(f)
                
                # 创建反向标签映射
                self.reverse_label_map = {v: k for k, v in self.label_map.items()}
                
                print("✅ 成功加载微调后的意图分类器！")
            except Exception as e:
                print(f"⚠️  加载微调分类器失败: {e}")
                self.classifier = None
        else:
            print("⚠️  微调分类器目录不存在，将使用默认的相似度匹配方式")
    
    def _precompute_intent_vectors(self):
        """
        预计算所有节点意图的向量，提高实时识别效率
        """
        if self.model is None:
            return
            
        for node_id, node in self.node_id2node_info.items():
            for intent in node.get('intent', []):
                if intent not in self.intent_vectors_cache:
                    vector = self.model.encode(intent, convert_to_tensor=True)
                    self.intent_vectors_cache[intent] = vector
    
    def intent_score(self, node, memory):
        """
        计算意图分数
        
        Args:
            node (dict): 节点信息
            memory (dict): 对话记忆
            
        Returns:
            float: 意图分数
        """
        user_input = memory['user_input']
        max_score = 0
        
        # 如果有BGE模型，使用向量相似度计算
        if self.model is not None:
            # 计算用户输入的向量
            user_vector = self.model.encode(user_input, convert_to_tensor=True)
            
            for intent in node.get('intent', []):
                # 从缓存中获取意图向量
                intent_vector = self.intent_vectors_cache.get(intent)
                if intent_vector is None:
                    # 如果缓存中没有，计算并缓存
                    intent_vector = self.model.encode(intent, convert_to_tensor=True)
                    self.intent_vectors_cache[intent] = intent_vector
                
                # 计算余弦相似度
                score = torch.nn.functional.cosine_similarity(
                    user_vector.unsqueeze(0), 
                    intent_vector.unsqueeze(0)
                ).item()
                
                if score > max_score:
                    max_score = score
        else:
            # 没有BGE模型，使用简单的关键词匹配
            for intent in node.get('intent', []):
                # 计算用户输入和意图之间的关键词匹配度
                user_words = set(user_input.lower().split())
                intent_words = set(intent.lower().split())
                
                # 计算交集大小
                common_words = user_words.intersection(intent_words)
                if len(intent_words) > 0:
                    score = len(common_words) / len(intent_words)
                    if score > max_score:
                        max_score = score
        
        return max_score
    
    def _get_node_hierarchy_level(self, node_id):
        """
        获取节点的层次级别
        
        Args:
            node_id (str): 节点ID
            
        Returns:
            int: 层次级别，数字越小级别越高
        """
        # 根据节点ID前缀判断层次级别
        # 例如：node1是顶级节点，node2-10是二级节点，以此类推
        if node_id == 'node1':
            return 0
        elif len(node_id) == 5 and node_id.startswith('node'):
            # node2-node10
            return 1
        else:
            # 更复杂的节点ID，根据下划线分隔判断
            return len(node_id.split('_')) - 1
    
    def intent_recognize_hierarchical(self, memory):
        """
        层次化意图识别
        先识别高级别意图，再识别低级别意图，提高识别准确率
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            dict: 更新后的对话记忆，包含识别到的意图
        """
        # 按层次级别对节点进行分组
        nodes_by_level = {}
        for node_id in memory['avaiable_nodes']:
            level = self._get_node_hierarchy_level(node_id)
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node_id)
        
        # 按层次从高到低进行意图识别
        # 首先识别最高级别的意图
        sorted_levels = sorted(nodes_by_level.keys())
        
        best_intent = None
        best_score = -1
        current_intent = memory.get('hit_intent')
        current_score = memory.get('hit_intent_score', 0)
        
        # 检查是否已有当前意图，如果有，优先考虑其层次
        if current_intent:
            current_level = self._get_node_hierarchy_level(current_intent)
            # 先检查当前层次和更高层次
            priority_levels = [l for l in sorted_levels if l <= current_level] + [l for l in sorted_levels if l > current_level]
        else:
            priority_levels = sorted_levels
        
        for level in priority_levels:
            if level not in nodes_by_level:
                continue
            
            level_nodes = nodes_by_level[level]
            level_best_score = -1
            level_best_intent = None
            
            # 在当前层次中找到最佳意图
            for node_id in level_nodes:
                node = self.node_id2node_info[node_id]
                score = self.intent_score(node, memory)
                
                if score > level_best_score:
                    level_best_score = score
                    level_best_intent = node_id
            
            # 如果当前层次的最佳分数高于之前的最佳分数，更新最佳意图
            if level_best_score > best_score:
                best_score = level_best_score
                best_intent = level_best_intent
        
        # 增强意图识别稳定性：
        # 如果当前有意图，且新意图分数没有明显高于当前意图，保持当前意图
        if current_intent and best_intent and best_intent != current_intent:
            # 设置意图切换阈值
            threshold = config.INTENT_SWITCH_THRESHOLD
            
            # 基于分数差异判断是否切换意图
            if (best_score - current_score) < threshold:
                best_intent = current_intent
                best_score = current_score
        
        memory['hit_intent'] = best_intent
        memory['hit_intent_score'] = best_score
        memory['intent_level'] = self._get_node_hierarchy_level(best_intent) if best_intent else 0
        
        return memory
    
    def intent_recognize(self, memory):
        """
        识别用户意图
        结合微调分类器和传统相似度匹配的优势
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            dict: 更新后的对话记忆，包含识别到的意图
        """
        try:
            # 确保输入的memory包含user_input键
            if 'user_input' not in memory:
                raise ValueError("对话记忆中缺少user_input键")
            
            # 定义法律领域名称映射
            legal_field_map = {
                'LEGAL_CONSULTATION': '法律咨询',
                'LABOR_CONSULTATION': '劳动纠纷',
                'FAMILY_CONSULTATION': '婚姻家庭',
                'TRAFFIC_CONSULTATION': '交通事故',
                'REAL_ESTATE_CONSULTATION': '房产纠纷',
                'IP_CONSULTATION': '知识产权',
                'CRIMINAL_CONSULTATION': '刑事案件',
                'ADMINISTRATIVE_CONSULTATION': '行政诉讼',
                'CONTRACT_CONSULTATION': '合同纠纷',
                'OTHER_LEGAL_CONSULTATION': '其他法律问题'
            }
            
            # 首先尝试使用微调分类器
            classifier_result = None
            if self.classifier is not None:
                classifier_result = self._intent_recognize_with_classifier(memory.copy())
            
            # 同时使用传统相似度匹配
            similarity_result = None
            if self.use_hierarchical:
                similarity_result = self.intent_recognize_hierarchical(memory.copy())
            else:
                similarity_result = self._intent_recognize_basic(memory.copy())
            
            # 融合两种方法的结果
            result = self._merge_recognition_results(classifier_result, similarity_result, memory)
            
            # 确保结果包含user_input键
            result['user_input'] = memory['user_input']
            
            # 置信度阈值过滤
            if result['hit_intent_score'] < self.confidence_threshold:
                result['hit_intent'] = None
                result['hit_intent_score'] = 0
                result['confidence_level'] = 'low'
                # 置信度低时，直接标记为非法律问题
                result['is_legal_question'] = False
                result['legal_field'] = '其他类型'
            else:
                result['confidence_level'] = 'high'
                # 置信度高的情况下，肯定是法律问题
                result['is_legal_question'] = True
                
                # 转换为友好的法律领域名称
                if result['hit_intent'] and result['hit_intent'] in self.node_id2node_info:
                    node = self.node_id2node_info[result['hit_intent']]
                    action = node.get('action', ['LEGAL_CONSULTATION'])[0]
                    # 只使用明确的法律领域，不使用'OTHER_LEGAL_CONSULTATION'
                    if action != 'OTHER_LEGAL_CONSULTATION':
                        result['legal_field'] = legal_field_map.get(action, '法律咨询')
                    else:
                        # 对于其他法律问题，标记为非法律问题
                        result['is_legal_question'] = False
                        result['legal_field'] = '其他类型'
                else:
                    result['legal_field'] = '法律咨询'
            
            # 如果是非法律问题，将意图设置为None
            if not result.get('is_legal_question', False):
                result['hit_intent'] = None
            
            return result
        except Exception as e:
            import traceback
            error_info = f"意图识别失败: {e}\n{traceback.format_exc()}"
            print(error_info)
            # 返回默认结果，确保包含user_input键
            return {
                'user_input': memory.get('user_input', ''),
                'hit_intent': None,
                'hit_intent_score': 0,
                'confidence_level': 'low',
                'is_legal_question': True,  # 即使识别失败，也尝试将其视为法律问题
                'legal_field': '刑事',  # 默认使用刑事类型
                'intent_level': 0
            }
    
    def _merge_recognition_results(self, classifier_result, similarity_result, memory):
        """
        融合分类器和相似度匹配的结果
        
        Args:
            classifier_result (dict): 分类器识别结果
            similarity_result (dict): 相似度匹配结果
            memory (dict): 对话记忆
            
        Returns:
            dict: 融合后的结果
        """
        # 确保返回的结果包含user_input键
        if classifier_result and classifier_result['hit_intent'] and classifier_result['hit_intent_score'] >= self.confidence_threshold:
            # 确保分类器结果包含user_input
            if 'user_input' not in classifier_result and 'user_input' in memory:
                classifier_result['user_input'] = memory['user_input']
            return classifier_result
        
        # 否则使用相似度匹配结果
        # 确保相似度匹配结果包含user_input
        if similarity_result and 'user_input' not in similarity_result and 'user_input' in memory:
            similarity_result['user_input'] = memory['user_input']
        return similarity_result
    
    def _intent_recognize_with_classifier(self, memory):
        """
        使用微调后的分类器进行意图识别
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            dict: 更新后的对话记忆，包含识别到的意图
        """
        user_input = memory['user_input']
        
        # 如果没有BGE模型，无法生成嵌入，直接返回None
        if self.model is None:
            return None
        
        try:
            # 生成用户输入的嵌入
            user_embedding = self.model.encode(user_input, convert_to_tensor=False)
            
            # 使用分类器预测
            predicted_label_idx = self.classifier.predict([user_embedding])[0]
            
            # 获取预测概率
            predicted_proba = self.classifier.predict_proba([user_embedding])[0]
            
            # 获取预测概率最高的类别概率
            max_proba = predicted_proba[predicted_label_idx]
            
            # 转换为节点ID
            predicted_node_id = self.reverse_label_map.get(predicted_label_idx)
            
            # 应用置信度阈值过滤
            if max_proba < self.confidence_threshold:
                # 置信度太低，返回None，让系统回退到其他识别方法
                return None
            
            # 更新对话记忆
            memory['hit_intent'] = predicted_node_id
            memory['hit_intent_score'] = max_proba
            memory['intent_level'] = self._get_node_hierarchy_level(predicted_node_id) if predicted_node_id else 0
            
            return memory
        except Exception as e:
            print(f"⚠️  使用分类器进行意图识别失败: {e}")
            return None
    
    def _intent_recognize_basic(self, memory):
        """
        基础意图识别，不考虑层次关系
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            dict: 更新后的对话记忆，包含识别到的意图
        """
        # 从所有当前可以访问的节点中找到最高分节点
        max_score = -1
        hit_intent = None
        current_intent = memory.get('hit_intent')
        current_score = memory.get('hit_intent_score', 0)
        
        for node_id in memory['avaiable_nodes']:
            node = self.node_id2node_info[node_id]
            score = self.intent_score(node, memory)
            
            if score > max_score:
                max_score = score
                hit_intent = node_id      
        
        # 增强意图识别稳定性：
        # 如果当前有意图，且新意图分数没有明显高于当前意图，保持当前意图
        if current_intent and hit_intent and hit_intent != current_intent:
            # 设置意图切换阈值
            threshold = config.INTENT_SWITCH_THRESHOLD
            
            # 基于分数差异判断是否切换意图
            if (max_score - current_score) < threshold:
                hit_intent = current_intent
                max_score = current_score
        
        memory['hit_intent'] = hit_intent
        memory['hit_intent_score'] = max_score
        memory['intent_level'] = self._get_node_hierarchy_level(hit_intent) if hit_intent else 0
        
        return memory
    
    def add_new_intent(self, intent):
        """
        添加新意图并计算其向量
        
        Args:
            intent (str): 新的意图文本
        """
        if self.model is not None and intent not in self.intent_vectors_cache:
            vector = self.model.encode(intent, convert_to_tensor=True)
            self.intent_vectors_cache[intent] = vector

"""意图识别模块"""
from config import Config
config = Config()

class IntentRecognizer:
    """
    意图识别类，负责识别用户输入的意图
    """
    def __init__(self, tfidf_calculator, node_id2node_info):
        self.tfidf_calculator = tfidf_calculator
        self.node_id2node_info = node_id2node_info
    
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
        
        for intent in node.get('intent', []):
            # 计算用户输入与意图的相似度
            intent_vec = self.tfidf_calculator.calculate_tfidf(intent)
            user_vec = self.tfidf_calculator.calculate_tfidf(user_input)
            score = self.tfidf_calculator.cosine_similarity(intent_vec, user_vec)
            if score > max_score:
                max_score = score
        
        return max_score
    
    def intent_recognize(self, memory):
        """
        识别用户意图
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            dict: 更新后的对话记忆，包含识别到的意图
        """
        # 从所有当前可以访问的节点中找到最高分节点
        max_score = -1
        hit_intent = None
        hit_action = None
        current_intent = memory.get('hit_intent')
        current_score = memory.get('hit_intent_score', 0)
        
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
        
        for node_id in memory['avaiable_nodes']:
            node = self.node_id2node_info[node_id]
            score = self.intent_score(node, memory)
            
            if score > max_score:
                max_score = score
                hit_intent = node_id
                hit_action = node.get('action', ['LEGAL_CONSULTATION'])[0]      
        
        # 增强意图识别稳定性：
        # 如果当前有意图，且新意图分数没有明显高于当前意图，保持当前意图
        if current_intent and hit_intent and hit_intent != current_intent:
            # 设置意图切换阈值
            threshold = config.INTENT_SWITCH_THRESHOLD
            
            # 基于分数差异判断是否切换意图，不再基于节点类型
            # 这样可以确保不同法律领域之间能够正常切换
            if (max_score - current_score) < threshold:
                hit_intent = current_intent
                hit_action = self.node_id2node_info[current_intent].get('action', ['LEGAL_CONSULTATION'])[0]
                max_score = current_score
        
        # 转换为友好的法律领域名称
        legal_field = legal_field_map.get(hit_action, '法律咨询')
        
        # 置信度阈值过滤：如果得分低于阈值，使用默认意图
        if max_score < config.INTENT_CONFIDENCE_THRESHOLD:
            hit_intent = None
            # 置信度低时，直接标记为非法律问题
            is_legal_question = False
            # 如果是法律问题，保留为法律咨询；否则为其他类型
            legal_field = '其他类型'
            max_score = 0
        else:
            # 置信度高的情况下，默认为法律问题
            is_legal_question = True
            # 只使用明确的法律领域，不使用'OTHER_LEGAL_CONSULTATION'
            if hit_action == 'OTHER_LEGAL_CONSULTATION':
                is_legal_question = False
                legal_field = '其他类型'
        
        # 如果是非法律问题，将意图设置为None
        if not is_legal_question:
            hit_intent = None
        
        memory['hit_intent'] = hit_intent
        memory['hit_intent_score'] = max_score
        memory['legal_field'] = legal_field
        memory['is_legal_question'] = is_legal_question
        return memory

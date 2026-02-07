"""对话历史管理模块"""
import time
import json
from config import Config
config = Config()

class DialogHistoryManager:
    """
    对话历史管理类，负责管理用户的对话历史
    """
    def __init__(self, max_history_length=config.MAX_DIALOG_HISTORY):
        self.max_history_length = max_history_length
        # 使用字典存储对话历史，key为session_id，value为对话历史列表
        self.history_store = {}
    
    def add_turn(self, session_id, user_input, system_response, intent=None, sentiment=None, entities=None, legal_field=None, is_legal_question=None):
        """
        添加对话轮次到历史记录
        
        Args:
            session_id (str): 会话ID
            user_input (str): 用户输入
            system_response (str): 系统响应
            intent (str, optional): 识别到的意图节点ID，默认为None
            sentiment (str, optional): 情感分析结果，默认为None
            entities (dict, optional): 实体识别结果，默认为None
            legal_field (str, optional): 识别到的法律领域名称，默认为None
            is_legal_question (bool, optional): 是否为法律问题，默认为None
        """
        # 初始化会话历史
        if session_id not in self.history_store:
            self.history_store[session_id] = []
        
        # 创建对话轮次
        dialog_turn = {
            'timestamp': time.time(),
            'user_input': user_input,
            'system_response': system_response,
            'intent': intent,
            'legal_field': legal_field,
            'sentiment': sentiment,
            'entities': entities,
            'is_legal_question': is_legal_question
        }
        
        # 添加到历史记录
        self.history_store[session_id].append(dialog_turn)
        
        # 限制历史记录长度
        if len(self.history_store[session_id]) > self.max_history_length:
            self.history_store[session_id] = self.history_store[session_id][-self.max_history_length:]
    
    def get_history(self, session_id, limit=None):
        """
        获取对话历史
        
        Args:
            session_id (str): 会话ID
            limit (int, optional): 返回的历史记录数量，默认为None，返回所有
            
        Returns:
            list: 对话历史列表
        """
        if session_id not in self.history_store:
            return []
        
        history = self.history_store[session_id]
        if limit:
            return history[-limit:]
        return history
    
    def clear_history(self, session_id):
        """
        清理对话历史
        
        Args:
            session_id (str): 会话ID
        """
        if session_id in self.history_store:
            del self.history_store[session_id]
    
    def get_last_turn(self, session_id):
        """
        获取最后一轮对话
        
        Args:
            session_id (str): 会话ID
            
        Returns:
            dict or None: 最后一轮对话，或None（如果没有历史记录）
        """
        history = self.get_history(session_id)
        if history:
            return history[-1]
        return None
    
    def export_history(self, session_id, file_path=None):
        """
        导出对话历史
        
        Args:
            session_id (str): 会话ID
            file_path (str, optional): 导出文件路径，默认为None，返回JSON字符串
            
        Returns:
            str or None: 导出的JSON字符串，或None（如果指定了文件路径）
        """
        history = self.get_history(session_id)
        if not history:
            return None
        
        export_data = {
            'session_id': session_id,
            'export_time': time.time(),
            'history': history
        }
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            return None
        else:
            return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def import_history(self, file_path):
        """
        导入对话历史
        
        Args:
            file_path (str): 导入文件路径
            
        Returns:
            str or None: 导入的会话ID，或None（如果导入失败）
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            session_id = import_data.get('session_id')
            history = import_data.get('history', [])
            
            if session_id and history:
                self.history_store[session_id] = history
                return session_id
            return None
        except Exception as e:
            print(f"导入对话历史失败: {e}")
            return None
    
    def get_all_sessions(self):
        """
        获取所有会话ID
        
        Returns:
            list: 会话ID列表
        """
        return list(self.history_store.keys())
    
    def get_session_count(self):
        """
        获取会话数量
        
        Returns:
            int: 会话数量
        """
        return len(self.history_store)

"""对话总结模块"""
import time
import json
from config import Config
config = Config()

class DialogSummaryManager:
    """
    对话总结管理类，负责生成和管理对话总结
    """
    def __init__(self):
        # 使用字典存储对话总结，key为session_id，value为总结内容
        self.summary_store = {}
    
    def generate_summary(self, session_id, dialog_history):
        """
        生成对话总结
        
        Args:
            session_id (str): 会话ID
            dialog_history (list): 对话历史列表
            
        Returns:
            str: 对话总结
        """
        if not dialog_history:
            return "当前对话暂无内容"
        
        # 提取对话中的关键信息
        user_inputs = [turn['user_input'] for turn in dialog_history]
        system_responses = [turn['system_response'] for turn in dialog_history]
        intents = [turn['intent'] for turn in dialog_history if turn['intent']]
        sentiments = [turn['sentiment'] for turn in dialog_history if turn['sentiment']]
        
        # 生成总结
        summary = f"对话包含 {len(dialog_history)} 轮交互\n\n"
        
        # 添加用户主要问题
        if user_inputs:
            summary += f"用户主要问题：\n{chr(10).join(user_inputs[:3])}\n\n"  # 最多显示3个用户问题
        
        # 添加系统主要回答
        if system_responses:
            summary += f"系统主要回答：\n{chr(10).join(system_responses[:2])}\n\n"  # 最多显示2个系统回答
        
        # 添加意图分析
        if intents:
            # 统计意图出现次数
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            # 获取最频繁的意图
            most_common_intent = max(intent_counts, key=intent_counts.get)
            summary += f"主要意图：{most_common_intent}\n\n"
        
        # 添加情感分析
        if sentiments:
            # 统计情感分布
            sentiment_counts = {}
            for sentiment in sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            # 获取最频繁的情感
            most_common_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            summary += f"主要情感：{most_common_sentiment}\n"
        
        # 保存总结
        self.summary_store[session_id] = {
            'summary': summary,
            'generated_at': time.time(),
            'history_length': len(dialog_history)
        }
        
        return summary
    
    def get_summary(self, session_id):
        """
        获取对话总结
        
        Args:
            session_id (str): 会话ID
            
        Returns:
            str or None: 对话总结，或None（如果不存在）
        """
        summary_data = self.summary_store.get(session_id)
        if summary_data:
            return summary_data['summary']
        return None
    
    def update_summary(self, session_id, dialog_history):
        """
        更新对话总结
        
        Args:
            session_id (str): 会话ID
            dialog_history (list): 对话历史列表
            
        Returns:
            str: 更新后的对话总结
        """
        return self.generate_summary(session_id, dialog_history)
    
    def clear_summary(self, session_id):
        """
        清除对话总结
        
        Args:
            session_id (str): 会话ID
        """
        if session_id in self.summary_store:
            del self.summary_store[session_id]
    
    def export_summary(self, session_id, file_path=None):
        """
        导出对话总结
        
        Args:
            session_id (str): 会话ID
            file_path (str, optional): 导出文件路径，默认为None，返回JSON字符串
            
        Returns:
            str or None: 导出的JSON字符串，或None（如果指定了文件路径）
        """
        summary_data = self.summary_store.get(session_id)
        if not summary_data:
            return None
        
        export_data = {
            'session_id': session_id,
            'export_time': time.time(),
            'summary': summary_data['summary'],
            'generated_at': summary_data['generated_at'],
            'history_length': summary_data['history_length']
        }
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            return None
        else:
            return json.dumps(export_data, ensure_ascii=False, indent=2)

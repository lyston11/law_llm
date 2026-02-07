"""用户反馈管理模块"""
import time
import json
from collections import defaultdict

class FeedbackManager:
    """
    用户反馈管理类，负责存储和管理用户反馈
    """
    def __init__(self):
        # 使用字典存储反馈，key为反馈类型，value为反馈列表
        self.feedback_store = defaultdict(list)
        # 存储每个会话的反馈，key为session_id，value为反馈列表
        self.session_feedback = defaultdict(list)
    
    def submit_feedback(self, session_id, turn_index, rating, comment=None, feedback_type="general"):
        """
        提交用户反馈
        
        Args:
            session_id (str): 会话ID
            turn_index (int): 对话轮次索引
            rating (int): 评分，1-5分
            comment (str, optional): 反馈内容，默认为None
            feedback_type (str, optional): 反馈类型，默认为"general"
            
        Returns:
            str: 反馈ID
        """
        # 创建反馈ID
        feedback_id = f"feedback_{int(time.time() * 1000)}"
        
        # 创建反馈对象
        feedback = {
            'feedback_id': feedback_id,
            'session_id': session_id,
            'turn_index': turn_index,
            'rating': rating,
            'comment': comment,
            'feedback_type': feedback_type,
            'timestamp': time.time()
        }
        
        # 存储反馈
        self.feedback_store[feedback_type].append(feedback)
        self.session_feedback[session_id].append(feedback)
        
        return feedback_id
    
    def get_feedback_by_session(self, session_id):
        """
        获取指定会话的反馈
        
        Args:
            session_id (str): 会话ID
            
        Returns:
            list: 反馈列表
        """
        return self.session_feedback.get(session_id, [])
    
    def get_feedback_by_type(self, feedback_type):
        """
        获取指定类型的反馈
        
        Args:
            feedback_type (str): 反馈类型
            
        Returns:
            list: 反馈列表
        """
        return self.feedback_store.get(feedback_type, [])
    
    def get_all_feedback(self):
        """
        获取所有反馈
        
        Returns:
            list: 所有反馈列表
        """
        all_feedback = []
        for feedback_list in self.feedback_store.values():
            all_feedback.extend(feedback_list)
        return all_feedback
    
    def get_feedback_stats(self):
        """
        获取反馈统计信息
        
        Returns:
            dict: 反馈统计信息
        """
        all_feedback = self.get_all_feedback()
        total_feedback = len(all_feedback)
        
        if total_feedback == 0:
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                'feedback_type_distribution': {}
            }
        
        # 计算平均评分
        total_rating = sum(feedback['rating'] for feedback in all_feedback)
        average_rating = total_rating / total_feedback
        
        # 计算评分分布
        rating_distribution = defaultdict(int)
        for feedback in all_feedback:
            rating_distribution[feedback['rating']] += 1
        
        # 计算反馈类型分布
        feedback_type_distribution = defaultdict(int)
        for feedback in all_feedback:
            feedback_type_distribution[feedback['feedback_type']] += 1
        
        return {
            'total_feedback': total_feedback,
            'average_rating': round(average_rating, 2),
            'rating_distribution': dict(rating_distribution),
            'feedback_type_distribution': dict(feedback_type_distribution)
        }
    
    def export_feedback(self, file_path=None, feedback_type=None):
        """
        导出反馈数据
        
        Args:
            file_path (str, optional): 导出文件路径，默认为None，返回JSON字符串
            feedback_type (str, optional): 反馈类型，默认为None，导出所有类型
            
        Returns:
            str or None: 导出的JSON字符串，或None（如果指定了文件路径）
        """
        if feedback_type:
            feedback_list = self.get_feedback_by_type(feedback_type)
        else:
            feedback_list = self.get_all_feedback()
        
        export_data = {
            'export_time': time.time(),
            'feedback_count': len(feedback_list),
            'feedback': feedback_list
        }
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            return None
        else:
            return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def clear_feedback(self, session_id=None):
        """
        清除反馈数据
        
        Args:
            session_id (str, optional): 会话ID，默认为None，清除所有反馈
        """
        if session_id:
            # 清除指定会话的反馈
            if session_id in self.session_feedback:
                # 从反馈类型存储中删除相关反馈
                session_feedbacks = self.session_feedback[session_id]
                for feedback in session_feedbacks:
                    feedback_type = feedback['feedback_type']
                    self.feedback_store[feedback_type] = [f for f in self.feedback_store[feedback_type] if f['feedback_id'] != feedback['feedback_id']]
                # 从会话反馈存储中删除
                del self.session_feedback[session_id]
        else:
            # 清除所有反馈
            self.feedback_store.clear()
            self.session_feedback.clear()

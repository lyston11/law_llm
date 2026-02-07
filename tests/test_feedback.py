"""用户反馈管理模块测试"""
import pytest
import json
from src.feedback import FeedbackManager


class TestFeedbackManager:
    """用户反馈管理测试类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.feedback_manager = FeedbackManager()
    
    def test_submit_feedback(self):
        """测试提交反馈"""
        feedback_id = self.feedback_manager.submit_feedback(
            session_id="test_session_123",
            turn_index=1,
            rating=5,
            comment="测试反馈",
            feedback_type="general"
        )
        
        assert isinstance(feedback_id, str)
        assert feedback_id.startswith("feedback_")
    
    def test_get_feedback_by_session(self):
        """测试按会话获取反馈"""
        # 提交多条反馈
        self.feedback_manager.submit_feedback(
            session_id="test_session_123",
            turn_index=1,
            rating=5,
            comment="测试反馈1"
        )
        self.feedback_manager.submit_feedback(
            session_id="test_session_123",
            turn_index=2,
            rating=4,
            comment="测试反馈2"
        )
        self.feedback_manager.submit_feedback(
            session_id="test_session_456",
            turn_index=1,
            rating=3,
            comment="测试反馈3"
        )
        
        # 获取会话123的反馈
        session_feedback = self.feedback_manager.get_feedback_by_session("test_session_123")
        assert len(session_feedback) == 2
        assert all(f['session_id'] == "test_session_123" for f in session_feedback)
        
        # 获取不存在的会话反馈
        empty_feedback = self.feedback_manager.get_feedback_by_session("non_existent")
        assert empty_feedback == []
    
    def test_get_feedback_by_type(self):
        """测试按类型获取反馈"""
        # 提交不同类型的反馈
        self.feedback_manager.submit_feedback(
            session_id="test_session_123",
            turn_index=1,
            rating=5,
            feedback_type="general"
        )
        self.feedback_manager.submit_feedback(
            session_id="test_session_123",
            turn_index=2,
            rating=4,
            feedback_type="bug"
        )
        self.feedback_manager.submit_feedback(
            session_id="test_session_456",
            turn_index=1,
            rating=3,
            feedback_type="bug"
        )
        
        # 获取general类型反馈
        general_feedback = self.feedback_manager.get_feedback_by_type("general")
        assert len(general_feedback) == 1
        assert all(f['feedback_type'] == "general" for f in general_feedback)
        
        # 获取bug类型反馈
        bug_feedback = self.feedback_manager.get_feedback_by_type("bug")
        assert len(bug_feedback) == 2
        assert all(f['feedback_type'] == "bug" for f in bug_feedback)
        
        # 获取不存在的类型反馈
        empty_feedback = self.feedback_manager.get_feedback_by_type("non_existent")
        assert empty_feedback == []
    
    def test_get_all_feedback(self):
        """测试获取所有反馈"""
        # 初始状态应该没有反馈
        all_feedback = self.feedback_manager.get_all_feedback()
        assert len(all_feedback) == 0
        
        # 提交一些反馈
        self.feedback_manager.submit_feedback(
            session_id="test_session_123",
            turn_index=1,
            rating=5
        )
        self.feedback_manager.submit_feedback(
            session_id="test_session_456",
            turn_index=1,
            rating=4
        )
        
        # 再次获取所有反馈
        all_feedback = self.feedback_manager.get_all_feedback()
        assert len(all_feedback) == 2
    
    def test_get_feedback_stats(self):
        """测试获取反馈统计信息"""
        # 初始状态统计
        stats = self.feedback_manager.get_feedback_stats()
        assert stats['total_feedback'] == 0
        assert stats['average_rating'] == 0
        
        # 提交一些反馈
        self.feedback_manager.submit_feedback(
            session_id="test_session_123",
            turn_index=1,
            rating=5,
            feedback_type="general"
        )
        self.feedback_manager.submit_feedback(
            session_id="test_session_456",
            turn_index=1,
            rating=4,
            feedback_type="bug"
        )
        self.feedback_manager.submit_feedback(
            session_id="test_session_789",
            turn_index=1,
            rating=5,
            feedback_type="general"
        )
        
        # 再次获取统计信息
        stats = self.feedback_manager.get_feedback_stats()
        assert stats['total_feedback'] == 3
        assert stats['average_rating'] == round((5 + 4 + 5) / 3, 2)  # 期望值为4.67
        assert stats['rating_distribution'][5] == 2
        assert stats['rating_distribution'][4] == 1
        assert stats['feedback_type_distribution']['general'] == 2
        assert stats['feedback_type_distribution']['bug'] == 1
    
    def test_export_feedback(self):
        """测试导出反馈"""
        # 提交一些反馈
        self.feedback_manager.submit_feedback(
            session_id="test_session_123",
            turn_index=1,
            rating=5
        )
        
        # 导出为JSON字符串
        export_str = self.feedback_manager.export_feedback()
        assert isinstance(export_str, str)
        
        # 解析JSON字符串
        export_data = json.loads(export_str)
        assert isinstance(export_data, dict)
        assert 'export_time' in export_data
        assert export_data['feedback_count'] == 1
        assert len(export_data['feedback']) == 1
        
        # 按类型导出
        export_str_bug = self.feedback_manager.export_feedback(feedback_type="bug")
        export_data_bug = json.loads(export_str_bug)
        assert export_data_bug['feedback_count'] == 0
    
    def test_clear_feedback(self):
        """测试清除反馈"""
        # 提交一些反馈
        self.feedback_manager.submit_feedback(
            session_id="test_session_123",
            turn_index=1,
            rating=5
        )
        self.feedback_manager.submit_feedback(
            session_id="test_session_456",
            turn_index=1,
            rating=4
        )
        
        # 清除特定会话反馈
        self.feedback_manager.clear_feedback(session_id="test_session_123")
        session_123_feedback = self.feedback_manager.get_feedback_by_session("test_session_123")
        session_456_feedback = self.feedback_manager.get_feedback_by_session("test_session_456")
        assert len(session_123_feedback) == 0
        assert len(session_456_feedback) == 1
        
        # 清除所有反馈
        self.feedback_manager.clear_feedback()
        all_feedback = self.feedback_manager.get_all_feedback()
        assert len(all_feedback) == 0
        session_456_feedback = self.feedback_manager.get_feedback_by_session("test_session_456")
        assert len(session_456_feedback) == 0

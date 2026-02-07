"""对话历史管理模块测试"""
import pytest
import json
import tempfile
from src.history import DialogHistoryManager


class TestDialogHistoryManager:
    """对话历史管理测试类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.history_manager = DialogHistoryManager(max_history_length=10)
    
    def test_add_turn(self):
        """测试添加对话轮次"""
        self.history_manager.add_turn(
            session_id="test_session_123",
            user_input="你好",
            system_response="您好，有什么可以帮助您的？",
            intent="greeting",
            sentiment="neutral",
            entities={}
        )
        
        # 检查历史记录
        history = self.history_manager.get_history("test_session_123")
        assert len(history) == 1
        assert history[0]['user_input'] == "你好"
        assert history[0]['system_response'] == "您好，有什么可以帮助您的？"
        assert history[0]['intent'] == "greeting"
    
    def test_history_length_limit(self):
        """测试历史记录长度限制"""
        # 添加超过最大长度的对话轮次
        for i in range(15):
            self.history_manager.add_turn(
                session_id="test_session_123",
                user_input=f"测试输入{i}",
                system_response=f"测试响应{i}"
            )
        
        # 检查历史记录长度
        history = self.history_manager.get_history("test_session_123")
        assert len(history) == 10  # 最大长度限制为10
    
    def test_get_history(self):
        """测试获取对话历史"""
        # 添加多条对话
        for i in range(5):
            self.history_manager.add_turn(
                session_id="test_session_123",
                user_input=f"测试输入{i}",
                system_response=f"测试响应{i}"
            )
        
        # 获取所有历史
        all_history = self.history_manager.get_history("test_session_123")
        assert len(all_history) == 5
        
        # 获取限制数量的历史
        limited_history = self.history_manager.get_history("test_session_123", limit=3)
        assert len(limited_history) == 3
        assert [h['user_input'] for h in limited_history] == ["测试输入2", "测试输入3", "测试输入4"]
        
        # 获取不存在的会话历史
        empty_history = self.history_manager.get_history("non_existent")
        assert empty_history == []
    
    def test_get_last_turn(self):
        """测试获取最后一轮对话"""
        # 初始状态没有历史
        last_turn = self.history_manager.get_last_turn("test_session_123")
        assert last_turn is None
        
        # 添加对话后获取最后一轮
        self.history_manager.add_turn(
            session_id="test_session_123",
            user_input="你好",
            system_response="您好"
        )
        self.history_manager.add_turn(
            session_id="test_session_123",
            user_input="我想咨询法律问题",
            system_response="请问您想咨询哪方面的法律问题？"
        )
        
        last_turn = self.history_manager.get_last_turn("test_session_123")
        assert last_turn is not None
        assert last_turn['user_input'] == "我想咨询法律问题"
    
    def test_clear_history(self):
        """测试清除对话历史"""
        # 添加对话
        self.history_manager.add_turn(
            session_id="test_session_123",
            user_input="你好",
            system_response="您好"
        )
        
        # 清除历史
        self.history_manager.clear_history("test_session_123")
        
        # 检查历史是否被清除
        history = self.history_manager.get_history("test_session_123")
        assert history == []
    
    def test_export_history(self):
        """测试导出对话历史"""
        # 添加对话
        self.history_manager.add_turn(
            session_id="test_session_123",
            user_input="你好",
            system_response="您好"
        )
        
        # 导出为JSON字符串
        export_str = self.history_manager.export_history("test_session_123")
        assert isinstance(export_str, str)
        
        # 解析JSON字符串
        export_data = json.loads(export_str)
        assert export_data['session_id'] == "test_session_123"
        assert len(export_data['history']) == 1
        
        # 导出不存在的会话
        non_existent_export = self.history_manager.export_history("non_existent")
        assert non_existent_export is None
        
        # 使用临时文件测试导出到文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
            temp_file_path = f.name
        
        try:
            result = self.history_manager.export_history("test_session_123", file_path=temp_file_path)
            assert result is None
            
            # 检查文件是否创建成功
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                assert file_data['session_id'] == "test_session_123"
        finally:
            import os
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_get_all_sessions(self):
        """测试获取所有会话ID"""
        # 初始状态没有会话
        sessions = self.history_manager.get_all_sessions()
        assert sessions == []
        
        # 添加多个会话
        self.history_manager.add_turn(
            session_id="session_1",
            user_input="测试1",
            system_response="响应1"
        )
        self.history_manager.add_turn(
            session_id="session_2",
            user_input="测试2",
            system_response="响应2"
        )
        
        # 获取所有会话
        sessions = self.history_manager.get_all_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) == 2
        assert "session_1" in sessions
        assert "session_2" in sessions
    
    def test_get_session_count(self):
        """测试获取会话数量"""
        # 初始状态会话数量为0
        count = self.history_manager.get_session_count()
        assert count == 0
        
        # 添加多个会话
        self.history_manager.add_turn(
            session_id="session_1",
            user_input="测试1",
            system_response="响应1"
        )
        self.history_manager.add_turn(
            session_id="session_2",
            user_input="测试2",
            system_response="响应2"
        )
        
        # 检查会话数量
        count = self.history_manager.get_session_count()
        assert count == 2
    
    def test_import_history(self):
        """测试导入对话历史"""
        # 创建测试数据
        test_data = {
            "session_id": "imported_session",
            "export_time": 1234567890,
            "history": [
                {
                    "timestamp": 1234567890,
                    "user_input": "导入测试1",
                    "system_response": "导入响应1",
                    "intent": "test",
                    "sentiment": "neutral",
                    "entities": {}
                }
            ]
        }
        
        # 使用临时文件测试导入
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            temp_file_path = f.name
        
        try:
            # 导入历史
            imported_session_id = self.history_manager.import_history(temp_file_path)
            assert imported_session_id == "imported_session"
            
            # 检查导入的历史
            imported_history = self.history_manager.get_history("imported_session")
            assert len(imported_history) == 1
            assert imported_history[0]['user_input'] == "导入测试1"
            
            # 测试导入无效文件
            invalid_session_id = self.history_manager.import_history("non_existent_file.json")
            assert invalid_session_id is None
        finally:
            import os
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

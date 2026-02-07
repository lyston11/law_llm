"""对话管理模块测试"""
import pytest
from src.legacy.dialog import DialogManager

class TestDialogManager:
    """对话管理测试类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.dialog_manager = DialogManager()
    
    def test_initialization(self):
        """测试初始化"""
        assert isinstance(self.dialog_manager, DialogManager)
        assert self.dialog_manager.node_id2node_info
        assert self.dialog_manager.slot_info
    
    def test_process_input(self):
        """测试处理用户输入"""
        user_input = "我想咨询劳动法问题"
        response, memory = self.dialog_manager.process_input(user_input)
        assert isinstance(response, str)
        assert isinstance(memory, dict)
    
    def test_dialog_flow(self):
        """测试对话流程"""
        # 初始查询
        user_input1 = "我想咨询劳动法问题"
        response1, memory = self.dialog_manager.process_input(user_input1)
        
        # 后续查询
        user_input2 = "我被公司辞退了"
        response2, memory = self.dialog_manager.process_input(user_input2, memory)
        
        assert isinstance(response1, str)
        assert isinstance(response2, str)
        assert isinstance(memory, dict)
    
    def test_response_generation(self):
        """测试响应生成"""
        memory = {
            'user_input': '我想咨询劳动法问题',
            'avaiable_nodes': list(self.dialog_manager.node_id2node_info.keys())
        }
        
        # 进行NLU和DST处理
        memory = self.dialog_manager.nlu(memory)
        memory = self.dialog_manager.state_tracker.dst(memory)
        
        # 生成响应
        response = self.dialog_manager.generate_response(memory)
        assert isinstance(response, str)

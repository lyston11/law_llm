"""
智能问题推荐模块
基于对话上下文生成相关问题推荐
"""
import logging
import re
from typing import List, Dict, Any
from config import Config

logger = logging.getLogger(__name__)


class QuestionRecommender:
    """智能问题推荐器"""

    def __init__(self):
        """初始化推荐器"""
        self.model = getattr(Config, 'RECOMMEND_MODEL', 'qwen-turbo')
        self.timeout = getattr(Config, 'RECOMMEND_TIMEOUT', 5)
        self.count_range = getattr(Config, 'RECOMMEND_COUNT', (3, 5))
        logger.info(f"QuestionRecommender 初始化: model={self.model}, timeout={self.timeout}")

    def _should_skip(self, conversation_history: List[Dict[str, str]], response: str) -> bool:
        """
        判断是否应该跳过推荐生成

        Args:
            conversation_history: 对话历史
            response: AI 回复内容

        Returns:
            True 表示跳过，False 表示生成推荐
        """
        # 输入验证：处理 None 值
        if conversation_history is None:
            conversation_history = []
        if response is None:
            response = ""

        # 场景 1: 检查简短对话模式（使用词边界匹配避免误报）
        # 使用正则表达式进行精确匹配，避免子字符串误匹配
        # 例如："我很感谢你" 不应匹配到 "感谢"
        # 匹配整个字符串为问候语，或问候语后接标点符号
        short_patterns = [
            r"^你好[！。]?$",  # 匹配整个字符串为 "你好" 或 "你好！"
            r"^您好[！。]?$",
            r"^谢谢[！。]?$",
            r"^感谢[！。]?$",
            r"^再见[！。]?$",
            r"^拜拜[！。]?$"
        ]
        last_user_msg = ""
        if conversation_history:
            last_user_msg = conversation_history[-1].get("content", "")

        # 检查最后一条用户消息是否完全匹配简短模式
        user_msg_stripped = last_user_msg.strip()
        for pattern in short_patterns:
            if re.match(pattern, user_msg_stripped):
                logger.debug("跳过推荐: 简短对话模式")
                return True

        # 场景 2: 用户明确表示没有其他问题
        # 使用词边界匹配避免误报
        no_question_patterns = [
            r"没有其他问题",
            r"暂无其他",
            r"不需要了"
        ]
        for pattern in no_question_patterns:
            if re.search(pattern, response):
                logger.debug("跳过推荐: 用户表示无其他问题")
                return True

        # 场景 3: 回复过短
        # 阈值 30 字符：用于过滤简短回复，确保推荐质量
        # 过短的回复通常包含的信息量不足以生成有价值的推荐问题
        if len(response) < 30:
            logger.debug(f"跳过推荐: 回复过短 (长度={len(response)})")
            return True

        return False

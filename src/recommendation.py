"""
智能问题推荐模块
基于对话上下文生成相关问题推荐
"""
import logging
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

    def _should_skip(self, conversation_history: List[Dict], response: str) -> bool:
        """
        判断是否应该跳过推荐生成

        Args:
            conversation_history: 对话历史
            response: AI 回复内容

        Returns:
            True 表示跳过，False 表示生成推荐
        """
        # 场景 1: 检查简短对话模式
        short_patterns = ["你好", "您好", "谢谢", "感谢", "再见", "拜拜"]
        last_user_msg = ""
        if conversation_history:
            last_user_msg = conversation_history[-1].get("content", "")

        if any(pattern in last_user_msg for pattern in short_patterns):
            logger.debug("跳过推荐: 简短对话模式")
            return True

        # 场景 2: 用户明确表示没有其他问题
        if any(pattern in response for pattern in ["没有其他问题", "暂无其他", "不需要了"]):
            logger.debug("跳过推荐: 用户表示无其他问题")
            return True

        # 场景 3: 回复过短
        if len(response) < 30:
            logger.debug(f"跳过推荐: 回复过短 (长度={len(response)})")
            return True

        return False

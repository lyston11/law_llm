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

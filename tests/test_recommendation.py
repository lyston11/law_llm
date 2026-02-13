"""
测试推荐模块
"""
import pytest
from src.recommendation import QuestionRecommender


def test_recommender_initialization():
    """测试 QuestionRecommender 正确初始化"""
    recommender = QuestionRecommender()
    assert recommender.model == "qwen-turbo"
    assert recommender.timeout == 5
    assert recommender.count_range == (3, 5)

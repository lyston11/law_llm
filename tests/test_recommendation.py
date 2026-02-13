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


def test_should_skip_greeting():
    """测试打招呼场景应该跳过推荐"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "你好"}]
    response = "您好！有什么我可以帮您的？"
    assert recommender._should_skip(history, response) == True


def test_should_skip_thanks():
    """测试感谢场景应该跳过推荐"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "谢谢你的回答"}]
    response = "不客气！"
    assert recommender._should_skip(history, response) == True


def test_should_skip_short_response():
    """测试过短回复应该跳过推荐"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "好的"}]
    response = "好的"
    assert recommender._should_skip(history, response) == True


def test_should_not_skip_legal_question():
    """测试法律咨询场景不应该跳过"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "被公司辞退了怎么办"}]
    response = "根据劳动合同法第四十七条规定，用人单位违反劳动合同法规定解除或者终止劳动合同的，应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。"
    assert recommender._should_skip(history, response) == False

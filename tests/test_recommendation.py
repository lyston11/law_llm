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


def test_format_context():
    """测试上下文格式化"""
    recommender = QuestionRecommender()
    history = [
        {"role": "user", "content": "被辞退了"},
        {"role": "assistant", "content": "根据劳动合同法..."}
    ]
    actions = [
        {
            "tool": "search_legal_knowledge",
            "input": {"query": "辞退赔偿"},
            "result_summary": "找到 5 条相关法律资料"
        }
    ]
    response = "根据劳动合同法第四十七条..."

    context = recommender._format_context(history, actions, response)

    assert "# 对话历史" in context
    assert "# 工具调用记录" in context
    assert "# AI 回复" in context
    assert "search_legal_knowledge" in context
    assert "找到 5 条相关法律资料" in context


def test_format_context_empty_history():
    """测试空对话历史的格式化"""
    recommender = QuestionRecommender()
    history = []
    actions = []
    response = "这是一条回复"

    context = recommender._format_context(history, actions, response)

    assert "# 对话历史" in context
    assert "(无工具调用)" in context
    assert "# AI 回复" in context
    assert "这是一条回复" in context


def test_format_context_multiple_actions():
    """测试多个工具调用的格式化"""
    recommender = QuestionRecommender()
    history = [{"role": "user", "content": "查询劳动法"}]
    actions = [
        {"tool": "search_legal_knowledge", "result_summary": "找到 3 条法律"},
        {"tool": "search_lawsuit", "result_summary": "找到 2 个案例"}
    ]
    response = "根据搜索结果..."

    context = recommender._format_context(history, actions, response)

    assert "search_legal_knowledge" in context
    assert "search_lawsuit" in context
    assert "找到 3 条法律" in context
    assert "找到 2 个案例" in context


def test_format_context_with_none_values():
    """测试 None 值处理（防止 AttributeError 崩溃）"""
    recommender = QuestionRecommender()

    # 测试所有参数为 None
    context = recommender._format_context(None, None, None)
    assert "# 对话历史" in context
    assert "(无工具调用)" in context
    assert "# AI 回复" in context

    # 测试部分参数为 None
    context = recommender._format_context(
        [{"role": "user", "content": "test"}],
        None,
        None
    )
    assert "# 对话历史" in context
    assert "(无工具调用)" in context
    assert "# AI 回复" in context

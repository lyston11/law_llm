"""情感分析模块测试"""
import pytest
from src.sentiment import SentimentAnalyzer

class TestSentimentAnalyzer:
    """情感分析测试类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        """测试积极情感分析"""
        text = "谢谢，这个回答很好"
        sentiment, score = self.sentiment_analyzer.analyze_sentiment(text)
        assert sentiment == "positive"
        assert score["positive"] > 0
    
    def test_negative_sentiment(self):
        """测试消极情感分析"""
        text = "这个服务太糟糕了，我很生气"
        sentiment, score = self.sentiment_analyzer.analyze_sentiment(text)
        assert sentiment == "negative"
        assert score["negative"] > 0
    
    def test_neutral_sentiment(self):
        """测试中性情感分析"""
        text = "请问，这个问题怎么解决"
        sentiment, score = self.sentiment_analyzer.analyze_sentiment(text)
        assert sentiment == "neutral"
        assert score["neutral"] > 0

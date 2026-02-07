"""情感分析模块"""

class SentimentAnalyzer:
    """
    情感分析类，用于分析用户输入的情感倾向
    """
    def __init__(self):
        # 情感词典
        self.sentiment_dict = {
            'positive': [
                '谢谢', '感谢', '好的', '满意', '不错', '很好', '太棒了', '感谢', '谢谢', '感谢您'
            ],
            'negative': [
                '不', '不是', '不行', '不好', '不满意', '糟糕', '差', '坏', '讨厌', '生气',
                '难过', '痛苦', '悲伤', '愤怒', '失望', '委屈', '不公平', '投诉', '抗议'
            ],
            'neutral': [
                '请问', '咨询', '了解', '需要', '想', '知道', '如何', '怎样', '什么', '哪里'
            ]
        }
    
    def analyze_sentiment(self, text):
        """
        分析文本的情感倾向
        
        Args:
            text (str): 要分析的文本
            
        Returns:
            tuple: (主导情感, 情感得分字典)
        """
        sentiment_score = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        # 计算情感得分
        for sentiment, keywords in self.sentiment_dict.items():
            for keyword in keywords:
                if keyword in text:
                    sentiment_score[sentiment] += 1
        
        # 确定主导情感
        max_score = max(sentiment_score.values())
        if max_score == 0:
            return 'neutral', sentiment_score
        
        # 找出得分最高的情感
        dominant_sentiment = max(sentiment_score, key=sentiment_score.get)
        
        return dominant_sentiment, sentiment_score

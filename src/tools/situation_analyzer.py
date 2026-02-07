"""工具：法律场景分析

分析用户描述的法律场景，提取关键事实要素（实体、时间、金额、情感等）。
底层复用 src/entity.py 和 src/sentiment.py。
"""
import json

from src.entity import EntityRecognizer
from src.sentiment import SentimentAnalyzer

# 单例
_entity_recognizer = EntityRecognizer()
_sentiment_analyzer = SentimentAnalyzer()


# ==================== 工具定义 ====================

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "analyze_legal_situation",
        "description": (
            "分析用户描述的法律场景，自动提取其中的关键事实信息，"
            "包括涉及的人物、公司、时间、地点、情感倾向等。"
            "当用户描述一个复杂的法律纠纷场景时，可以先调用此工具提取关键要素，"
            "再结合知识库搜索给出更精准的回答。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "用户描述的法律场景文本",
                },
            },
            "required": ["text"],
        },
    },
}


# ==================== 工具实现 ====================

def execute(text: str) -> str:
    """
    执行法律场景分析

    Args:
        text: 用户描述的法律场景

    Returns:
        str: JSON 格式的分析结果
    """
    try:
        # 实体识别
        entities = _entity_recognizer.recognize_entities(text)

        # 情感分析
        sentiment, sentiment_scores = _sentiment_analyzer.analyze_sentiment(text)

        # 法律领域自动判断
        domain_keywords = {
            "劳动纠纷": ["劳动", "辞退", "解雇", "开除", "工资", "加班", "合同", "离职", "社保", "工伤", "裁员"],
            "婚姻家庭": ["婚姻", "离婚", "结婚", "子女", "继承", "家暴", "抚养", "财产分割"],
            "交通事故": ["交通", "车祸", "事故", "碰撞", "追尾", "肇事", "酒驾", "逃逸"],
            "房产纠纷": ["房产", "买房", "卖房", "租房", "产权", "拆迁", "装修", "物业"],
            "刑事案件": ["刑事", "犯罪", "盗窃", "诈骗", "故意伤害", "被打", "抢劫", "绑架"],
            "合同纠纷": ["合同", "违约", "买卖", "租赁", "借款", "担保"],
            "知识产权": ["知识产权", "专利", "商标", "著作权", "版权", "侵权"],
            "行政纠纷": ["行政", "诉讼", "复议", "处罚", "许可"],
        }

        detected_domains = []
        for domain, keywords in domain_keywords.items():
            matched = [kw for kw in keywords if kw in text]
            if matched:
                detected_domains.append({
                    "domain": domain,
                    "matched_keywords": matched,
                    "confidence": min(len(matched) / 3.0, 1.0),
                })

        # 按匹配数排序
        detected_domains.sort(key=lambda x: x["confidence"], reverse=True)

        return json.dumps({
            "success": True,
            "entities": {k: v for k, v in entities.items() if v},
            "sentiment": {
                "dominant": sentiment,
                "scores": sentiment_scores,
            },
            "detected_domains": detected_domains[:3],
            "text_length": len(text),
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"场景分析失败: {str(e)}",
        }, ensure_ascii=False)

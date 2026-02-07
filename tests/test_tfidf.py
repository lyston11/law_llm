"""TF-IDF计算模块测试"""
import pytest
from src.legacy.tfidf import TFIDFCalculator

class TestTFIDFCalculator:
    """TF-IDF计算测试类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.tfidf_calculator = TFIDFCalculator()
    
    def test_add_document(self):
        """测试添加文档"""
        self.tfidf_calculator.add_document("这是第一个测试文档")
        self.tfidf_calculator.add_document("这是第二个测试文档")
        assert len(self.tfidf_calculator.documents) == 2
        assert self.tfidf_calculator.doc_count == 2
    
    def test_tokenize(self):
        """测试分词功能"""
        text = "这是一个测试文档，包含一些标点符号！"
        tokens = self.tfidf_calculator.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "测试" in tokens  # 现在使用jieba分词，会产生"测试"这样的词
        assert "文档" in tokens
    
    def test_calculate_tf(self):
        """测试词频计算"""
        words = ["测试", "文档", "测试", "内容"]
        tf_dict = self.tfidf_calculator.calculate_tf(words)
        assert isinstance(tf_dict, dict)
        assert tf_dict["测试"] == 0.5  # 测试出现两次，总共有4个词
    
    def test_calculate_idf(self):
        """测试逆文档频率计算"""
        self.tfidf_calculator.add_document("这是第一个测试文档")
        self.tfidf_calculator.add_document("这是第二个测试文档")
        self.tfidf_calculator.calculate_idf()
        assert len(self.tfidf_calculator.idf_dict) > 0
    
    def test_calculate_tfidf(self):
        """测试TF-IDF计算"""
        self.tfidf_calculator.add_document("这是第一个测试文档")
        self.tfidf_calculator.calculate_idf()
        tfidf_dict = self.tfidf_calculator.calculate_tfidf("这是一个测试文档")
        assert isinstance(tfidf_dict, dict)
    
    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        vec1 = {"测试": 0.5, "文档": 0.5}
        vec2 = {"测试": 0.5, "文档": 0.5}
        vec3 = {"不同": 0.5, "内容": 0.5}
        
        similarity1 = self.tfidf_calculator.cosine_similarity(vec1, vec2)
        similarity2 = self.tfidf_calculator.cosine_similarity(vec1, vec3)
        
        # 使用近似相等的断言，处理浮点数精度问题
        assert abs(similarity1 - 1.0) < 0.0001  # 完全相同的向量相似度接近1
        assert abs(similarity2 - 0.0) < 0.0001  # 完全不同的向量相似度接近0

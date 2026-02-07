"""TF-IDF计算模块"""
import re
import jieba
from collections import defaultdict
from math import sqrt

class TFIDFCalculator:
    """
    TF-IDF计算器类，用于计算文档的TF-IDF向量
    """
    def __init__(self):
        self.documents = []
        self.vocabulary = set()
        self.tf_matrix = []
        self.idf_dict = {}
        self.doc_count = 0
        # 加载停用词
        self.stopwords = self.load_stopwords()
    
    def load_stopwords(self):
        """
        加载停用词列表
        
        Returns:
            set: 停用词集合
        """
        # 常用中文停用词
        stopwords = {
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '有', '在', '对', '以', '于', '为', '一个', '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '这', '那', '这些', '那些', '这里', '那里', '这里', '那里', '自己', '自己的', '这个', '那个', '之一', '第一', '第二', '第三', '首先', '其次', '最后', '然后', '但是', '然而', '所以', '因此', '因为', '由于', '如果', '要是', '假设', '比如', '例如', '或者', '还是', '并且', '而且', '虽然', '尽管', '但是', '但是', '不过', '只是', '只有', '只要', '除非', '否则', '于是', '因此', '所以', '从而', '进而', '同时', '同样', '另外', '还有', '以及', '其他', '别的', '同样', '同样地', '同样的', '同样地', '类似', '相似', '相反', '反之', '比如', '例如', '诸如', '像', '正如', '正如', '比如', '例如', '诸如', '像', '比如', '例如', '诸如', '像', '比如', '例如', '诸如', '像', '比如', '例如', '诸如', '像'
        }
        return stopwords
    
    def add_document(self, text):
        """添加文档到语料库
        
        Args:
            text (str): 要添加的文档文本
        """
        # 使用jieba分词
        words = self.tokenize(text)
        self.documents.append(words)
        self.vocabulary.update(words)
        self.doc_count += 1
    
    def tokenize(self, text):
        """分词函数，使用jieba分词
        
        Args:
            text (str): 要分词的文本
            
        Returns:
            list: 分词后的词列表
        """
        # 去除标点符号
        text = re.sub(r'[\s+\!"#$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]+', ' ', text)
        # 使用jieba分词
        words = jieba.cut(text)
        # 去除停用词和空词
        tokens = [word.strip() for word in words if word.strip() and word not in self.stopwords]
        return tokens
    
    def calculate_tf(self, words):
        """计算文档的词频向量
        
        Args:
            words (list): 分词后的词列表
            
        Returns:
            dict: 词频字典
        """
        tf_dict = defaultdict(float)
        word_count = len(words)
        
        for word in words:
            tf_dict[word] += 1.0 / word_count
        
        return tf_dict
    
    def calculate_idf(self):
        """计算逆文档频率
        """
        # 统计每个词在多少个文档中出现
        doc_freq = defaultdict(int)
        for words in self.documents:
            unique_words = set(words)
            for word in unique_words:
                doc_freq[word] += 1
        
        # 计算IDF
        for word, freq in doc_freq.items():
            self.idf_dict[word] = 1 + sqrt(self.doc_count / (freq + 1))
    
    def calculate_tfidf(self, text):
        """计算给定文本的TF-IDF向量
        
        Args:
            text (str): 要计算TF-IDF的文本
            
        Returns:
            dict: TF-IDF向量字典
        """
        # 分词
        words = self.tokenize(text)
        if not words:
            return defaultdict(float)
        
        # 计算TF
        tf_dict = self.calculate_tf(words)
        
        # 计算TF-IDF
        tfidf_dict = defaultdict(float)
        for word, tf in tf_dict.items():
            if word in self.idf_dict:
                tfidf_dict[word] = tf * self.idf_dict[word]
        
        return tfidf_dict
    
    def cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度
        
        Args:
            vec1 (dict): 第一个向量
            vec2 (dict): 第二个向量
            
        Returns:
            float: 余弦相似度值
        """
        # 计算点积
        dot_product = 0.0
        for word, val in vec1.items():
            if word in vec2:
                dot_product += val * vec2[word]
        
        # 计算模长
        norm1 = sqrt(sum(val**2 for val in vec1.values()))
        norm2 = sqrt(sum(val**2 for val in vec2.values()))
        
        # 计算余弦相似度
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

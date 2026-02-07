"""实体识别模块"""
import re

class EntityRecognizer:
    """
    实体识别类，用于识别用户输入中的关键实体
    """
    def __init__(self):
        # 实体类型定义
        self.entity_patterns = {
            'PERSON': [
                r'[张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗]+[\u4e00-\u9fa5]{1,2}',  # 简化的人名匹配，常见姓氏+1-2个汉字
                r'[A-Za-z]+(?:\s[A-Za-z]+)'
            ],
            'COMPANY': [
                r'(?<![\u4e00-\u9fa5])[\u4e00-\u9fa5]+(?:公司|企业|集团|有限公司|股份有限公司|工作室|研究所)',  # 前面不能跟汉字
                r'[A-Za-z]+(?:\s[A-Za-z]+)*(?:Inc\.|Ltd\.|Company|Corporation|Group)'
            ],
            'TIME': [
                r'[12]\d{3}年(?:[01]?\d月(?:[0123]?\d日)?)?',
                r'[01]?\d月(?:[0123]?\d日)?',
                r'[0123]?\d日',
                r'[1-9]\d*年(?:[1-9]个月)?',  # 支持1-999年
                r'[1-9]个月',
                r'[一二三四五六七八九十百千]+年',  # 支持中文数字年
                r'几个月|几年|一年多|两年多|不到一年'
            ],
            'LOCATION': [
                r'[\u4e00-\u9fa5]+(?:省|市|县|区|乡|镇|村|街道)',
                r'北京|上海|广州|深圳|杭州|成都|重庆|武汉|西安|苏州|天津|南京|长沙|郑州|东莞|青岛|沈阳|宁波|昆明'
            ]
        }
    
    def recognize_entities(self, text):
        """
        识别文本中的实体
        
        Args:
            text (str): 要识别的文本
            
        Returns:
            dict: 实体类型到实体列表的映射
        """
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            all_matches = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    # 合并所有匹配结果，去除重复项
                    all_matches.extend(matches)
            
            if all_matches:
                # 去重并保存结果
                unique_matches = list(set(all_matches))
                entities[entity_type] = unique_matches
        
        return entities

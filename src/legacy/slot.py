"""槽位填充模块"""
import re

class SlotFiller:
    """
    槽位填充类，负责从用户输入中提取槽位信息
    """
    def __init__(self, slot_info):
        self.slot_info = slot_info
    
    def slot_filling(self, memory):
        """
        从用户输入中提取槽位信息
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            dict: 更新后的对话记忆，包含填充的槽位
        """
        user_input = memory['user_input']
        
        # 处理系统正在等待用户填充特定槽位的情况
        if memory.get('need_slot'):
            need_slot = memory['need_slot']
            # 直接将用户输入作为该槽位的值
            memory[need_slot] = user_input
            # 清除need_slot标记
            memory['need_slot'] = None
            return memory
        
        # 处理hit_intent为None的情况
        hit_intent = memory['hit_intent']
        slots = []
        if hit_intent and hit_intent in memory['node_id2node_info']:
            slots = memory['node_id2node_info'][hit_intent].get('slot', [])
        
        # 槽位填充增强：使用通用化的关键词匹配和正则表达式结合的方式
        # 简化为更通用的配置，便于扩展到其他法律领域
        general_slot_config = {
            # 法律类型自动识别关键词
            'legal_types': {
                '劳动': ['劳动', '辞退', '解雇', '开除', '工资', '加班', '合同', '离职', '社保', '工伤'],
                '婚姻': ['婚姻', '离婚', '结婚', '子女', '继承', '家暴'],
                '交通': ['交通', '车祸', '事故', '碰撞', '追尾', '肇事'],
                '房产': ['房产', '买房', '卖房', '租房', '产权', '拆迁', '装修'],
                '刑事': ['刑事', '犯罪', '罪名', '盗窃', '故意伤害', '被打', '打人', '打架', '伤害']
            },
            # 子类型自动识别关键词（如劳动问题类型）
            'subtypes': {
                '#劳动问题类型#': {
                    '辞退': ['辞退', '解雇', '开除', '离职', '炒鱿鱼', '裁员', '非法解除', '解除合同'],
                    '工资': ['工资', '薪酬', '薪水', '薪资', '报酬', '拖欠', '克扣'],
                    '加班': ['加班', '加班费', '超时', '夜班', '周末加班', '法定节假日加班'],
                    '合同': ['合同', '劳动合同', '协议', '签订', '续签', '变更'],
                    '社保': ['社保', '社会保险', '五险一金', '公积金', '缴纳', '补缴'],
                    '工伤': ['工伤', '工伤赔偿', '工伤认定', '伤残', '受伤', '索赔', '腿断', '骨折', '伤残鉴定']
                }
            }
        }
        
        for slot in slots:
            if slot not in self.slot_info:  # 跳过不存在的槽位
                continue
            query, values = self.slot_info[slot]
            matched_value = None
            
            # 1. 首先尝试正则表达式匹配 - 通用逻辑
            try:
                match = re.search(values, user_input)
                if match:
                    matched_value = match.group()
            except re.error:
                # 正则表达式无效时跳过
                pass
            
            # 2. 如果正则匹配失败，尝试关键词匹配
            if not matched_value:
                # 法律类型特殊处理
                if slot == '#法律类型#':
                    for legal_type, keywords in general_slot_config['legal_types'].items():
                        if any(keyword in user_input for keyword in keywords):
                            matched_value = legal_type
                            break
                # 子类型处理（如劳动问题类型）
                elif slot in general_slot_config['subtypes']:
                    for subtype_value, keywords in general_slot_config['subtypes'][slot].items():
                        if any(keyword in user_input for keyword in keywords):
                            matched_value = subtype_value
                            break
            
            # 3. 如果仍未匹配，尝试从对话历史中获取
            if not matched_value:
                recent_history = memory.get('dialog_history', [])
                if recent_history:
                    # 检查最近的对话轮次，包括系统回复和用户输入
                    for turn in reversed(recent_history[:-1]):  # 排除当前轮次
                        turn_content = turn.get('user_input', '') + ' ' + turn.get('system_response', '')
                        
                        # 尝试正则表达式匹配历史对话内容
                        try:
                            match = re.search(values, turn_content)
                            if match:
                                matched_value = match.group()
                                break
                        except re.error:
                            # 正则表达式无效时跳过
                            pass
                        
                        # 法律类型特殊处理
                        if slot == '#法律类型#' and not matched_value:
                            for legal_type, keywords in general_slot_config['legal_types'].items():
                                if any(keyword in turn_content for keyword in keywords):
                                    matched_value = legal_type
                                    break
                        
                        # 子类型处理（如劳动问题类型）
                        elif slot in general_slot_config['subtypes'] and not matched_value:
                            for subtype_value, keywords in general_slot_config['subtypes'][slot].items():
                                if any(keyword in turn_content for keyword in keywords):
                                    matched_value = subtype_value
                                    break
                        
                        # 尝试从历史对话中提取公司名称
                        if slot == '#用人单位#' and not matched_value:
                            company_pattern = r'[\u4e00-\u9fa5]+(公司|企业|单位)'
                            company_match = re.search(company_pattern, turn_content)
                            if company_match:
                                matched_value = company_match.group()
                                break
                        
                        # 尝试从历史对话中提取时间信息
                        if '时长' in slot and not matched_value:
                            time_patterns = [
                                r'([1-9]|[1-9]\d+)年',
                                r'([1-9]|[1-9]\d+)个月',
                                r'[一二三四五六七八九十百千]+年',
                                r'[一二三四五六七八九十]+个月',
                                r'几个月',
                                r'几年',
                                r'一年多',
                                r'两年多',
                                r'不到一年'
                            ]
                            for pattern in time_patterns:
                                time_match = re.search(pattern, turn_content)
                                if time_match:
                                    matched_value = time_match.group()
                                    break
                        
                        # 尝试从历史对话中提取位置信息
                        if slot == '#房屋位置#' and not matched_value:
                            location_pattern = r'北京|上海|广州|深圳|杭州|成都|重庆|武汉|西安|苏州|天津|南京|长沙|郑州|东莞|青岛|沈阳|宁波|昆明|合肥|福州|厦门|济南|哈尔滨|长春|大连|石家庄|太原|南昌|贵阳|南宁|兰州|银川|西宁|乌鲁木齐|呼和浩特|拉萨|海口|三亚'
                            location_match = re.search(location_pattern, turn_content)
                            if location_match:
                                matched_value = location_match.group()
                                break
                        
                        if matched_value:
                            break
            
            # 4. 如果所有方法都失败，根据槽位类型提供合理的默认值
            if not matched_value:
                # 通用默认值处理
                if '问题类型' in slot:
                    matched_value = '相关争议'
            
            if matched_value:
                # 通用槽位处理，减少特殊情况
                memory[slot] = matched_value
                
                # 自动关联槽位 - 通用逻辑
                # 如果填充了子类型槽位且没有法律类型，自动填充对应法律类型
                slot_legal_type_map = {
                    '#劳动问题类型#': '劳动',
                    '#婚姻问题类型#': '婚姻',
                    '#事故类型#': '交通',
                    '#房产问题类型#': '房产'
                }
                
                if slot in slot_legal_type_map and '#法律类型#' not in memory:
                    memory['#法律类型#'] = slot_legal_type_map[slot]
                
                # 公司/单位名称特殊处理 - 增强提取逻辑
                if slot == '#用人单位#' and matched_value:
                    # 尝试从用户输入中提取更准确的公司名称
                    # 使用更简单的正则表达式，避免过度复杂的排除逻辑
                    company_pattern = r'[\u4e00-\u9fa5]+(公司|企业|单位|有限公司|股份有限公司)'
                    company_match = re.search(company_pattern, user_input)
                    if company_match:
                        memory[slot] = company_match.group()
        
        # 时间类槽位增强 - 通用处理，在循环外统一处理所有时间槽位
        # 检查用户输入中是否包含时间信息
        time_patterns = [
            r'([1-9]|[1-9]\d+)年',
            r'([1-9]|[1-9]\d+)个月',
            r'[一二三四五六七八九十百千]+年',
            r'[一二三四五六七八九十]+个月',
            r'几个月',
            r'几年',
            r'一年多',
            r'两年多',
            r'不到一年',
            r'一年',
            r'两年',
            r'三年',
            r'四年',
            r'五年',
            r'十年'
        ]
        
        time_match = None
        for pattern in time_patterns:
            time_match = re.search(pattern, user_input)
            if time_match:
                break
        
        if time_match:
            time_value = time_match.group()
            # 根据当前法律类型，填充对应的时间槽位
            legal_type = memory.get('#法律类型#')
            if legal_type == '劳动':
                # 劳动领域的时间槽位
                if '#工作时长#' not in memory:
                    memory['#工作时长#'] = time_value
            elif legal_type == '婚姻':
                # 婚姻领域的时间槽位
                if '#婚姻时长#' not in memory:
                    memory['#婚姻时长#'] = time_value
        
        # 劳动问题类型自动识别
        if '#法律类型#' in memory and memory['#法律类型#'] == '劳动' and '#劳动问题类型#' not in memory:
            # 检查用户输入中是否包含劳动问题类型关键词
            labor_issue_keywords = {
                '辞退': ['辞退', '解雇', '开除', '离职', '炒鱿鱼', '裁员', '非法解除', '解除合同'],
                '工资': ['工资', '薪酬', '薪水', '薪资', '报酬', '拖欠', '克扣'],
                '加班': ['加班', '加班费', '超时', '夜班', '周末加班', '法定节假日加班'],
                '合同': ['合同', '劳动合同', '协议', '签订', '续签', '变更'],
                '社保': ['社保', '社会保险', '五险一金', '公积金', '缴纳', '补缴'],
                '工伤': ['工伤', '工伤赔偿', '工伤认定', '伤残', '受伤', '索赔', '腿断', '骨折', '伤残鉴定']
            }
            for issue_type, keywords in labor_issue_keywords.items():
                if any(keyword in user_input for keyword in keywords):
                    memory['#劳动问题类型#'] = issue_type
                    break
        
        # 增强槽位填充：通用法律类型自动识别
        if '#法律类型#' not in memory:
            # 使用通用配置中的法律类型关键词
            for legal_type, keywords in general_slot_config['legal_types'].items():
                if any(keyword in user_input for keyword in keywords):
                    memory['#法律类型#'] = legal_type
                    break
        
        # 增强槽位填充：处理用户明确提到的法律类型
        # 例如，用户说"劳动法"、"婚姻法"等
        legal_type_pattern = r'([\u4e00-\u9fa5]+)法'
        legal_type_match = re.search(legal_type_pattern, user_input)
        if legal_type_match:
            legal_type = legal_type_match.group(1)
            # 映射到标准法律类型
            legal_type_map = {
                '劳动': '劳动',
                '婚姻': '婚姻',
                '合同': '合同',
                '公司': '公司',
                '商标': '知识产权',
                '专利': '知识产权',
                '著作权': '知识产权',
                '行政': '行政',
                '刑事': '刑事',
                '交通': '交通',
                '房产': '房产'
            }
            if legal_type in legal_type_map:
                memory['#法律类型#'] = legal_type_map[legal_type]
        
        # 增强槽位填充：通用子类型自动识别
        # 根据当前法律类型，自动填充对应的子类型槽位
        legal_type = memory.get('#法律类型#')
        if legal_type:
            # 定义法律类型与子类型槽位的映射关系
            legal_subtype_map = {
                '劳动': '#劳动问题类型#',
                '婚姻': '#婚姻问题类型#',
                '交通': '#事故类型#',
                '房产': '#房产问题类型#'
            }
            
            if legal_type in legal_subtype_map:
                subtype_slot = legal_subtype_map[legal_type]
                if subtype_slot not in memory and subtype_slot in general_slot_config['subtypes']:
                    for subtype_value, keywords in general_slot_config['subtypes'][subtype_slot].items():
                        if any(keyword in user_input for keyword in keywords):
                            memory[subtype_slot] = subtype_value
                            break
        
        return memory

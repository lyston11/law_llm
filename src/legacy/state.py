"""对话状态跟踪模块"""
import time
import re
from config import Config
config = Config()

class DialogStateTracker:
    """
    对话状态跟踪类，负责管理对话状态
    """
    def __init__(self, node_id2node_info, slot_info):
        self.node_id2node_info = node_id2node_info
        self.slot_info = slot_info
    
    def dst(self, memory):
        """
        对话状态跟踪，智能管理对话状态
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            dict: 更新后的对话记忆，包含对话状态
        """
        try:
            hit_intent = memory['hit_intent']
            
            # 处理hit_intent为None的情况
            if hit_intent and hit_intent in self.node_id2node_info:
                slots = self.node_id2node_info[hit_intent].get('slot', [])
            else:
                slots = []
            
            # 1. 初始化对话历史
            if 'dialog_history' not in memory:
                memory['dialog_history'] = []
            
            # 2. 记录当前对话轮次
            current_turn = {
                'user_input': memory['user_input'],
                'hit_intent': hit_intent,
                'timestamp': time.time(),
                'sentiment': memory.get('sentiment', 'neutral'),
                'entities': memory.get('entities', {}),
                'filled_slots': [slot for slot in slots if slot in memory]
            }
            memory['dialog_history'].append(current_turn)
            
            # 3. 限制对话历史长度，避免内存占用过大
            if len(memory['dialog_history']) > config.MAX_DIALOG_HISTORY:
                memory['dialog_history'] = memory['dialog_history'][-config.MAX_DIALOG_HISTORY:]
            
            # 4. 槽位依赖关系管理
            slot_dependencies = config.SLOT_DEPENDENCIES
            
            # 5. 上下文感知的槽位填充
            # 利用对话历史和实体信息智能填充槽位
            self._context_aware_slot_filling(memory)
            
            # 6. 检查槽位填充情况，考虑依赖关系
            missing_slots = self._check_missing_slots(memory, slots, slot_dependencies)
            
            # 7. 智能对话策略：根据对话状态和用户输入动态调整
            memory = self._apply_dialog_strategy(memory, slots, missing_slots)
            
            # 8. 记录对话策略
            memory['dialog_state'] = self._update_dialog_state(memory, hit_intent, slots, missing_slots)
            
            return memory
        except Exception as e:
            import traceback
            error_info = f"对话状态跟踪失败: {e}\n{traceback.format_exc()}"
            print(error_info)
            # 确保返回的memory包含必要的键
            if 'dialog_history' not in memory:
                memory['dialog_history'] = []
            if 'dialog_state' not in memory:
                memory['dialog_state'] = {
                    'intent': None,
                    'filled_slots': [],
                    'missing_slots': [],
                    'optional_slots': [],
                    'filled_optional_slots': [],
                    'missing_optional_slots': [],
                    'turn_count': 0,
                    'last_updated': time.time(),
                    'strategy': 'professional',
                    'legal_type': memory.get('#法律类型#')
                }
            if 'need_slot' not in memory:
                memory['need_slot'] = None
            return memory
    
    def _context_aware_slot_filling(self, memory):
        """
        上下文感知的槽位填充，利用对话历史和实体信息智能填充槽位
        
        Args:
            memory (dict): 对话记忆
        """
        user_input = memory['user_input']
        entities = memory.get('entities', {})
        sentiment = memory.get('sentiment', 'neutral')
        
        # 1. 利用当前轮次的实体信息填充槽位
        if entities:
            # 如果识别到公司实体，填充#用人单位#槽位
            if 'COMPANY' in entities and '#用人单位#' not in memory:
                memory['#用人单位#'] = entities['COMPANY'][0]
            
            # 如果识别到时间实体，填充相应的时间槽位
            if 'TIME' in entities:
                legal_type = memory.get('#法律类型#')
                time_value = entities['TIME'][0]
                
                # 劳动领域
                if legal_type == '劳动' and '#工作时长#' not in memory:
                    memory['#工作时长#'] = time_value
                # 婚姻领域
                elif legal_type == '婚姻' and '#婚姻时长#' not in memory:
                    memory['#婚姻时长#'] = time_value
                # 其他领域
                elif '#工作时长#' not in memory and '#婚姻时长#' not in memory:
                    # 尝试填充合适的时间槽位
                    if legal_type == '房产' and '#房产问题类型#' in memory:
                        memory['#房产问题类型#'] = time_value
            
            # 如果识别到位置实体，填充#房屋位置#槽位
            if 'LOCATION' in entities and '#房屋位置#' not in memory:
                memory['#房屋位置#'] = entities['LOCATION'][0]
        
        # 2. 利用对话历史填充槽位
        dialog_history = memory.get('dialog_history', [])
        if len(dialog_history) > 1:
            # 从历史对话中提取信息
            for turn in reversed(dialog_history[:-1]):
                # 检查历史对话中的实体
                if 'entities' in turn:
                    hist_entities = turn['entities']
                    
                    # 尝试填充#用人单位#槽位
                    if 'COMPANY' in hist_entities and '#用人单位#' not in memory:
                        memory['#用人单位#'] = hist_entities['COMPANY'][0]
                    
                    # 尝试填充时间槽位
                    if 'TIME' in hist_entities:
                        legal_type = memory.get('#法律类型#')
                        time_value = hist_entities['TIME'][0]
                        
                        if legal_type == '劳动' and '#工作时长#' not in memory:
                            memory['#工作时长#'] = time_value
                        elif legal_type == '婚姻' and '#婚姻时长#' not in memory:
                            memory['#婚姻时长#'] = time_value
                    
                    # 尝试填充位置槽位
                    if 'LOCATION' in hist_entities and '#房屋位置#' not in memory:
                        memory['#房屋位置#'] = hist_entities['LOCATION'][0]
    
    def _check_missing_slots(self, memory, slots, slot_dependencies):
        """
        检查缺失的槽位，考虑依赖关系
        
        Args:
            memory (dict): 对话记忆
            slots (list): 当前节点的槽位列表
            slot_dependencies (dict): 槽位依赖关系
            
        Returns:
            list: 按优先级排序的缺失槽位列表
        """
        legal_type = memory.get('#法律类型#')
        
        # 定义法律类型与相关槽位的映射，按优先级排序
        legal_type_slots = {
            '劳动': ['#法律类型#', '#劳动问题类型#', '#用人单位#', '#工作时长#'],
            '婚姻': ['#法律类型#', '#婚姻问题类型#', '#婚姻时长#'],
            '交通': ['#法律类型#', '#事故类型#', '#责任方#'],
            '房产': ['#法律类型#', '#房产问题类型#', '#房屋位置#'],
            '知识产权': ['#法律类型#', '#知识产权类型#'],
            '刑事': ['#法律类型#', '#刑事罪名#'],
            '行政': ['#法律类型#', '#行政案件类型#'],
            '合同': ['#法律类型#', '#合同类型#', '#合同标的#']
        }
        
        # 如果没有法律类型，直接返回空列表，不询问任何槽位
        if not legal_type:
            return []
        
        # 获取当前法律类型的相关槽位，按优先级顺序
        relevant_slots = legal_type_slots.get(legal_type, [])
        
        # 只考虑相关的槽位，按照定义的优先级顺序
        relevant_slots_in_node = [slot for slot in relevant_slots if slot in slots]
        
        missing_slots = []
        
        for slot in relevant_slots_in_node:
            if slot not in memory:
                # 检查依赖槽位是否已填充
                dependencies = slot_dependencies.get(slot, [])
                all_dependencies_met = True
                
                for dep_slot in dependencies:
                    if dep_slot not in memory:
                        all_dependencies_met = False
                        if dep_slot not in missing_slots and dep_slot in relevant_slots_in_node:  # 只添加相关槽位
                            missing_slots.append(dep_slot)
                        break
                
                if all_dependencies_met and slot not in missing_slots:  # 避免重复添加
                    missing_slots.append(slot)
        
        # 按相关槽位的定义顺序排序，确保优先级正确
        ordered_missing_slots = []
        for slot in relevant_slots:
            if slot in missing_slots and slot not in ordered_missing_slots:
                ordered_missing_slots.append(slot)
        
        return ordered_missing_slots
    
    def _apply_dialog_strategy(self, memory, slots, missing_slots):
        """
        应用智能对话策略，根据对话状态和用户输入动态调整
        
        Args:
            memory (dict): 对话记忆
            slots (list): 当前节点的槽位列表
            missing_slots (list): 缺失的槽位列表
            
        Returns:
            dict: 更新后的对话记忆
        """
        # 获取当前对话状态
        sentiment = memory.get('sentiment', 'neutral')
        legal_type = memory.get('#法律类型#')
        turn_count = len(memory.get('dialog_history', []))
        
        # 1. 根据情感状态调整对话策略
        if sentiment == 'negative':
            # 对负面情感的用户，更主动地提供帮助
            memory['dialog_strategy'] = 'empathy'
        elif sentiment == 'positive':
            # 对正面情感的用户，保持友好并提供更多建议
            memory['dialog_strategy'] = 'friendly_advisory'
        else:
            # 对中性情感的用户，保持专业和中立
            memory['dialog_strategy'] = 'professional'
        
        # 2. 根据对话轮次调整策略
        if turn_count > 5:
            # 长对话时，尝试总结并提供解决方案
            memory['dialog_strategy'] = 'summarize_solution'
        
        # 3. 过滤掉不相关的槽位，只保留当前法律类型相关的槽位
        if legal_type:
            # 定义法律类型与相关槽位的映射
            legal_type_slots = {
                '劳动': ['#法律类型#', '#劳动问题类型#', '#用人单位#', '#工作时长#'],
                '婚姻': ['#法律类型#', '#婚姻问题类型#', '#婚姻时长#'],
                '交通': ['#法律类型#', '#事故类型#', '#责任方#'],
                '房产': ['#法律类型#', '#房产问题类型#', '#房屋位置#'],
                '知识产权': ['#法律类型#', '#知识产权类型#'],
                '刑事': ['#法律类型#', '#刑事罪名#'],
                '行政': ['#法律类型#', '#行政案件类型#'],
                '合同': ['#法律类型#', '#合同类型#', '#合同标的#']
            }
            
            # 获取当前法律类型的相关槽位
            relevant_slots = legal_type_slots.get(legal_type, [])
            
            # 过滤缺失槽位，只保留相关槽位
            relevant_missing_slots = [slot for slot in missing_slots if slot in relevant_slots]
            
            # 如果有相关的缺失槽位，使用相关槽位；否则，使用所有缺失槽位
            if relevant_missing_slots:
                missing_slots = relevant_missing_slots
        
        # 4. 选择最合适的缺失槽位进行反问
        if missing_slots:
            # 优先选择依赖关系最浅、最关键的槽位
            memory['need_slot'] = missing_slots[0]
        else:
            # 所有必需槽位已填充，应用更复杂的对话策略
            memory['need_slot'] = self._apply_optional_slot_strategy(memory, legal_type)
        
        return memory
    
    def _apply_optional_slot_strategy(self, memory, legal_type):
        """
        应用可选槽位策略，根据法律类型和对话状态选择是否询问更多信息
        
        Args:
            memory (dict): 对话记忆
            legal_type (str): 法律类型
            
        Returns:
            str or None: 可选槽位或None
        """
        # 定义不同法律类型的可选槽位和策略
        optional_slot_strategies = {
            '劳动': {
                'slots': ['#劳动问题类型#', '#用人单位#', '#工作时长#'],
                'strategy': 'ask_all_important'
            },
            '婚姻': {
                'slots': ['#婚姻问题类型#', '#婚姻时长#'],
                'strategy': 'ask_key_only'
            },
            '交通': {
                'slots': ['#事故类型#', '#责任方#'],
                'strategy': 'ask_all_important'
            },
            '房产': {
                'slots': ['#房产问题类型#', '#房屋位置#'],
                'strategy': 'ask_all_important'
            },
            '知识产权': {
                'slots': ['#知识产权类型#'],
                'strategy': 'ask_key_only'
            },
            '刑事': {
                'slots': ['#刑事罪名#'],
                'strategy': 'ask_key_only'
            },
            '行政': {
                'slots': ['#行政案件类型#'],
                'strategy': 'ask_key_only'
            },
            '合同': {
                'slots': ['#合同类型#', '#合同标的#'],
                'strategy': 'ask_all_important'
            }
        }
        
        # 确保法律类型存在且有效
        if not legal_type or legal_type not in optional_slot_strategies:
            return None
        
        # 获取当前法律类型的策略
        strategy = optional_slot_strategies[legal_type]
        optional_slots = strategy['slots']
        strategy_type = strategy['strategy']
        
        # 确保只处理当前法律类型相关的槽位
        legal_type_slots = {
            '劳动': ['#法律类型#', '#劳动问题类型#', '#用人单位#', '#工作时长#'],
            '婚姻': ['#法律类型#', '#婚姻问题类型#', '#婚姻时长#'],
            '交通': ['#法律类型#', '#事故类型#', '#责任方#'],
            '房产': ['#法律类型#', '#房产问题类型#', '#房屋位置#'],
            '知识产权': ['#法律类型#', '#知识产权类型#'],
            '刑事': ['#法律类型#', '#刑事罪名#'],
            '行政': ['#法律类型#', '#行政案件类型#'],
            '合同': ['#法律类型#', '#合同类型#', '#合同标的#']
        }
        
        # 获取当前法律类型的所有相关槽位
        relevant_slots = legal_type_slots.get(legal_type, [])
        
        # 根据策略类型选择可选槽位，但只返回相关槽位
        if strategy_type == 'ask_all_important':
            # 询问所有重要的可选槽位
            for slot in optional_slots:
                # 确保槽位是当前法律类型相关的
                if slot in relevant_slots and slot not in memory:
                    return slot
        elif strategy_type == 'ask_key_only':
            # 只询问关键槽位
            key_slots = [slot for slot in optional_slots if '#类型#' in slot and slot in relevant_slots]
            for slot in key_slots:
                if slot not in memory:
                    return slot
        
        return None
    
    def _update_dialog_state(self, memory, hit_intent, slots, missing_slots):
        """
        更新对话状态
        
        Args:
            memory (dict): 对话记忆
            hit_intent (str): 识别到的意图
            slots (list): 当前节点的槽位列表
            missing_slots (list): 缺失的槽位列表
            
        Returns:
            dict: 对话状态
        """
        legal_type = memory.get('#法律类型#')
        
        # 获取当前法律类型的可选槽位
        optional_slots = {}
        if legal_type:
            optional_slots = {
                '劳动': ['#劳动问题类型#', '#用人单位#', '#工作时长#'],
                '婚姻': ['#婚姻问题类型#', '#婚姻时长#'],
                '交通': ['#事故类型#', '#责任方#'],
                '房产': ['#房产问题类型#', '#房屋位置#'],
                '知识产权': ['#知识产权类型#'],
                '刑事': ['#刑事罪名#'],
                '行政': ['#行政案件类型#'],
                '合同': ['#合同类型#', '#合同标的#']
            }.get(legal_type, [])
        
        # 记录可选槽位的填充情况
        filled_optional_slots = [slot for slot in optional_slots if slot in memory]
        missing_optional_slots = [slot for slot in optional_slots if slot not in memory]
        
        return {
            'intent': hit_intent,
            'filled_slots': [slot for slot in slots if slot in memory],
            'missing_slots': missing_slots,
            'optional_slots': optional_slots,
            'filled_optional_slots': filled_optional_slots,
            'missing_optional_slots': missing_optional_slots,
            'turn_count': len(memory.get('dialog_history', [])),
            'last_updated': time.time(),
            'strategy': memory.get('dialog_strategy', 'professional'),
            'legal_type': legal_type
        }
    
    def detect_dialog_errors(self, memory):
        """
        对话错误检测，识别并修复常见的对话问题
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            list: 错误列表
        """
        dialog_history = memory.get('dialog_history', [])
        errors = []
        
        # 获取当前法律类型，用于后续检测
        legal_type = memory.get('#法律类型#')
        user_input = memory.get('user_input', '')
        
        # 1. 重复回答检测
        if len(dialog_history) >= 2:
            # 过滤掉没有system_response的历史记录
            valid_history = [turn for turn in dialog_history if 'system_response' in turn]
            if len(valid_history) >= 2:
                recent_responses = [turn['system_response'] for turn in valid_history[-3:]]
                # 检查是否有重复的连续回复
                for i in range(len(recent_responses) - 1):
                    if recent_responses[i] == recent_responses[i+1]:
                        errors.append({
                            'type': 'repeated_response',
                            'description': '系统重复了相同的回复',
                            'severity': 'high',
                            'context': f"连续回复: {recent_responses[i]}"
                        })
                        break
        
        # 2. 槽位冲突检测和修复
        filled_slots = [slot for slot in memory if slot.startswith('#')]
        if filled_slots and legal_type:
                # 定义不同法律类型的专属槽位
                legal_type_slots = config.LEGAL_TYPE_SLOTS
                
                # 获取当前法律类型的专属槽位
                current_slots = legal_type_slots.get(legal_type, [])
                
                # 检查所有填充的槽位是否属于当前法律类型
                for slot in list(filled_slots):
                    # 跳过法律类型本身
                    if slot == '#法律类型#':
                        continue
                    
                    # 检查槽位是否属于当前法律类型
                    is_valid_slot = False
                    for other_type, other_slots in legal_type_slots.items():
                        if other_type == legal_type and slot in other_slots:
                            is_valid_slot = True
                            break
                    
                    # 如果槽位不属于当前法律类型，清除该槽位
                    if not is_valid_slot:
                        errors.append({
                            'type': 'slot_conflict',
                            'description': f'槽位 {slot} 属于其他领域，与当前法律类型 {legal_type} 冲突',
                            'severity': 'high',
                            'context': f'当前法律类型: {legal_type}, 冲突槽位: {slot}'
                        })
                        # 清除冲突的槽位
                        del memory[slot]
        
        return errors

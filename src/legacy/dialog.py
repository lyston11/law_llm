"""旧版对话管理模块（v1.0/v2.0 规则系统）

已被 src/agent.py 的 DomainAgent（v3.0 Agent 架构）替代。
保留用于论文对比实验。
"""
import os
import json
import re
import pandas as pd
from config import Config
config = Config()

# 共用模块（仍在 src/ 根目录）
from src.sentiment import SentimentAnalyzer
from src.entity import EntityRecognizer
from src.rag import RAGRetriever

# 旧版专用模块（同在 src/legacy/ 下）
from .tfidf import TFIDFCalculator
from .intent import IntentRecognizer
from .intent_bert import BERTIntentRecognizer
from .slot import SlotFiller
from .state import DialogStateTracker

class DialogManager:
    """
    对话管理类，负责整体对话流程的管理
    """
    def __init__(self, use_qwen=True, use_bert_intent=None):
        self.use_qwen = use_qwen
        # 如果没有指定use_bert_intent，使用配置文件中的默认值
        self.use_bert_intent = use_bert_intent if use_bert_intent is not None else config.USE_BERT_INTENT
        self.qwen_api_key = config.API_KEY
        self.qwen_api_url = config.API_URL
        # 初始化TF-IDF计算器（无论使用哪种意图识别，都需要用于其他功能）
        self.tfidf_calculator = TFIDFCalculator()
        # 初始化实体识别器
        self.entity_recognizer = EntityRecognizer()
        # 初始化情感分析器
        self.sentiment_analyzer = SentimentAnalyzer()
        
        self.node_id2node_info = {}
        self.scenario = None
        self.slot_info = {}
        self.intent_recognizer = None
        self.slot_filler = None
        self.state_tracker = None
        
        # 初始化 RAG 检索器
        self.rag_retriever = None
        self.use_rag = config.USE_RAG
        if self.use_rag:
            try:
                self.rag_retriever = RAGRetriever()
            except Exception as e:
                print(f"⚠️  RAG 初始化失败，将使用非 RAG 模式: {e}")
                self.rag_retriever = None
        
        self.load()
    
    def load(self):
        """
        加载场景和槽位模板
        """
        # 加载场景
        self.load_scenario(os.path.join(config.SCENARIO_DIR, 'scenario-法律问答.json'))

        # 加载槽位模版
        try:
            self.slot_templates(os.path.join(config.SCENARIO_DIR, 'slot_fitting_templet-法律.xlsx'))
        except Exception as e:
            print(f"加载槽位模板失败: {e}")
            # 如果Excel加载失败，使用内置槽位信息
            self.default_slot_info()
        
        # 初始化各模块
        if self.use_bert_intent:
            # 使用基于BERT的意图识别器
            self.intent_recognizer = BERTIntentRecognizer(self.node_id2node_info)
        else:
            # 使用传统的TF-IDF意图识别器
            self.intent_recognizer = IntentRecognizer(self.tfidf_calculator, self.node_id2node_info)
        
        self.slot_filler = SlotFiller(self.slot_info)
        self.state_tracker = DialogStateTracker(self.node_id2node_info, self.slot_info)
   
    def load_scenario(self, scenario_file):
        """
        加载场景配置
        
        Args:
            scenario_file (str): 场景配置文件路径
        """
        scenario_name = os.path.basename(scenario_file).split('.')[0]
        with open(scenario_file, 'r', encoding='utf-8') as f:
            self.scenario = json.load(f)
        for node in self.scenario:
            node_id = node['id']
            node_id = scenario_name + '_' + node_id
            if 'childnode' in node:
                new_child = []
                for child in node.get('childnode', []):
                    child = scenario_name + '_' + child
                    new_child.append(child)
                node['childnode'] = new_child
            
            # 将节点的意图添加到TF-IDF语料库中
            for intent in node.get('intent', []):
                self.tfidf_calculator.add_document(intent)
            
            self.node_id2node_info[node_id] = node
        
        # 计算IDF
        self.tfidf_calculator.calculate_idf()
        print('场景加载成功')
    
    def slot_templates(self, slot_templet_file):
        """
        加载槽位模板
        
        Args:
            slot_templet_file (str): 槽位模板文件路径
        """
        # 定义备选模板文件列表
        template_files = [
            slot_templet_file,
            os.path.join(config.SCENARIO_DIR, 'slot_fitting_templet.xlsx')  # 备选模板文件
        ]
        
        # 初始化槽位信息为空字典
        self.slot_info = {}
        
        # 尝试可用的Excel引擎列表
        excel_engines = ['openpyxl']  # 只使用openpyxl引擎
        
        for template in template_files:
            try:
                # 检查文件是否存在且大小合理
                if os.path.exists(template) and os.path.getsize(template) > 100:  # 文件大小至少100字节
                    # 尝试使用可用的Excel引擎
                    for engine in excel_engines:
                        try:
                            df = pd.read_excel(template, engine=engine)
                            print(f"成功加载Excel槽位模板: {template} (使用{engine}引擎)")
                            # 处理Excel数据
                            for _, row in df.iterrows():
                                slot = row['slot']
                                query = row['query']
                                values = row['values']
                                self.slot_info[slot] = [query, values]
                            # 检查并修复槽位信息
                            self.check_and_fix_slot_info()
                            return
                        except Exception as e:
                            print(f"使用{engine}引擎加载Excel槽位模板失败: {e}")
                            continue  # 尝试下一个引擎
                elif os.path.exists(template):
                    print(f"Excel槽位模板文件太小或为空: {template}")
            except Exception as e:
                print(f"加载Excel槽位模板失败: {e}")
        
        # 如果所有模板文件都失败，使用默认槽位信息
        print(f"所有Excel槽位模板加载失败，将使用默认槽位信息")
        self.default_slot_info()
        # 检查并修复槽位信息
        self.check_and_fix_slot_info()
    
    def check_and_fix_slot_info(self):
        """
        检查并修复槽位信息，确保必要的槽位存在
        """
        # 定义必要的槽位列表
        essential_slots = {
            '#法律类型#': ['您想咨询哪方面的法律问题？', '劳动|婚姻|交通|房产|合同|刑事|行政|知识产权|辞退|解雇|开除|离婚|交通事故|房产纠纷|工资|加班|合同解除'],
            '#劳动问题类型#': ['具体是什么劳动问题？', '工资|加班|合同解除|试用期|社保|工伤|辞退|解雇|开除'],
            '#用人单位#': ['您的用人单位名称是？', '[一-龥]+公司|[一-龥]+企业|[一-龥]+单位'],
            '#工作时长#': ['您在该单位工作了多长时间？', '[1-9]年|[1-9]个月|几个月|几年']
        }
        
        # 检查必要槽位是否存在，不存在则添加默认值
        missing_slots = []
        for slot, default_info in essential_slots.items():
            if slot not in self.slot_info:
                self.slot_info[slot] = default_info
                missing_slots.append(slot)
        
        if missing_slots:
            print(f"已修复缺失的必要槽位: {missing_slots}")
    
    def default_slot_info(self):
        """
        默认槽位信息，当Excel加载失败时使用
        """
        self.slot_info = {
            '#法律类型#': ['您想咨询哪方面的法律问题？', '劳动|婚姻|交通|房产|知识产权|刑事|行政|合同|公司|海商|环境|金融|辞退|解雇|开除|离婚|交通事故|房产纠纷|工资|加班|合同解除|工伤|社保|继承|家暴|产权|拆迁|装修'],
            '#劳动问题类型#': ['具体是什么劳动问题？', '工资|加班|合同解除|试用期|社保|工伤|辞退|解雇|开除|离职|拖欠工资|未签合同|非法解除|经济补偿|赔偿金|竞业限制|年假|病假|调岗|降薪|年终奖|提成|加班费|双倍工资|社保补缴|公积金|劳动仲裁|劳动合同|试用期|转正'],
            '#用人单位#': ['您的用人单位名称是？', r'[一-龥]+公司|[一-龥]+企业|[一-龥]+单位|[一-龥]+有限公司|[一-龥]+股份有限公司|[A-Za-z]+(?:\s[A-Za-z]+)*(?:Inc\.|Ltd\.|Company|Corporation|Group)'],
            '#工作时长#': ['您在该单位工作了多长时间？', '[1-9]年|[1-9]个月|几个月|几年|一年多|两年多|不到一年|三年以上|五年以上|十年以上|一个月|两个月|三个月|半年|一年|两年|三年|四年|五年|十年|三十年'],
            '#婚姻问题类型#': ['具体是什么婚姻家庭问题？', '离婚|财产分割|子女抚养|继承|家暴|分居|重婚|遗弃|虐待|赡养|抚养费|赡养费|探视权|婚姻无效|撤销婚姻|彩礼|嫁妆|婚前财产|婚后财产|共同财产|个人财产|遗嘱|法定继承|代位继承|转继承'],
            '#婚姻时长#': ['您结婚多长时间了？', '[1-9]年|几个月|几年|一年多|两年多|不到一年|三年以上|五年以上|十年以上|一个月|两个月|三个月|半年|一年|两年|三年|四年|五年|十年|三十年'],
            '#事故类型#': ['具体是什么类型的交通事故？', '车祸|碰撞|追尾|刮擦|伤人|死亡|碾压|侧翻|翻车|撞人|撞车|撞墙|撞树|酒驾|醉驾|超速|闯红灯|逆行|无证驾驶|疲劳驾驶|肇事逃逸|追尾事故|碰撞事故|刮擦事故|伤人事故|死亡事故'],
            '#责任方#': ['事故责任方是谁？', '我|对方|我方|他方|机动车|非机动车|行人|司机|车主|乘客|酒驾|醉驾|超速|闯红灯|逆行|无证驾驶|疲劳驾驶|肇事逃逸'],
            '#房产问题类型#': ['具体是什么房产问题？', '买卖|租赁|产权|拆迁|装修|抵押|过户|继承|赠与|分割|纠纷|违约|退房|逾期|质量|物业费|停车费|中介费|定金|首付|贷款|产权证|房产证|土地证|不动产证|学区房|商品房|二手房|经济适用房|保障房|公租房|廉租房|小产权房|农村自建房|别墅|公寓|商铺|写字楼'],
            '#房屋位置#': ['房屋位于哪个城市？', '北京|上海|广州|深圳|杭州|成都|重庆|武汉|西安|苏州|天津|南京|长沙|郑州|东莞|青岛|沈阳|宁波|昆明|合肥|福州|厦门|济南|哈尔滨|长春|大连|石家庄|太原|南昌|贵阳|南宁|兰州|银川|西宁|乌鲁木齐|呼和浩特|拉萨|海口|三亚'],
            '#知识产权类型#': ['具体是什么知识产权问题？', '专利|商标|著作权|版权|域名|商业秘密|不正当竞争|专利申请|商标注册|版权登记|侵权|维权'],
            '#刑事罪名#': ['具体是什么刑事罪名？', '盗窃|故意伤害|诈骗|抢劫|强奸|贪污|受贿|职务侵占|交通肇事|危险驾驶|妨害公务|聚众斗殴|寻衅滋事|敲诈勒索|非法拘禁|绑架|故意杀人|过失致人死亡|放火|爆炸|投放危险物质'],
            '#行政案件类型#': ['具体是什么行政案件？', '行政诉讼|行政复议|行政处罚|行政许可|政府信息公开|行政强制|行政征收|行政给付|行政确认|行政裁决|行政协议|行政指导|行政调解|行政不作为'],
            '#合同类型#': ['具体是什么类型的合同？', '买卖合同|租赁合同|借款合同|劳动合同|服务合同|建设工程合同|委托合同|保管合同|运输合同|技术合同|赠与合同|租赁合同|融资租赁合同|承揽合同|行纪合同|居间合同|保理合同|物业服务合同|合伙合同'],
            '#合同标的#': ['合同的标的是什么？', '[\u4e00-\u9fa5]+|[A-Za-z]+(?:\\s[A-Za-z]+)*'],
            '#其他法律类型#': ['具体是什么类型的法律问题？', '公司|海商|环境|金融|票据|保险|证券|信托|基金|银行|税法|会计法|审计法|统计法|环境保护法|自然资源法|能源法|土地管理法|水法|矿产资源法|森林法|草原法|野生动物保护法|旅游法|劳动法|劳动合同法|社会保险法|劳动争议调解仲裁法|就业促进法|职业病防治法|安全生产法|工会法|妇女权益保障法|未成年人保护法|老年人权益保障法|残疾人保障法']
        }
    
    def nlu(self, memory):
        """
        自然语言理解，包括意图识别、情感分析、实体识别和槽位填充
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            dict: 更新后的对话记忆，包含NLU结果
        """
        try:
            # 确保输入的memory包含user_input键
            if 'user_input' not in memory:
                raise ValueError("对话记忆中缺少user_input键")
            
            user_input = memory['user_input']
            
            # 意图识别
            memory = self.intent_recognizer.intent_recognize(memory)
            
            # 确保意图识别后memory包含必要的键
            if 'hit_intent' not in memory:
                memory['hit_intent'] = None
            if 'hit_intent_score' not in memory:
                memory['hit_intent_score'] = 0
            if 'is_legal_question' not in memory:
                memory['is_legal_question'] = True
            if 'legal_field' not in memory:
                memory['legal_field'] = '刑事'
            
            # 检查用户输入中是否包含法律类型关键词，确保法律问题不被误判
            legal_type_keywords = {
                '劳动': ['劳动', '辞退', '解雇', '开除', '工资', '加班', '合同', '离职', '社保', '工伤'],
                '婚姻': ['婚姻', '离婚', '结婚', '子女', '继承', '家暴', '想离婚'],
                '交通': ['交通', '车祸', '事故', '碰撞', '追尾', '肇事'],
                '房产': ['房产', '买房', '卖房', '租房', '产权', '拆迁', '装修'],
                '知识产权': ['知识产权', '专利', '商标', '著作权', '版权'],
                '刑事': ['刑事', '犯罪', '罪名', '盗窃', '偷', '被偷', '偷盗', '失窃', '抢劫', '被抢', '抢夺', '故意伤害', '被打', '打人', '打架', '伤害'],
                '行政': ['行政', '诉讼', '复议', '处罚'],
                '合同': ['合同', '违约', '买卖', '租赁', '借款']
            }
            
            # 检查是否包含法律类型关键词
            detected_legal_type = None
            for legal_type, keywords in legal_type_keywords.items():
                for keyword in keywords:
                    if keyword in user_input:
                        detected_legal_type = legal_type
                        break
                if detected_legal_type:
                    break
            
            # 如果检测到法律类型，确保标记为法律问题
            if detected_legal_type:
                memory['is_legal_question'] = True
                memory['legal_field'] = detected_legal_type
                # 直接设置法律类型槽位
                memory['#法律类型#'] = detected_legal_type
            
            # 情感分析
            sentiment, sentiment_score = self.sentiment_analyzer.analyze_sentiment(user_input)
            memory['sentiment'] = sentiment
            memory['sentiment_score'] = sentiment_score
            
            # 实体识别
            entities = self.entity_recognizer.recognize_entities(user_input)
            memory['entities'] = entities
            
            # 将node_id2node_info添加到memory中，供slot_filling使用
            memory['node_id2node_info'] = self.node_id2node_info
            memory['slot_info'] = self.slot_info
            
            # 槽位填充
            memory = self.slot_filler.slot_filling(memory)
            
            # 使用识别到的实体增强槽位填充
            if entities:
                # 如果识别到公司实体，填充#用人单位#槽位
                if 'COMPANY' in entities and '#用人单位#' not in memory:
                    memory['#用人单位#'] = entities['COMPANY'][0]
                
                # 如果识别到时间实体，填充#工作时长#或#婚姻时长#槽位
                if 'TIME' in entities:
                    if '#工作时长#' not in memory and (memory.get('#法律类型#') == '劳动' or not memory.get('#法律类型#')):
                        memory['#工作时长#'] = entities['TIME'][0]
                    elif '#婚姻时长#' not in memory and memory.get('#法律类型#') == '婚姻':
                        memory['#婚姻时长#'] = entities['TIME'][0]
                
                # 如果识别到位置实体，填充#房屋位置#槽位
                if 'LOCATION' in entities and '#房屋位置#' not in memory and memory.get('#法律类型#') == '房产':
                    memory['#房屋位置#'] = entities['LOCATION'][0]
            
            return memory
        except Exception as e:
            # 如果NLU处理失败，返回默认值
            import traceback
            error_info = f"NLU处理失败: {e}\n{traceback.format_exc()}"
            print(error_info)
            
            user_input = memory.get('user_input', '')
            
            # 检查用户输入中是否包含法律类型关键词
            legal_type_keywords = {
                '劳动': ['劳动', '辞退', '解雇', '开除', '工资', '加班', '合同', '离职', '社保', '工伤'],
                '婚姻': ['婚姻', '离婚', '结婚', '子女', '继承', '家暴', '想离婚'],
                '交通': ['交通', '车祸', '事故', '碰撞', '追尾', '肇事'],
                '房产': ['房产', '买房', '卖房', '租房', '产权', '拆迁', '装修'],
                '知识产权': ['知识产权', '专利', '商标', '著作权', '版权'],
                '刑事': ['刑事', '犯罪', '罪名', '盗窃', '偷', '被偷', '偷盗', '失窃', '抢劫', '被抢', '抢夺', '故意伤害', '被打', '打人', '打架', '伤害'],
                '行政': ['行政', '诉讼', '复议', '处罚'],
                '合同': ['合同', '违约', '买卖', '租赁', '借款']
            }
            
            # 检查是否包含法律类型关键词
            detected_legal_type = None
            for legal_type, keywords in legal_type_keywords.items():
                for keyword in keywords:
                    if keyword in user_input:
                        detected_legal_type = legal_type
                        break
                if detected_legal_type:
                    break
            
            # 确保返回的memory包含所有必要的键
            result = {
                'user_input': user_input,
                'hit_intent': None,
                'hit_intent_score': 0,
                'is_legal_question': True if detected_legal_type else False,
                'legal_field': detected_legal_type if detected_legal_type else '刑事',
                'sentiment': 'neutral',
                'sentiment_score': 0.5,
                'entities': {},
                'node_id2node_info': self.node_id2node_info,
                'slot_info': self.slot_info,
                'nlu_error': str(e)
            }
            
            # 如果检测到法律类型，设置法律类型槽位
            if detected_legal_type:
                result['#法律类型#'] = detected_legal_type
            
            # 保留原memory中的其他键
            for key in memory:
                if key not in result:
                    result[key] = memory[key]
            
            return result
    
    def call_qwen(self, prompt):
        """
        调用Qwen API生成回答
        
        Args:
            prompt (str): 提示词
            
        Returns:
            str: 生成的回答
        """
        try:
            import requests
            import json
            
            # 使用最新的OpenAI兼容格式API URL
            api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            
            # 构造请求数据 - 使用OpenAI兼容格式
            data = {
                "model": "qwen-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一名专业的法律智能助手，请根据中国现行法律法规，为用户提供准确、专业、简洁的法律建议。回答不要超过500字，避免重复。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            # 构造请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.qwen_api_key}"
            }
            
            # 发送请求
            response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=30)
            
            # 打印响应内容，用于调试
            print(f"Qwen API响应: {response.text}")
            
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            elif "output" in result and "text" in result["output"]:
                return result["output"]["text"]
            elif "result" in result:
                # 处理不同的响应格式
                return result["result"]
            else:
                return "我不太理解您的问题，请您换个方式提问。"
        except Exception as e:
            print(f"调用Qwen API失败: {e}")
            # 返回友好的错误信息，同时保留本地生成的回答
            return f"根据您提供的信息，建议您咨询专业律师，了解相关法律法规，维护自己的合法权益。\n\n（注：当前API服务暂时不可用，以上回答为本地生成）"
    
    def _is_direct_knowledge_query(self, user_input):
        """
        判断用户输入是否为直接知识查询（如法条查询、概念查询），
        这类问题应直接通过 RAG 回答，不需要槽位填充。
        
        Args:
            user_input (str): 用户输入
            
        Returns:
            bool: 是否为直接知识查询
        """
        import re
        
        # 1. 法条查询：如"劳动法第二十条"、"民法典第1087条"
        law_article_patterns = [
            r'.{1,10}法.{0,5}第.{1,10}[条章节]',        # XX法第X条/章/节
            r'.{1,10}法.{0,5}第?\d+条',                  # XX法第123条
            r'第.{1,10}[条章节].{0,5}(规定|内容|说)',     # 第X条规定/内容
            r'.{1,10}(条例|规定|办法|细则)',              # XX条例/规定
        ]
        for pattern in law_article_patterns:
            if re.search(pattern, user_input):
                return True
        
        # 2. 概念/定义查询
        concept_patterns = [
            r'什么是.+',          # 什么是XX
            r'.+是什么',          # XX是什么
            r'.+的(定义|含义|意思|概念|区别|区分)',
            r'(解释|说明|介绍).+',
            r'.+(怎么理解|如何理解)',
            r'(哪些|有什么).+(情形|条件|要求|类型)',
        ]
        for pattern in concept_patterns:
            if re.search(pattern, user_input):
                return True
        
        # 3. 法律知识问答（不含个人情况）
        knowledge_indicators = [
            '规定了什么', '怎么规定', '如何规定', '法律规定',
            '法律依据', '法条', '司法解释', '最高法',
            '诉讼时效', '构成要件', '法律责任', '处罚标准',
        ]
        if any(indicator in user_input for indicator in knowledge_indicators):
            return True
        
        return False
    
    def _is_skip_slot_request(self, user_input):
        """
        判断用户是否要求跳过槽位填充，直接获得回答
        
        Args:
            user_input (str): 用户输入
            
        Returns:
            bool: 是否要求跳过
        """
        skip_phrases = [
            '先回答', '直接回答', '先告诉我', '先说', '别问了',
            '不想回答', '不要问了', '跳过', '不重要', '你先说',
            '回答我的问题', '先回复', '你倒是说啊', '快回答',
            '不用问', '不需要', '直接说',
        ]
        # 检查跳过短语
        if any(phrase in user_input for phrase in skip_phrases):
            return True
        # 只有问号或感叹号（用户表示不耐烦）
        if user_input.strip() in ['?', '？', '!', '！', '??', '？？', '...', '。。。']:
            return True
        return False
    
    def generate_response(self, memory):
        """
        生成系统响应（RAG 增强版）
        
        流程：意图识别 → RAG 检索相关法律资料 → 构建增强 Prompt → 大模型生成回答
        
        Args:
            memory (dict): 对话记忆
            
        Returns:
            str: 系统响应
        """
        # 检查是否是非法律问题
        is_legal_question = memory.get('is_legal_question', True)
        if not is_legal_question:
            return "抱歉，我无法回答这个问题，我主要专注于法律相关问题的咨询。"
        
        user_input = memory.get('user_input', '')
        
        # ========== 智能槽位策略 ==========
        # 判断是否为直接知识查询或用户要求跳过槽位
        is_direct_query = self._is_direct_knowledge_query(user_input)
        is_skip_request = self._is_skip_slot_request(user_input)
        
        # 检查槽位追问轮次（避免无限追问）
        slot_ask_count = memory.get('_slot_ask_count', 0)
        max_slot_asks = 2  # 最多追问 2 轮槽位
        
        if is_direct_query or is_skip_request or slot_ask_count >= max_slot_asks:
            # 跳过槽位填充，直接进入 RAG 回答
            memory['need_slot'] = None
            memory['_slot_ask_count'] = 0
            if is_skip_request:
                # 如果用户要求跳过，尝试用对话历史中的原始问题来检索
                original_question = None
                for turn in reversed(memory.get('dialog_history', [])):
                    q = turn.get('user_input', '')
                    if q and not self._is_skip_slot_request(q) and len(q) > 2:
                        original_question = q
                        break
                if original_question:
                    memory['_rag_query_override'] = original_question
        elif memory.get('need_slot'):
            # 正常槽位填充流程
            slot = memory['need_slot']
            memory['_slot_ask_count'] = slot_ask_count + 1
            if slot in self.slot_info:
                return self.slot_info[slot][0]
            else:
                return f"请提供您的{slot.replace('#', '')}信息。"
        
        # 获取法律类型
        legal_type = memory.get('#法律类型#')
        if legal_type and isinstance(legal_type, list):
            legal_type = legal_type[0] if legal_type else None
        elif legal_type and isinstance(legal_type, str) and legal_type.startswith('#'):
            legal_type = None
        
        valid_legal_types = ['劳动', '婚姻', '交通', '房产', '知识产权', '刑事', '行政', '合同']
        if legal_type not in valid_legal_types:
            legal_type = None
        
        user_input = memory.get('user_input', '')
        dialog_history = memory.get('dialog_history', [])
        
        # ========== RAG 增强路径 ==========
        if self.use_rag and self.rag_retriever and self.rag_retriever.is_ready:
            # 1. 构建检索查询（结合用户问题和法律类型）
            # 如果有查询覆盖（用户跳过槽位时，使用原始问题），优先使用
            search_query = memory.pop('_rag_query_override', None) or user_input
            if legal_type:
                search_query = f"{legal_type} {user_input}"
            
            # 2. 从向量知识库中检索相关法律文档
            retrieved_docs = self.rag_retriever.retrieve(
                query=search_query,
                k=config.RAG_TOP_K,
            )
            
            # 3. 构建 RAG 增强 Prompt
            prompt = self.rag_retriever.build_rag_prompt(
                user_input=user_input,
                retrieved_docs=retrieved_docs,
                legal_type=legal_type,
                dialog_history=dialog_history[-3:] if dialog_history else None,
            )
            
            # 4. 调用大模型 API 生成回答
            if self.use_qwen:
                response = self.call_qwen(prompt)
                # 保存检索到的文档引用到 memory 中
                memory['rag_sources'] = [
                    {
                        'domain': doc['metadata'].get('domain', '未知'),
                        'score': doc['score'],
                        'snippet': doc['content'][:100],
                    }
                    for doc in retrieved_docs[:3]
                ]
                return response
        
        # ========== 降级路径（无 RAG 时使用原有逻辑）==========
        user_info = self._build_user_info_summary(memory, legal_type)
        
        prompt = (
            f"你是一名专业的法律智能助手，请根据以下用户信息，为用户提供准确、专业的法律建议：\n\n"
            f"{user_info}\n\n"
            f"请严格按照以下要求生成回答：\n"
            f"1. 回答必须基于中国现行法律法规\n"
            f"2. 保持专业、简洁、准确\n"
            f"3. 使用中文回答\n"
            f"4. 结尾附上免责声明\n"
        )
        
        if self.use_qwen:
            return self.call_qwen(prompt)
        else:
            return f"根据您提供的信息，建议您咨询专业律师，了解相关法律法规，维护自己的合法权益。"
    
    def _build_user_info_summary(self, memory, legal_type):
        """构建用户信息摘要（用于降级路径的 Prompt）"""
        user_info = ""
        if legal_type:
            user_info = f"用户咨询{legal_type}相关法律问题。"
            
            slot_map = {
                '劳动': [('#劳动问题类型#', '问题'), ('#用人单位#', '单位'), ('#工作时长#', '工作年限')],
                '婚姻': [('#婚姻问题类型#', '问题'), ('#婚姻时长#', '结婚年限')],
                '交通': [('#事故类型#', '事故类型'), ('#责任方#', '责任方')],
                '房产': [('#房产问题类型#', '问题'), ('#房屋位置#', '位置')],
                '知识产权': [('#知识产权类型#', '问题')],
                '刑事': [('#刑事罪名#', '罪名')],
                '行政': [('#行政案件类型#', '问题')],
                '合同': [('#合同类型#', '合同类型'), ('#合同标的#', '标的')],
            }
            
            for slot_key, label in slot_map.get(legal_type, []):
                val = memory.get(slot_key, '')
                if val and not isinstance(val, list) and not str(val).startswith('#'):
                    user_info += f"{label}是{val}。"
        
        user_input = memory.get('user_input', '')
        user_info += f"用户最新问题是：{user_input}"
        return user_info
    
    def process_input(self, user_input, memory=None):
        """
        处理用户输入，生成系统响应
        
        Args:
            user_input (str): 用户输入
            memory (dict, optional): 对话记忆，默认为None
            
        Returns:
            tuple: (系统响应, 更新后的对话记忆)
        """
        try:
            # 初始化对话记忆
            if memory is None:
                memory = {
                    'user_input': user_input,
                    'avaiable_nodes': list(self.node_id2node_info.keys())
                }
            else:
                memory['user_input'] = user_input
            
            # 检查是否是明确的新问题
            switch_keywords = ["另一个", "换个", "其他", "别的", "新的", "不同的", "其他问题", "换个话题", "换个问题"]
            is_new_question = any(keyword in user_input for keyword in switch_keywords)
            
            # 检查是否是初始问候
            greeting_keywords = ["你好", "您好", "hi", "hello", "嗨", "您好", "早安", "午安", "晚安"]
            is_greeting = any(greeting in user_input for greeting in greeting_keywords)
            
            # 检查系统是否正在等待用户填充槽位
            is_waiting_slot = bool(memory.get('need_slot'))
            
            # 检查是否是非法律问题表达
            # 但如果系统正在等待用户填充槽位（need_slot不为空），则不进行非法律问题检查
            is_non_legal = False
            if not memory.get('need_slot'):
                non_legal_expressions = ["我不好", "我很好", "谢谢", "不客气", "好的", "知道了", "再见", "拜拜", "nice", "good", "ok", "yes", "no", "什么意思", "我不想问你", "弄不好", "不想", "不要"]
                # 注意：不使用"不是"作为非法律问题标记，因为它可能出现在法律问题中（如"我被偷了"）
                # 注意：不单独使用"不"作为非法律问题标记，因为它可能出现在法律问题中（如"感情不和"）
                is_non_legal = any(expr in user_input for expr in non_legal_expressions)
            
            # 如果是问候语，直接处理，不进行意图识别
            if is_greeting:
                memory['hit_intent'] = None
                memory['hit_intent_score'] = 0
                memory['is_legal_question'] = False
                memory['legal_field'] = '其他类型'
                memory['need_slot'] = False
                # 生成问候语回应
                return "您好！我是法律智能聊天机器人，很高兴为您提供法律问题咨询服务。您可以咨询劳动纠纷、婚姻家庭、交通事故、房产纠纷等法律问题。", memory
            
            # 如果是非法律问题，直接返回抱歉
            if is_non_legal:
                memory['hit_intent'] = None
                memory['hit_intent_score'] = 0
                memory['is_legal_question'] = False
                memory['legal_field'] = '其他类型'
                memory['need_slot'] = False
                return "抱歉，我无法回答这个问题，我主要专注于法律相关问题的咨询。", memory
            
            # 保存当前状态
            current_state = {
                'intent': memory.get('hit_intent'),
                'need_slot': memory.get('need_slot'),
                'legal_type': memory.get('#法律类型#')
            }
            
            # 自然语言理解
            memory = self.nlu(memory)
            
            # 如果系统正在等待用户填充槽位，确保is_legal_question为True
            if is_waiting_slot:
                memory['is_legal_question'] = True
                memory['legal_field'] = current_state['legal_type'] if current_state['legal_type'] else '劳动'
            
            # 检查是否切换了法律类型
            new_legal_type = memory.get('#法律类型#')
            has_legal_type_change = False
            
            if current_state['legal_type'] and new_legal_type and current_state['legal_type'] != new_legal_type:
                has_legal_type_change = True
            
            # 检查用户是否明确表示想要切换到不同法律类型
            # 定义法律类型关键词
            legal_type_keywords = {
                '劳动': ['劳动', '辞退', '解雇', '开除', '工资', '加班', '合同', '离职', '社保', '工伤'],
                '婚姻': ['婚姻', '离婚', '结婚', '子女', '继承', '家暴', '想离婚'],
                '交通': ['交通', '车祸', '事故', '碰撞', '追尾', '肇事'],
                '房产': ['房产', '买房', '卖房', '租房', '产权', '拆迁', '装修'],
                '知识产权': ['知识产权', '专利', '商标', '著作权', '版权'],
                '刑事': ['刑事', '犯罪', '罪名', '盗窃', '故意伤害', '被打', '打人', '打架', '伤害'],
                '行政': ['行政', '诉讼', '复议', '处罚'],
                '合同': ['合同', '违约', '买卖', '租赁', '借款']
            }
            
            # 检查用户输入中是否包含其他法律类型的关键词
            detected_legal_type = None
            for legal_type, keywords in legal_type_keywords.items():
                for keyword in keywords:
                    if keyword in user_input:
                        detected_legal_type = legal_type
                        break
                if detected_legal_type:
                    break
            
            # 如果检测到不同的法律类型，重置对话状态
            if detected_legal_type and current_state['legal_type'] and detected_legal_type != current_state['legal_type']:
                has_legal_type_change = True
            
            # 如果是新问题、问候、法律类型变化或检测到不同法律类型，重置对话状态
            if is_new_question or is_greeting or has_legal_type_change:
                # 清空已填充的槽位
                for slot in list(memory.keys()):
                    if slot.startswith('#'):
                        del memory[slot]
                # 重置意图和需要填充的槽位
                memory['hit_intent'] = None
                memory['need_slot'] = None
                
                # 重新进行自然语言理解，获取新的法律类型
                memory = self.nlu(memory)
            
            # 意图稳定性增强：如果当前有意图且不是新问题，直接恢复当前意图
            if current_state['intent'] and not is_new_question and not is_greeting and not has_legal_type_change:
                # 直接恢复当前意图
                memory['hit_intent'] = current_state['intent']
                memory['hit_intent_score'] = 1.0  # 设置高置信度
            # 问候语处理
            if is_greeting:
                # 查找正确的初始意图节点ID
                initial_node_id = None
                for node_id, node in self.node_id2node_info.items():
                    if 'node1' in node_id:  # 查找包含'node1'的节点ID
                        initial_node_id = node_id
                        break
                
                if initial_node_id:
                    # 强制设置为初始意图
                    memory['hit_intent'] = initial_node_id
                    memory['hit_intent_score'] = 0.9  # 设置高置信度
                    # 确保生成问候响应
                    memory['need_slot'] = None
                    # 清空所有槽位
                    for slot in list(memory.keys()):
                        if slot.startswith('#'):
                            del memory[slot]
            
            # 法律类型稳定性：如果当前有法律类型，保持当前法律类型
            if current_state['legal_type'] and not is_new_question and not is_greeting and not has_legal_type_change:
                memory['#法律类型#'] = current_state['legal_type']
            
            # 对话状态跟踪
            memory = self.state_tracker.dst(memory)
            
            # 检测对话错误
            errors = self.state_tracker.detect_dialog_errors(memory)
            if errors:
                # 简单处理：如果有错误，重新生成响应
                print(f"检测到对话错误: {errors}")
            
            # 生成响应
            response = self.generate_response(memory)
            
            # 更新对话历史
            if 'dialog_history' in memory and memory['dialog_history']:
                memory['dialog_history'][-1]['system_response'] = response
            
            return response, memory
        except Exception as e:
            import traceback
            error_info = f"处理用户输入失败: {e}\n{traceback.format_exc()}"
            print(error_info)
            # 返回默认响应和记忆
            default_memory = {
                'user_input': user_input,
                'hit_intent': None,
                'hit_intent_score': 0,
                'is_legal_question': True,
                'legal_field': '刑事',
                'sentiment': 'neutral',
                'sentiment_score': 0.5,
                'entities': {},
                'need_slot': None,
                'dialog_history': [],
                'dialog_state': {
                    'intent': None,
                    'filled_slots': [],
                    'missing_slots': [],
                    'optional_slots': [],
                    'filled_optional_slots': [],
                    'missing_optional_slots': [],
                    'turn_count': 0,
                    'last_updated': 0,
                    'strategy': 'professional',
                    'legal_type': '刑事'
                }
            }
            return "根据您提供的信息，您的情况可能涉及刑事法律问题，建议您咨询专业律师，了解相关法律法规，维护自己的合法权益。", default_memory

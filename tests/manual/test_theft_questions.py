#!/usr/bin/env python3
# 测试盗窃相关问题的识别

from src.legacy.dialog import DialogManager

# 创建对话管理器
dm = DialogManager(use_qwen=False, use_bert_intent=False)

# 测试用例
test_cases = [
    '我家被偷了',
    '我的钱包被偷了',
    '有人偷了我的自行车',
    '我被小偷偷了'
]

# 执行测试
for test_case in test_cases:
    print(f"\n用户输入: {test_case}")
    response, memory = dm.process_input(test_case)
    print(f"系统响应: {response}")
    print(f"is_legal_question: {memory.get('is_legal_question')}")
    print(f"legal_field: {memory.get('legal_field')}")

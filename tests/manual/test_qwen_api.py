#!/usr/bin/env python3
# 测试Qwen API调用

from src.legacy.dialog import DialogManager

# 创建对话管理器，显式设置use_qwen=True
dm = DialogManager(use_qwen=True, use_bert_intent=False)

# 测试call_qwen方法
try:
    prompt = "你好，我是一名法律智能助手，请根据用户咨询的劳动相关法律问题，提供准确、专业的法律建议：用户咨询劳动相关法律问题。具体问题是辞退。用户在华南公司工作。工作年限为三年。用户最新问题是：三年"
    print("调用Qwen API...")
    response = dm.call_qwen(prompt)
    print(f"API响应: {response}")
except Exception as e:
    print(f"调用失败: {e}")

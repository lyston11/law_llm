"""旧版规则对话系统（v1.0 / v2.0）

本目录包含项目早期基于规则的对话管理系统，已被 Agent 架构（v3.0）替代。
保留这些代码用于：
  1. 论文中的对比实验（规则系统 vs Agent 系统）
  2. 技术演进的参考

模块说明：
  - dialog.py       旧版对话管理器（规则流水线）
  - intent.py       TF-IDF 意图识别
  - intent_bert.py   BERT 意图识别
  - tfidf.py        TF-IDF 计算器
  - slot.py         槽位填充
  - state.py        对话状态跟踪
"""

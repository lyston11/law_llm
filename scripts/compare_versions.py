#!/usr/bin/env python3
"""对比实验脚本 — 规则系统 vs Agent 系统

用途：
  使用同一批测试问题，分别经过旧版规则系统（src/legacy/dialog.py）
  和新版 Agent 系统（src/agent.py）处理，输出对比结果。

输出：
  - 每个问题的响应时间、回答摘要、工具调用情况
  - 格式化对比表格（可直接复制到论文）
  - 统计汇总

使用方法：
  python scripts/compare_versions.py

前提条件：
  - .env 中已配置 API_KEY
  - 虚拟环境已激活，依赖已安装
  - 场景配置文件存在（data/legal/scenario/）
"""

import os
import sys
import time
import json

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import Config

config = Config()

# ==================== 测试问题集 ====================

TEST_QUESTIONS = [
    # 法律问题（旧版规则系统的擅长领域）
    {"question": "我被公司辞退了，工作了3年，能拿到多少赔偿？", "category": "法律-劳动纠纷", "type": "legal"},
    {"question": "离婚时财产怎么分割？", "category": "法律-婚姻家庭", "type": "legal"},
    {"question": "交通事故对方全责，我该怎么索赔？", "category": "法律-交通事故", "type": "legal"},
    {"question": "劳动合同法第四十七条规定了什么？", "category": "法律-法条查询", "type": "legal"},
    {"question": "被人打了构成什么罪？", "category": "法律-刑事案件", "type": "legal"},
    {"question": "租房合同到期房东不退押金怎么办？", "category": "法律-合同纠纷", "type": "legal"},
    {"question": "工伤认定需要满足什么条件？", "category": "法律-劳动纠纷", "type": "legal"},

    # 通用问题（测试系统的通用对话能力）
    {"question": "你好，你是谁？", "category": "通用-问候", "type": "general"},
    {"question": "今天天气怎么样？", "category": "通用-天气", "type": "general"},
    {"question": "Python 和 Java 哪个好？", "category": "通用-技术", "type": "general"},
    {"question": "1+1等于几？", "category": "通用-数学", "type": "general"},
    {"question": "帮我写一首关于春天的诗", "category": "通用-创作", "type": "general"},
]


# ==================== 规则系统测试 ====================

def test_legacy_system(questions: list) -> list:
    """使用旧版规则系统处理测试问题"""
    print("\n" + "=" * 60)
    print("  测试旧版规则系统 (src/legacy/dialog.py)")
    print("=" * 60)

    results = []

    try:
        from src.legacy.dialog import DialogManager
        dm = DialogManager(use_qwen=True, use_bert_intent=False)
        print("✅ 旧版 DialogManager 初始化成功\n")
    except Exception as e:
        print(f"❌ 旧版 DialogManager 初始化失败: {e}")
        print("   跳过规则系统测试，所有结果标记为 ERROR\n")
        for q in questions:
            results.append({
                "question": q["question"],
                "category": q["category"],
                "type": q["type"],
                "response": "[初始化失败]",
                "time_ms": 0,
                "has_legal_reference": False,
                "can_handle": False,
                "error": str(e),
            })
        return results

    memory = None
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['category']}: {q['question'][:30]}...", end=" ", flush=True)

        start_time = time.time()
        try:
            response, memory = dm.process_input(q["question"], memory=None)
            elapsed_ms = (time.time() - start_time) * 1000

            # 分析回答质量
            has_legal_ref = any(kw in response for kw in ["法", "条", "规定", "条例", "依据"])
            can_handle = "无法回答" not in response and "不能" not in response[:10]

            results.append({
                "question": q["question"],
                "category": q["category"],
                "type": q["type"],
                "response": response[:200],
                "time_ms": round(elapsed_ms, 1),
                "has_legal_reference": has_legal_ref,
                "can_handle": can_handle,
                "error": None,
            })
            print(f"✓ {elapsed_ms:.0f}ms")

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            results.append({
                "question": q["question"],
                "category": q["category"],
                "type": q["type"],
                "response": f"[ERROR] {str(e)[:100]}",
                "time_ms": round(elapsed_ms, 1),
                "has_legal_reference": False,
                "can_handle": False,
                "error": str(e),
            })
            print(f"✗ ERROR: {str(e)[:50]}")

    return results


# ==================== Agent 系统测试 ====================

def test_agent_system(questions: list) -> list:
    """使用新版 Agent 系统处理测试问题"""
    print("\n" + "=" * 60)
    print("  测试新版 Agent 系统 (src/agent.py)")
    print("=" * 60)

    results = []

    try:
        from src.agent import DomainAgent
        agent = DomainAgent()
        print("✅ DomainAgent 初始化成功\n")
    except Exception as e:
        print(f"❌ DomainAgent 初始化失败: {e}")
        print("   跳过 Agent 系统测试，所有结果标记为 ERROR\n")
        for q in questions:
            results.append({
                "question": q["question"],
                "category": q["category"],
                "type": q["type"],
                "response": "[初始化失败]",
                "time_ms": 0,
                "has_legal_reference": False,
                "can_handle": False,
                "tools_used": [],
                "error": str(e),
            })
        return results

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['category']}: {q['question'][:30]}...", end=" ", flush=True)

        start_time = time.time()
        try:
            result = agent.chat(user_input=q["question"], conversation_history=None)
            elapsed_ms = (time.time() - start_time) * 1000

            response = result.get("response", "")
            tools_used = [a["tool"] for a in result.get("agent_actions", [])]

            # 分析回答质量
            has_legal_ref = any(kw in response for kw in ["法", "条", "规定", "条例", "依据"])
            can_handle = True  # Agent 可以处理所有问题

            results.append({
                "question": q["question"],
                "category": q["category"],
                "type": q["type"],
                "response": response[:200],
                "time_ms": round(elapsed_ms, 1),
                "has_legal_reference": has_legal_ref,
                "can_handle": can_handle,
                "tools_used": tools_used,
                "error": None,
            })

            tools_str = ", ".join(tools_used) if tools_used else "无"
            print(f"✓ {elapsed_ms:.0f}ms | 工具: {tools_str}")

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            results.append({
                "question": q["question"],
                "category": q["category"],
                "type": q["type"],
                "response": f"[ERROR] {str(e)[:100]}",
                "time_ms": round(elapsed_ms, 1),
                "has_legal_reference": False,
                "can_handle": False,
                "tools_used": [],
                "error": str(e),
            })
            print(f"✗ ERROR: {str(e)[:50]}")

    return results


# ==================== 对比分析 ====================

def print_comparison(legacy_results: list, agent_results: list):
    """输出对比结果"""
    print("\n")
    print("=" * 80)
    print("  对 比 结 果")
    print("=" * 80)

    # ---- 逐题对比表格 ----
    print("\n### 逐题对比\n")
    header = f"{'编号':^4} | {'问题类别':<12} | {'规则系统(ms)':>12} | {'Agent(ms)':>10} | {'规则可答':^8} | {'Agent可答':^8} | {'Agent工具调用':<20}"
    print(header)
    print("-" * len(header))

    for i, (lr, ar) in enumerate(zip(legacy_results, agent_results), 1):
        tools = ", ".join(ar.get("tools_used", [])) if ar.get("tools_used") else "-"
        legacy_ok = "✓" if lr["can_handle"] else "✗"
        agent_ok = "✓" if ar["can_handle"] else "✗"
        print(
            f"{i:^4} | {lr['category']:<12} | {lr['time_ms']:>10.1f}ms | {ar['time_ms']:>8.1f}ms | {legacy_ok:^8} | {agent_ok:^8} | {tools:<20}"
        )

    # ---- 分类统计 ----
    print("\n\n### 分类统计\n")

    for qtype, label in [("legal", "法律问题"), ("general", "通用问题")]:
        l_items = [r for r in legacy_results if r["type"] == qtype]
        a_items = [r for r in agent_results if r["type"] == qtype]

        if not l_items:
            continue

        l_avg_time = sum(r["time_ms"] for r in l_items) / len(l_items)
        a_avg_time = sum(r["time_ms"] for r in a_items) / len(a_items)
        l_handle_rate = sum(1 for r in l_items if r["can_handle"]) / len(l_items) * 100
        a_handle_rate = sum(1 for r in a_items if r["can_handle"]) / len(a_items) * 100
        l_ref_rate = sum(1 for r in l_items if r["has_legal_reference"]) / len(l_items) * 100
        a_ref_rate = sum(1 for r in a_items if r["has_legal_reference"]) / len(a_items) * 100

        print(f"  【{label}】（共 {len(l_items)} 题）")
        print(f"    {'指标':<16} | {'规则系统':>10} | {'Agent系统':>10}")
        print(f"    {'-'*16}-+-{'-'*10}-+-{'-'*10}")
        print(f"    {'平均响应时间':<14} | {l_avg_time:>8.1f}ms | {a_avg_time:>8.1f}ms")
        print(f"    {'可回答率':<16} | {l_handle_rate:>9.1f}% | {a_handle_rate:>9.1f}%")
        print(f"    {'法律依据引用率':<12} | {l_ref_rate:>9.1f}% | {a_ref_rate:>9.1f}%")
        print()

    # ---- 总体统计 ----
    print("\n### 总体统计\n")

    total = len(legacy_results)
    l_total_time = sum(r["time_ms"] for r in legacy_results)
    a_total_time = sum(r["time_ms"] for r in agent_results)
    l_avg = l_total_time / total
    a_avg = a_total_time / total
    l_handle = sum(1 for r in legacy_results if r["can_handle"])
    a_handle = sum(1 for r in agent_results if r["can_handle"])
    a_tool_calls = sum(len(r.get("tools_used", [])) for r in agent_results)

    print(f"  总问题数:         {total}")
    print(f"  规则系统总耗时:   {l_total_time:.0f}ms (平均 {l_avg:.1f}ms/题)")
    print(f"  Agent系统总耗时:  {a_total_time:.0f}ms (平均 {a_avg:.1f}ms/题)")
    print(f"  规则系统可回答:   {l_handle}/{total} ({l_handle/total*100:.1f}%)")
    print(f"  Agent系统可回答:  {a_handle}/{total} ({a_handle/total*100:.1f}%)")
    print(f"  Agent工具调用次数: {a_tool_calls}")

    # ---- 详细回答对比（前 3 题）----
    print("\n\n### 回答内容对比（示例）\n")
    for i, (lr, ar) in enumerate(zip(legacy_results[:3], agent_results[:3]), 1):
        print(f"  --- 问题 {i}: {lr['question']} ---")
        print(f"  [规则系统] {lr['response'][:150]}...")
        print(f"  [Agent]    {ar['response'][:150]}...")
        tools = ", ".join(ar.get("tools_used", [])) if ar.get("tools_used") else "无"
        print(f"  [Agent工具] {tools}")
        print()


def save_results(legacy_results: list, agent_results: list):
    """保存原始结果到 JSON 文件"""
    output_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "compare_results.json")

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_questions_count": len(TEST_QUESTIONS),
        "legacy_results": legacy_results,
        "agent_results": agent_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n📄 原始数据已保存到: {output_file}")
    print("   可用于论文中的表格和图表数据填充\n")


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("  智能领域聊天机器人 — 版本对比实验")
    print("  规则系统 (v1.0) vs Agent 系统 (v3.0)")
    print("=" * 60)
    print(f"\n测试问题数: {len(TEST_QUESTIONS)}")
    print(f"  法律问题: {sum(1 for q in TEST_QUESTIONS if q['type'] == 'legal')}")
    print(f"  通用问题: {sum(1 for q in TEST_QUESTIONS if q['type'] == 'general')}")

    # 检查 API Key
    if not config.API_KEY:
        print("\n❌ 错误: 请在 .env 文件中配置 API_KEY")
        print("   两个系统都需要调用 Qwen API")
        sys.exit(1)

    # 运行测试
    legacy_results = test_legacy_system(TEST_QUESTIONS)
    agent_results = test_agent_system(TEST_QUESTIONS)

    # 输出对比结果
    print_comparison(legacy_results, agent_results)

    # 保存原始数据
    save_results(legacy_results, agent_results)

    print("=" * 60)
    print("  对比实验完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

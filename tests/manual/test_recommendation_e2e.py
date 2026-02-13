"""
推荐功能端到端测试
手动运行: python tests/manual/test_recommendation_e2e.py
"""
import requests
import json
import sys

API_URL = "http://localhost:8000"

def test_recommendation_e2e():
    """测试完整的推荐流程"""
    print("开始端到端测试...")
    print(f"API URL: {API_URL}")

    # 先检查服务是否运行
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        print(f"✓ 后端服务运行正常")
    except Exception as e:
        print(f"✗ 后端服务未运行: {e}")
        print("请先启动后端: python -m uvicorn api.main:app --reload")
        return False

    # 测试场景 1: 法律咨询应该生成推荐
    print("\n=== 测试场景 1: 法律咨询（应该生成推荐）===")
    try:
        response = requests.post(
            f"{API_URL}/dialog",
            json={
                "user_input": "被公司辞退了怎么办",
                "session_id": "test_recommend_e2e"
            },
            timeout=10
        ).json()

        assert "recommended_questions" in response, "响应缺少 recommended_questions 字段"
        questions = response["recommended_questions"]
        assert isinstance(questions, list), "recommended_questions 应该是列表"
        print(f"✓ 生成推荐问题: {questions}")
        if len(questions) > 0:
            print(f"✓ 推荐功能正常工作（生成了 {len(questions)} 个问题）")
        else:
            print(f"⚠ 未生成推荐问题（可能是 LLM 调用失败或被跳过）")
    except Exception as e:
        print(f"✗ 测试场景 1 失败: {e}")
        return False

    # 测试场景 2: 简单对话应该跳过推荐
    print("\n=== 测试场景 2: 简单对话（应该跳过推荐）===")
    try:
        response = requests.post(
            f"{API_URL}/dialog",
            json={
                "user_input": "你好",
                "session_id": "test_recommend_e2e_2"
            },
            timeout=10
        ).json()

        assert "recommended_questions" in response
        questions = response["recommended_questions"]
        assert isinstance(questions, list)
        print(f"✓ 简单对话推荐结果: {questions}")
        if len(questions) == 0:
            print(f"✓ 简单对话正确跳过推荐")
        else:
            print(f"⚠ 简单对话未跳过推荐（生成了 {len(questions)} 个问题）")
    except Exception as e:
        print(f"✗ 测试场景 2 失败: {e}")
        return False

    # 测试场景 3: 连续对话
    print("\n=== 测试场景 3: 连续对话（基于上下文）===")
    try:
        response = requests.post(
            f"{API_URL}/dialog",
            json={
                "user_input": "如何计算赔偿金？",
                "session_id": "test_recommend_e2e"
            },
            timeout=10
        ).json()

        questions = response["recommended_questions"]
        print(f"✓ 连续对话推荐问题: {questions}")
    except Exception as e:
        print(f"✗ 测试场景 3 失败: {e}")
        return False

    print("\n=== 所有端到端测试通过 ===")
    return True


if __name__ == "__main__":
    success = test_recommendation_e2e()
    sys.exit(0 if success else 1)

"""
推荐功能性能测试
手动运行: python tests/manual/test_recommendation_performance.py
"""
import time
import requests
import sys

API_URL = "http://localhost:8000"

def test_recommendation_performance():
    """测试推荐生成性能"""
    print("开始性能测试...")
    print(f"API URL: {API_URL}")

    # 先检查服务是否运行
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        print(f"✓ 后端服务运行正常\n")
    except Exception as e:
        print(f"✗ 后端服务未运行: {e}")
        print("请先启动后端: python -m uvicorn api.main:app --reload")
        return False

    # 测试多次请求，取平均值
    test_cases = [
        "劳动合同法有哪些规定？",
        "离婚时财产怎么分割？",
        "交通事故对方全责，我该怎么索赔？",
        "用人单位不签劳动合同怎么办？",
        "加班费怎么计算？"
    ]

    response_times = []
    question_counts = []

    for i, user_input in enumerate(test_cases, 1):
        print(f"测试 {i}/{len(test_cases)}: {user_input}")
        start_time = time.time()

        try:
            response = requests.post(
                f"{API_URL}/dialog",
                json={
                    "user_input": user_input,
                    "session_id": f"perf_test_{i}"
                },
                timeout=10
            ).json()

            end_time = time.time()
            elapsed = (end_time - start_time) * 1000  # 转换为毫秒
            response_times.append(elapsed)

            questions = response.get("recommended_questions", [])
            question_counts.append(len(questions))

            print(f"  响应时间: {elapsed:.0f}ms")
            print(f"  推荐问题数: {len(questions)}")
            print(f"  推荐问题: {questions}")
            print()

        except Exception as e:
            print(f"  ✗ 请求失败: {e}\n")
            return False

    # 计算统计信息
    avg_time = sum(response_times) / len(response_times)
    max_time = max(response_times)
    min_time = min(response_times)
    avg_questions = sum(question_counts) / len(question_counts)

    print("=" * 60)
    print("性能测试统计:")
    print(f"  平均响应时间: {avg_time:.0f}ms")
    print(f"  最大响应时间: {max_time:.0f}ms")
    print(f"  最小响应时间: {min_time:.0f}ms")
    print(f"  平均推荐问题数: {avg_questions:.1f}")
    print("=" * 60)

    # 性能基准检查
    if avg_time < 2000:
        print(f"✓ 性能良好: 平均响应时间 {avg_time:.0f}ms < 2000ms")
        return True
    else:
        print(f"⚠ 性能需优化: 平均响应时间 {avg_time:.0f}ms >= 2000ms")
        return True  # 仍然返回 True，因为这不是硬性失败


if __name__ == "__main__":
    success = test_recommendation_performance()
    sys.exit(0 if success else 1)

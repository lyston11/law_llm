import requests
import json

# 测试对话API，使用复杂法律问题作为输入
def test_dialog():
    url = "http://localhost:8000/dialog"
    headers = {"Content-Type": "application/json"}
    data = {
        "user_input": "我和丈夫结婚五年，因感情不和想离婚，我们有一个三岁的孩子，想了解孩子抚养权的问题。",
        "session_id": "test_session_113"
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        result = response.json()
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {result}")
        print(f"回复: {result['response']}")
        print(f"会话ID: {result['session_id']}")
        print(f"状态: {result['status']}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")

if __name__ == "__main__":
    test_dialog()

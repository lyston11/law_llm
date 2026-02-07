import requests
import json

# 测试连续对话，模拟用户与系统的多轮交互
def test_conversation():
    url = "http://localhost:8000/dialog"
    headers = {"Content-Type": "application/json"}
    session_id = "test_session_114"
    
    # 第一轮：问候
    user_input1 = "你好"
    print(f"用户: {user_input1}")
    
    data1 = {
        "user_input": user_input1,
        "session_id": session_id
    }
    
    response1 = requests.post(url, headers=headers, data=json.dumps(data1))
    if response1.status_code == 200:
        result1 = response1.json()
        print(f"系统: {result1['response']}")
    else:
        print(f"请求失败，状态码: {response1.status_code}")
        return
    
    # 第二轮：提出法律问题
    user_input2 = "我被无故辞退了"
    print(f"\n用户: {user_input2}")
    
    data2 = {
        "user_input": user_input2,
        "session_id": session_id
    }
    
    response2 = requests.post(url, headers=headers, data=json.dumps(data2))
    if response2.status_code == 200:
        result2 = response2.json()
        print(f"系统: {result2['response']}")
    else:
        print(f"请求失败，状态码: {response2.status_code}")
        return
    
    # 第三轮：回答系统追问
    user_input3 = "华南公司"
    print(f"\n用户: {user_input3}")
    
    data3 = {
        "user_input": user_input3,
        "session_id": session_id
    }
    
    response3 = requests.post(url, headers=headers, data=json.dumps(data3))
    if response3.status_code == 200:
        result3 = response3.json()
        print(f"系统: {result3['response']}")
    else:
        print(f"请求失败，状态码: {response3.status_code}")
        return
    
    print(f"\n对话状态: {result3['status']}")
    print(f"会话ID: {result3['session_id']}")

if __name__ == "__main__":
    test_conversation()

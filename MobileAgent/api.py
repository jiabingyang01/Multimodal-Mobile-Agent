import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def inference_chat(chat, model, api_url, token):    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': 0.0,
        "seed": 1234
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    res = requests.post(api_url, headers=headers, json=data)
    # 检查请求是否成功
    if res.status_code == 200:
        res_json = res.json()
        res_content = res_json['choices'][0]['message']['content']
    else:
        print("Error:", res.status_code, res.text)

    # while True:
    #     try:
    #         res = requests.post(api_url, headers=headers, json=data)
    #         res_json = res.json()
    #         res_content = res_json['choices'][0]['message']['content']
    #     except:
    #         print("Network Error:")
    #         try:
    #             print(res.json())
    #         except:
    #             print("Request Failed")
    #     else:
    #         break
    
    return res_content

# - *- coding: utf-8 -*-
import requests
request_url = 'http://192.168.1.64:80/ISAPI/System/deviceInfo'
# 设置认证信息
auth = requests.auth.HTTPDigestAuth('admin', 'gene2025')
# 发送请求，获取响应
response = requests.get(request_url, auth=auth)
# 输出响应内容
print(response.text)

response = requests.get("http://192.168.1.64/ISAPI/System/capabilities?type=all", auth=auth)
# 输出响应内容
print(response.text)

response = requests.get("http://192.168.1.64/ISAPI/PTZCtrl/channels/1/capabilities", auth=auth)
# 输出响应内容
print(response.text)


response = requests.get("http://192.168.1.64/ISAPI/PTZCtrl/channels/1/presets", auth=auth)
# 输出响应内容
print(response.text)


response = requests.get("http://192.168.1.64/ISAPI/PTZCtrl/channels/1/status", auth=auth)
# 输出响应内容
print(response.text)

response = requests.get("http://192.168.1.64/ISAPI/Streaming/channels/101/picture", auth=auth)
if response.status_code == 200:
    with open('camera_image.jpg', 'wb') as f:
        f.write(response.content)
    print("图片保存成功！")
else:
    print(f"请求失败，状态码：{response.status_code}")

response = requests.get("http://192.168.1.64/ISAPI/Image/channels/1/focusConfiguration", auth=auth)
# 输出响应内容
print(response.text)
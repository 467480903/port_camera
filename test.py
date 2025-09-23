# - *- coding: utf-8 -*-
import requests
request_url = 'http://192.168.1.64:80/ISAPI/System/deviceInfo'
# 设置认证信息
auth = requests.auth.HTTPDigestAuth('admin', 'gene2025')
# 发送请求，获取响应
response = requests.get(request_url, auth=auth)
# 输出响应内容
print(response.text)
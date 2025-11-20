import socket
import json

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 1991))
client.send(json.dumps({"command": "detect"}).encode('utf-8'))
response = client.recv(4096).decode('utf-8')
print(response)
client.close()
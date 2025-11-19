import matplotlib.pyplot as plt

# 数据
data = [
    { "distance": 2.004, "focus": 9443, "zoom": 0 },
    { "distance": 3.051, "focus": 7625, "zoom": 446 },
    { "distance": 4.083, "focus": 6519, "zoom": 680 },
    { "distance": 5.015, "focus": 5522, "zoom": 866 },
    { "distance": 6.042, "focus": 4688, "zoom": 1006 },
    { "distance": 7.072, "focus": 4028, "zoom": 1116 },
    { "distance": 8.047, "focus": 3534, "zoom": 1196 },
    { "distance": 9.051, "focus": 3144, "zoom": 1256 },
    { "distance": 10.035, "focus": 2773, "zoom": 1316 },
    { "distance": 12, "focus": 2280, "zoom": 1396 },
    { "distance": 14.054, "focus": 1818, "zoom": 1486 },
    { "distance": 18.054, "focus": 1530, "zoom": 1606 },
    { "distance": 21.046, "focus": 1712, "zoom": 1656 },
    { "distance": 23.708, "focus": 1952, "zoom": 1696 }
]

# 提取坐标
distance = [item["distance"] for item in data]
zoom = [item["zoom"] for item in data]

# 绘制图表
plt.figure(figsize=(8,5))
plt.scatter(distance, zoom)
plt.xlabel("distance")
plt.ylabel("zoom")
plt.title("Distance vs Zoom")
plt.grid(True)
plt.show()

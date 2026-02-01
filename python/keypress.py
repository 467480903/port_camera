#!/usr/bin/env python3
"""
键盘监听程序
监听键盘按键并将按键信息推送到HTTP端点
"""

import requests
from pynput import keyboard
import json
from datetime import datetime
import time

# 配置
API_URL = "http://localhost:1881/keypress/"
RETRY_DELAY = 5  # 连接失败时的重试延迟（秒）
MONITORED_KEYS = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}  # 要监听的按键

def send_keypress(key_info):
    """
    发送按键信息到API端点
    
    Args:
        key_info: 包含按键信息的字典
    """
    try:
        response = requests.post(
            API_URL,
            json=key_info,
            timeout=2
        )
        if response.status_code == 200:
            print(f"✓ 已发送: {key_info['key']}")
        else:
            print(f"✗ 发送失败 (状态码: {response.status_code}): {key_info['key']}")
    except requests.exceptions.ConnectionError:
        print(f"✗ 连接错误: 无法连接到 {API_URL}")
    except requests.exceptions.Timeout:
        print(f"✗ 超时: {API_URL}")
    except Exception as e:
        print(f"✗ 错误: {e}")

def on_press(key):
    """
    按键按下事件处理
    
    Args:
        key: 按下的按键对象
    """
    try:
        # 尝试获取字符键
        key_char = key.char
        key_name = key_char
    except AttributeError:
        # 特殊键（如Ctrl, Alt等）- 不在监听范围内，直接返回
        return
    
    # 只处理指定的按键
    if key_name not in MONITORED_KEYS:
        return
    
    # 构造按键信息
    key_info = {
        'key': key_name.upper(),  # 统一转换为大写
        'event': 'press',
        'timestamp': datetime.now().isoformat(),
        'time': time.time()
    }
    
    # 发送到API
    send_keypress(key_info)

def on_release(key):
    """
    按键释放事件处理（可选）
    
    Args:
        key: 释放的按键对象
    
    Returns:
        False表示停止监听器
    """
    # 按ESC键退出程序
    if key == keyboard.Key.esc:
        print("\n检测到ESC键，程序退出...")
        return False

def main():
    """
    主函数
    """
    print("=" * 60)
    print("键盘监听程序")
    print("=" * 60)
    print(f"API端点: {API_URL}")
    print(f"监听按键: A B C D E F G H")
    print("按 ESC 键退出程序")
    print("=" * 60)
    print()
    
    # 创建并启动键盘监听器
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    ) as listener:
        listener.join()
    
    print("\n程序已停止")

if __name__ == "__main__":
    main()
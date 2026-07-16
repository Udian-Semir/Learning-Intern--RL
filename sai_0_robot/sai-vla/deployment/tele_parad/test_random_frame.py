"""
从视频中随机挑选一帧发送给 OFT server 进行测试

使用方法:
    python test_random_frame.py --video /path/to/video.mp4
    python test_random_frame.py --video /path/to/video.mp4 --server http://localhost:8000
"""

import argparse
import random
import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


def extract_random_frame(video_path: str) -> np.ndarray:
    """
    从视频中随机提取一帧
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        BGR 格式的 numpy 数组 [H, W, C]
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        raise ValueError(f"视频帧数无效: {total_frames}")
    
    # 随机选择一帧
    random_frame_idx = random.randint(0, total_frames - 1)
    print(f"视频总帧数: {total_frames}, 随机选择帧: {random_frame_idx}")
    
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
    
    # 读取帧
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取第 {random_frame_idx} 帧")
    
    return frame, random_frame_idx


def frame_to_base64(frame: np.ndarray) -> str:
    """
    将 BGR 帧转换为 base64 编码的 JPEG
    
    Args:
        frame: BGR 格式的 numpy 数组
        
    Returns:
        base64 编码的字符串
    """
    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 转换为 PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # 编码为 JPEG
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    
    # Base64 编码
    return base64.b64encode(buffer.read()).decode('utf-8')


def send_to_server(image_base64: str, server_url: str) -> dict:
    """
    发送图像到推理服务器
    
    Args:
        image_base64: base64 编码的图像
        server_url: 服务器地址 (如 http://localhost:8000)
        
    Returns:
        服务器响应
    """
    endpoint = f"{server_url}/predict"
    
    payload = {
        "images": [image_base64],
        "state": None,  # 不发送 state
    }
    
    print(f"发送请求到: {endpoint}")
    print(f"图像 base64 长度: {len(image_base64)} 字符")
    
    response = requests.post(endpoint, json=payload, timeout=60)
    
    if response.status_code != 200:
        print(f"请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        raise RuntimeError(f"Server 返回错误: {response.status_code}")
    
    return response.json()


def main():
    parser = argparse.ArgumentParser(description='从视频随机挑选一帧发送给 OFT server')
    parser.add_argument(
        '--video', 
        type=str, 
        default='/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset256/videos/chunk-000/observation.images.main/episode_000000.mp4',
        help='视频文件路径'
    )
    parser.add_argument(
        '--server', 
        type=str, 
        default='http://localhost:8000',
        help='服务器地址'
    )
    parser.add_argument(
        '--save-frame',
        action='store_true',
        help='保存提取的帧到本地'
    )
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not Path(args.video).exists():
        print(f"错误: 视频文件不存在: {args.video}")
        return
    
    print("=" * 60)
    print(f"视频路径: {args.video}")
    print(f"服务器地址: {args.server}")
    print("=" * 60)
    
    # 1. 提取随机帧
    print("\n[1/3] 从视频提取随机帧...")
    frame, frame_idx = extract_random_frame(args.video)
    print(f"帧尺寸: {frame.shape}")
    
    # 可选: 保存帧到本地
    if args.save_frame:
        save_path = f"random_frame_{frame_idx}.jpg"
        cv2.imwrite(save_path, frame)
        print(f"帧已保存到: {save_path}")
    
    # 2. 转换为 base64
    print("\n[2/3] 转换为 base64...")
    image_base64 = frame_to_base64(frame)
    print(f"Base64 编码完成")
    
    # 3. 发送到服务器
    print("\n[3/3] 发送到服务器...")
    try:
        result = send_to_server(image_base64, args.server)
        
        print("\n" + "=" * 60)
        print("✓ 推理成功!")
        print("=" * 60)
        print(f"Chunk size: {result['chunk_size']}")
        print(f"Action dim: {result['action_dim']}")
        print(f"\n时间统计:")
        timing = result['timing']
        print(f"  VLM 推理: {timing['vlm_time']:.3f}s")
        print(f"  OFT Pipeline 推理: {timing['oft_time']:.3f}s")
        print(f"  总耗时: {timing['total_time']:.3f}s")
        
        print(f"\n预测动作 (前 5 步):")
        actions = result['actions']
        
        for i in range(min(5, len(actions))):
            action_str = ', '.join([f'{v:.4f}' for v in actions[i][:7]])
            print(f"  Step {i}: [{action_str}...]")
        
    except requests.exceptions.ConnectionError:
        print(f"\n错误: 无法连接到服务器 {args.server}")
        print("请确保服务器已启动: python server.py --config config.yaml")
    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()

"""
发送指定图片和 state 给 server 进行测试

使用方法:
    python test_image_state.py --image /path/to/image.jpg --state 0.1,0.2,0.3,...
    python test_image_state.py --image /path/to/image.jpg --state-file state.json
"""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests
from PIL import Image


def load_image_to_base64(image_path: str) -> str:
    """
    加载图片并转换为 base64
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        base64 编码的字符串
    """
    image = Image.open(image_path).convert('RGB')
    print(f"图片尺寸: {image.size[0]}x{image.size[1]}")
    
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')


def parse_state(state_values: Optional[List[str]], state_file: Optional[str]) -> Optional[List[float]]:
    """
    解析 state
    
    Args:
        state_values: state 值列表（支持逗号/空格分隔）
        state_file: state JSON 文件路径
        
    Returns:
        state 列表或 None
    """
    if state_file and Path(state_file).exists():
        with open(state_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [float(v) for v in data]
            elif isinstance(data, dict) and 'state' in data:
                return [float(v) for v in data['state']]
            else:
                raise ValueError(f"无法从 JSON 文件解析 state: {state_file}")
    
    if state_values:
        # 合并所有参数，支持空格和逗号混合分隔
        combined = ' '.join(state_values)
        # 将逗号替换为空格，然后按空格分割
        combined = combined.replace(',', ' ')
        # 过滤空字符串并转换为 float
        values = [float(v.strip()) for v in combined.split() if v.strip()]
        if values:
            return values
    
    return None


def send_to_server(
    image_base64: str, 
    state: Optional[List[float]], 
    server_url: str
) -> dict:
    """
    发送图像和 state 到推理服务器
    
    Args:
        image_base64: base64 编码的图像
        state: state 列表
        server_url: 服务器地址
        
    Returns:
        服务器响应
    """
    endpoint = f"{server_url}/predict"
    
    payload = {
        "images": [image_base64],
        "state": state,
    }
    
    print(f"\n发送请求到: {endpoint}")
    print(f"图像 base64 长度: {len(image_base64)} 字符")
    if state:
        print(f"State 维度: {len(state)}")
        print(f"State 值: {state}")
    else:
        print("State: None")
    
    response = requests.post(endpoint, json=payload, timeout=60)
    
    if response.status_code != 200:
        print(f"请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        raise RuntimeError(f"Server 返回错误: {response.status_code}")
    
    return response.json()


def main():
    parser = argparse.ArgumentParser(description='发送图片和 state 给 server')
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='图片文件路径'
    )
    parser.add_argument(
        '--state', 
        type=str, 
        nargs='*',
        default=None,
        help='state 值，支持逗号/空格分隔，例如: 0.1,0.2,0.3 或 0.1 0.2 0.3'
    )
    parser.add_argument(
        '--state-file', 
        type=str, 
        default=None,
        help='state JSON 文件路径'
    )
    parser.add_argument(
        '--server', 
        type=str, 
        default='http://localhost:8000',
        help='服务器地址'
    )
    args = parser.parse_args()
    
    # 检查图片文件
    if not Path(args.image).exists():
        print(f"错误: 图片文件不存在: {args.image}")
        return
    
    print("=" * 60)
    print(f"图片路径: {args.image}")
    print(f"服务器地址: {args.server}")
    print("=" * 60)
    
    # 1. 加载图片
    print("\n[1/3] 加载图片...")
    image_base64 = load_image_to_base64(args.image)
    print("图片加载完成")
    
    # 2. 解析 state
    print("\n[2/3] 解析 state...")
    state = parse_state(args.state, args.state_file)
    if state:
        print(f"State 维度: {len(state)}")
    else:
        print("未提供 state")
    
    # 3. 发送到服务器
    print("\n[3/3] 发送到服务器...")
    try:
        result = send_to_server(image_base64, state, args.server)
        
        print("\n" + "=" * 60)
        print("✓ 推理成功!")
        print("=" * 60)
        print(f"Chunk size: {result['chunk_size']}")
        print(f"Action dim: {result['action_dim']}")
        print(f"\n时间统计:")
        timing = result['timing']
        print(f"  VLM 推理: {timing['vlm_time']:.3f}s")
        print(f"  Pons 推理: {timing['pons_time']:.3f}s")
        print(f"  ParaCAT 推理: {timing['paracat_time']:.3f}s")
        print(f"  总耗时: {timing['total_time']:.3f}s")
        
        print(f"\n预测动作 (全部 {len(result['actions'])} 步):")
        actions = result['actions']
        discrete_actions = result['discrete_actions']
        
        for i in range(len(actions)):
            action_str = ', '.join([f'{v:.4f}' for v in actions[i]])
            discrete_str = ', '.join([str(v) for v in discrete_actions[i]])
            print(f"  Step {i:2d}: [{action_str}]")
            print(f"          discrete: [{discrete_str}]")
        
    except requests.exceptions.ConnectionError:
        print(f"\n错误: 无法连接到服务器 {args.server}")
        print("请确保服务器已启动: python server.py --config config.yaml")
    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == "__main__":
    main()

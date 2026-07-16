"""
Sai0 VLA 推理客户端
用于与推理服务器通信
"""

import time
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import requests
import numpy as np
from PIL import Image


class Sai0Client:
    """
    Sai0 VLA 推理客户端
    
    使用示例:
        >>> client = Sai0Client("http://localhost:5000")
        >>> 
        >>> # 加载图像
        >>> images = [Image.open("frame1.jpg"), Image.open("frame2.jpg")]
        >>> state = np.zeros(16)
        >>> 
        >>> # 预测
        >>> result = client.predict(images, state, prompt="Pick up an apple.")
        >>> actions = result['actions']
        >>> print(f"Actions: {actions}")
    """
    
    def __init__(self, server_url: str = "http://localhost:5000", timeout: int = 30):
        """
        初始化客户端
        
        Args:
            server_url: 服务器 URL
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        
        # 检查服务器连接
        self._check_connection()
    
    def _check_connection(self):
        """检查服务器连接"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ 连接到服务器: {self.server_url}")
                print(f"  状态: {data['status']}")
                print(f"  Pipeline 已加载: {data['pipeline_loaded']}")
            else:
                print(f"⚠ 服务器响应异常: {response.status_code}")
        except Exception as e:
            print(f"✗ 无法连接到服务器: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        response = requests.get(f"{self.server_url}/info", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def encode_image(image: Union[Image.Image, str, Path]) -> str:
        """
        将图像编码为 base64 字符串
        
        Args:
            image: PIL Image, 图像文件路径, 或 Path 对象
        
        Returns:
            base64 编码的字符串
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    @staticmethod
    def decode_image(image_str: str) -> Image.Image:
        """
        从 base64 字符串解码图像
        
        Args:
            image_str: base64 编码的图像字符串
        
        Returns:
            PIL Image
        """
        if ',' in image_str:
            image_str = image_str.split(',', 1)[1]
        
        image_data = base64.b64decode(image_str)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    
    def predict(
        self,
        images: List[Union[Image.Image, str, Path]],
        state: Union[np.ndarray, List[float]],
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        预测动作
        
        Args:
            images: 图像列表（PIL Image, 文件路径, 或 Path）
            state: 机器人状态向量
            prompt: 可选的 prompt
        
        Returns:
            预测结果字典:
            {
                'actions': [[a1, a2, ...], ...],
                'timing': {...},
                'metadata': {...}
            }
        """
        # 编码图像
        encoded_images = [self.encode_image(img) for img in images]
        
        # 转换状态为列表
        if isinstance(state, np.ndarray):
            state = state.tolist()
        
        # 构建请求
        payload = {
            'images': encoded_images,
            'state': state
        }
        
        if prompt is not None:
            payload['prompt'] = prompt
        
        # 发送请求
        start_time = time.time()
        response = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=self.timeout
        )
        request_time = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        # 添加客户端侧的请求时间
        result['request_time'] = request_time
        
        return result
    
    def predict_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        批量预测
        
        Args:
            batch: 批次列表，每个元素包含:
                {
                    'images': [Image, ...],
                    'state': [s1, s2, ...],
                    'prompt': "..."  # 可选
                }
        
        Returns:
            批量预测结果:
            {
                'results': [
                    {'actions': [...], 'timing': {...}, 'metadata': {...}},
                    ...
                ]
            }
        """
        # 编码批次
        encoded_batch = []
        for item in batch:
            encoded_item = {
                'images': [self.encode_image(img) for img in item['images']],
                'state': item['state'].tolist() if isinstance(item['state'], np.ndarray) else item['state']
            }
            if 'prompt' in item:
                encoded_item['prompt'] = item['prompt']
            encoded_batch.append(encoded_item)
        
        # 发送请求
        response = requests.post(
            f"{self.server_url}/predict_batch",
            json={'batch': encoded_batch},
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def predict_stream(
        self,
        images_generator,
        state_generator,
        prompt: Optional[str] = None,
        callback=None
    ):
        """
        流式预测（持续发送图像和状态，接收动作）
        
        Args:
            images_generator: 图像生成器，每次 yield 一个图像列表
            state_generator: 状态生成器，每次 yield 一个状态向量
            prompt: 可选的 prompt
            callback: 回调函数，接收预测结果
        
        Yields:
            预测结果
        """
        for images, state in zip(images_generator, state_generator):
            result = self.predict(images, state, prompt=prompt)
            
            if callback:
                callback(result)
            
            yield result


def main():
    """示例：使用客户端进行推理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sai0 VLA 推理客户端示例')
    parser.add_argument('--server_url', type=str, default='http://localhost:5000',
                       help='服务器 URL')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='测试图像目录')
    parser.add_argument('--state_dim', type=int, default=16,
                       help='状态维度')
    parser.add_argument('--prompt', type=str, default='Pick up an apple.',
                       help='Prompt')
    
    args = parser.parse_args()
    
    # 创建客户端
    client = Sai0Client(args.server_url)
    
    # 获取模型信息
    print("\n" + "="*60)
    print("模型信息")
    print("="*60)
    info = client.get_info()
    for key, value in info.items():
        print(f"{key:25s}: {value}")
    print("="*60 + "\n")
    
    # 加载测试图像
    image_dir = Path(args.image_dir)
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    if not image_files:
        print(f"未找到图像: {image_dir}")
        return
    
    # 限制图像数量（避免过多）
    image_files = image_files[:5]
    
    print(f"加载 {len(image_files)} 张图像:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    images = [Image.open(f) for f in image_files]
    
    # 创建 dummy state
    state = np.zeros(args.state_dim, dtype=np.float32)
    
    # 执行预测
    print(f"\n发送推理请求...")
    print(f"  Prompt: {args.prompt}")
    print(f"  图像数量: {len(images)}")
    print(f"  State shape: {state.shape}")
    
    start_time = time.time()
    result = client.predict(images, state, prompt=args.prompt)
    client_total_time = time.time() - start_time
    
    # 打印结果
    print(f"\n{'='*60}")
    print("预测结果")
    print(f"{'='*60}")
    
    actions = np.array(result['actions'])
    print(f"Actions shape: {actions.shape}")
    print(f"Actions 范围: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"前几个值: {actions.flatten()[:10]}")
    
    print(f"\n时间统计:")
    timing = result['timing']
    print(f"  VLM 推理: {timing['vlm_time']:.4f}s")
    print(f"  Action Head 推理: {timing['action_head_time']:.4f}s")
    print(f"  服务器总时间: {timing['total_time']:.4f}s")
    print(f"  客户端总时间: {client_total_time:.4f}s")
    print(f"  网络开销: {client_total_time - timing['total_time']:.4f}s")
    
    metadata = result['metadata']
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

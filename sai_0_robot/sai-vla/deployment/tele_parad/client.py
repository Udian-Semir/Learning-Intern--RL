"""
OFT 遥操作推理客户端

使用方法:
    # 基本使用
    from client import OFTClient
    
    client = OFTClient("http://localhost:8000")
    
    # 从文件预测
    actions = client.predict_from_files(["image.jpg"], state=[...])
    
    # 从 numpy 数组预测
    actions = client.predict(images=[np_array], state=[...])
    
    # 从 PIL 图像预测
    actions = client.predict_pil(images=[pil_image], state=[...])
"""

import os
import time
import base64
from io import BytesIO
from typing import List, Optional, Union, Dict, Any

import numpy as np
from PIL import Image
import requests


class OFTClient:
    """OFT 推理客户端"""
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        """
        初始化客户端
        
        Args:
            server_url: 服务器地址
            timeout: 请求超时时间 (秒)
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _encode_image(self, image: Union[np.ndarray, Image.Image, str]) -> str:
        """
        将图像编码为 Base64
        
        Args:
            image: numpy 数组、PIL 图像或文件路径
            
        Returns:
            Base64 编码的字符串
        """
        if isinstance(image, str):
            # 文件路径
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        
        elif isinstance(image, np.ndarray):
            # numpy 数组
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        elif isinstance(image, Image.Image):
            # PIL 图像
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
    
    def predict(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        state: Optional[List[float]] = None,
        instruction: Optional[str] = None,
        # 图像预处理参数 (覆盖服务器全局配置)
        image_resize: Optional[List[int]] = None,
        flip_horizontal: Optional[bool] = None,
        flip_vertical: Optional[bool] = None,
        rotate_180: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        执行预测
        
        Args:
            images: 图像列表 (numpy 数组、PIL 图像或文件路径)
            state: 机器人状态向量 (可选)
            instruction: 任务指令 (可选)
            image_resize: 图像 resize [width, height] (覆盖服务器配置)
            flip_horizontal: 水平翻转 (覆盖服务器配置)
            flip_vertical: 垂直翻转 (覆盖服务器配置)
            rotate_180: 旋转 180 度 (覆盖服务器配置)
            
        Returns:
            预测结果字典:
            - actions: 连续动作 [chunk_size, action_dim]
            - timing: 时间统计
            - chunk_size: chunk 大小
            - action_dim: 动作维度
        """
        # 编码图像
        encoded_images = [self._encode_image(img) for img in images]
        
        # 构建请求
        payload = {
            "images": encoded_images,
        }
        
        if state is not None:
            payload["state"] = state
        
        if instruction is not None:
            payload["instruction"] = instruction
        
        # 图像预处理参数 (仅当不为 None 时添加)
        if image_resize is not None:
            payload["image_resize"] = image_resize
        if flip_horizontal is not None:
            payload["flip_horizontal"] = flip_horizontal
        if flip_vertical is not None:
            payload["flip_vertical"] = flip_vertical
        if rotate_180 is not None:
            payload["rotate_180"] = rotate_180
        
        # 发送请求
        url = f"{self.server_url}/predict"
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            raise TimeoutError(f"请求超时 ({self.timeout}s)")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"请求失败: {e}")
    
    def predict_from_files(
        self,
        image_paths: List[str],
        state: Optional[List[float]] = None,
        instruction: Optional[str] = None,
        image_resize: Optional[List[int]] = None,
        flip_horizontal: Optional[bool] = None,
        flip_vertical: Optional[bool] = None,
        rotate_180: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        从图像文件执行预测
        
        Args:
            image_paths: 图像文件路径列表
            state: 机器人状态向量 (可选)
            instruction: 任务指令 (可选)
            image_resize: 图像 resize [width, height]
            flip_horizontal: 水平翻转
            flip_vertical: 垂直翻转
            rotate_180: 旋转 180 度
            
        Returns:
            预测结果字典
        """
        return self.predict(
            images=image_paths,
            state=state,
            instruction=instruction,
            image_resize=image_resize,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
            rotate_180=rotate_180,
        )
    
    def predict_pil(
        self,
        images: List[Image.Image],
        state: Optional[List[float]] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        从 PIL 图像执行预测
        
        Args:
            images: PIL 图像列表
            state: 机器人状态向量 (可选)
            instruction: 任务指令 (可选)
            
        Returns:
            预测结果字典
        """
        return self.predict(images=images, state=state, instruction=instruction)
    
    def predict_numpy(
        self,
        images: List[np.ndarray],
        state: Optional[List[float]] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        从 numpy 数组执行预测
        
        Args:
            images: numpy 数组列表 (HWC 格式, RGB)
            state: 机器人状态向量 (可选)
            instruction: 任务指令 (可选)
            
        Returns:
            预测结果字典
        """
        return self.predict(images=images, state=state, instruction=instruction)
    
    def get_actions(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        state: Optional[List[float]] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        执行预测并只返回动作数组
        
        Args:
            images: 图像列表
            state: 机器人状态向量 (可选)
            instruction: 任务指令 (可选)
            
        Returns:
            动作数组 [chunk_size, action_dim]
        """
        result = self.predict(images=images, state=state, instruction=instruction)
        return np.array(result['actions'])
    
    def health(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        url = f"{self.server_url}/health"
        response = self.session.get(url, timeout=5.0)
        response.raise_for_status()
        return response.json()
    
    def info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        url = f"{self.server_url}/info"
        response = self.session.get(url, timeout=5.0)
        response.raise_for_status()
        return response.json()
    
    def is_ready(self) -> bool:
        """
        检查服务是否就绪
        
        Returns:
            是否就绪
        """
        try:
            health = self.health()
            return health.get('status') == 'ready'
        except:
            return False
    
    def wait_until_ready(self, timeout: float = 60.0, interval: float = 1.0):
        """
        等待服务就绪
        
        Args:
            timeout: 超时时间 (秒)
            interval: 检查间隔 (秒)
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ready():
                return
            time.sleep(interval)
        
        raise TimeoutError(f"服务未在 {timeout}s 内就绪")


# ==================== 命令行工具 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='OFT 客户端测试')
    parser.add_argument('--server', type=str, default='http://localhost:8000', help='服务器地址')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--state', type=float, nargs='+', default=None, help='状态向量')
    parser.add_argument('--instruction', type=str, default=None, help='任务指令')
    args = parser.parse_args()
    
    # 创建客户端
    client = OFTClient(args.server)
    
    # 检查健康状态
    print("检查服务状态...")
    try:
        health = client.health()
        print(f"服务状态: {health['status']}")
        print(f"模型加载: {health['models_loaded']}")
    except Exception as e:
        print(f"无法连接服务器: {e}")
        return
    
    # 获取模型信息
    try:
        info = client.info()
        print(f"\n模型信息:")
        print(f"  VLM: {info['vlm_type']}")
        print(f"  Chunk Size: {info['chunk_size']}")
        print(f"  Action Dim: {info['action_dim']}")
    except Exception as e:
        print(f"获取模型信息失败: {e}")
    
    # 执行预测
    print(f"\n执行预测...")
    print(f"  图像: {args.image}")
    print(f"  状态: {args.state}")
    print(f"  指令: {args.instruction}")
    
    try:
        result = client.predict_from_files(
            image_paths=[args.image],
            state=args.state,
            instruction=args.instruction,
        )
        
        print(f"\n预测结果:")
        print(f"  Chunk Size: {result['chunk_size']}")
        print(f"  Action Dim: {result['action_dim']}")
        print(f"  时间统计:")
        print(f"    VLM: {result['timing']['vlm_time']:.4f}s")
        print(f"    OFT: {result['timing']['oft_time']:.4f}s")
        print(f"    总计: {result['timing']['total_time']:.4f}s")
        
        print(f"\n动作序列 (前 5 步):")
        actions = np.array(result['actions'])
        for i in range(min(5, len(actions))):
            print(f"  Step {i}: {actions[i][:6]}... (前6维)")
        
    except Exception as e:
        print(f"预测失败: {e}")


if __name__ == "__main__":
    main()

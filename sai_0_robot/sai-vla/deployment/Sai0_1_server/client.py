"""
Sai0_1 VLA 推理服务客户端

提供便捷的 Python 客户端接口，用于与 Sai0_1 推理服务器通信。

使用示例:
    from client import Sai0Client
    
    client = Sai0Client("http://localhost:5000")
    
    # 单次预测
    result = client.predict(
        images=[img1, img2],
        state=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        prompt="pick up the red apple"
    )
    print(f"Actions: {result['actions']}")
    
    # 批量预测
    results = client.predict_batch([
        {"images": [img1, img2], "state": state1, "prompt": "task1"},
        {"images": [img3, img4], "state": state2, "prompt": "task2"},
    ])
"""

import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Union

import numpy as np
import requests
from PIL import Image


class Sai0Client:
    """
    Sai0_1 推理服务客户端
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        timeout: float = 30.0,
    ):
        """
        初始化客户端
        
        Args:
            base_url: 服务器 URL
            timeout: 请求超时时间 (秒)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        response = requests.get(
            f"{self.base_url}/info",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_preprocess_config(self) -> Dict[str, Any]:
        """获取当前预处理配置"""
        response = requests.get(
            f"{self.base_url}/config/preprocess",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def update_preprocess_config(
        self,
        image_preprocess: Optional[Dict[str, Any]] = None,
        state_preprocess: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        更新预处理配置
        
        Args:
            image_preprocess: 图像预处理配置
                - resize: [width, height] 或 null
                - flip_horizontal: bool
                - flip_vertical: bool
                - rotate_180: bool
            state_preprocess: 状态预处理配置
                格式: {索引: {enable_normalization, min_val, max_val, zero_to_minus_one}}
        
        Returns:
            更新后的配置
        """
        data = {}
        if image_preprocess is not None:
            data["image_preprocess"] = image_preprocess
        if state_preprocess is not None:
            data["state_preprocess"] = {
                str(k): v for k, v in state_preprocess.items()
            }
        
        response = requests.post(
            f"{self.base_url}/config/update",
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        state: Union[List[float], np.ndarray],
        prompt: Optional[str] = None,
        image_format: str = "base64",
        # 图像预处理覆盖参数
        image_resize: Optional[List[int]] = None,
        image_flip_horizontal: Optional[bool] = None,
        image_flip_vertical: Optional[bool] = None,
        image_rotate_180: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        预测动作
        
        Args:
            images: 图像列表 (PIL.Image, numpy array, 或文件路径)
            state: 机器人状态向量
            prompt: 任务指令 (可选)
            image_format: 图像发送格式 ('base64' 或 'numpy')
            image_resize: 覆盖图像 resize 配置
            image_flip_horizontal: 覆盖水平翻转配置
            image_flip_vertical: 覆盖垂直翻转配置
            image_rotate_180: 覆盖旋转 180 度配置
        
        Returns:
            预测结果字典，包含:
            - actions: 动作序列
            - timing: 时间统计
            - metadata: 元数据
        """
        # 处理图像
        encoded_images = self._encode_images(images, image_format)
        
        # 处理状态
        if isinstance(state, np.ndarray):
            state = state.tolist()
        
        # 构建请求
        data = {
            "images": encoded_images,
            "state": state,
            "image_format": image_format,
        }
        
        if prompt is not None:
            data["prompt"] = prompt
        if image_resize is not None:
            data["image_resize"] = image_resize
        if image_flip_horizontal is not None:
            data["image_flip_horizontal"] = image_flip_horizontal
        if image_flip_vertical is not None:
            data["image_flip_vertical"] = image_flip_vertical
        if image_rotate_180 is not None:
            data["image_rotate_180"] = image_rotate_180
        
        # 发送请求
        response = requests.post(
            f"{self.base_url}/predict",
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def predict_batch(
        self,
        batch: List[Dict[str, Any]],
        image_format: str = "base64",
    ) -> List[Dict[str, Any]]:
        """
        批量预测动作
        
        Args:
            batch: 批次列表，每个元素包含:
                - images: 图像列表
                - state: 状态向量
                - prompt: 任务指令 (可选)
                - image_resize: 图像 resize (可选)
                - image_flip_horizontal: 水平翻转 (可选)
                - image_flip_vertical: 垂直翻转 (可选)
                - image_rotate_180: 旋转 180 度 (可选)
            image_format: 图像发送格式
        
        Returns:
            预测结果列表
        """
        # 处理批次
        processed_batch = []
        for item in batch:
            encoded_images = self._encode_images(item["images"], image_format)
            state = item["state"]
            if isinstance(state, np.ndarray):
                state = state.tolist()
            
            processed_item = {
                "images": encoded_images,
                "state": state,
                "image_format": image_format,
            }
            
            if "prompt" in item:
                processed_item["prompt"] = item["prompt"]
            if "image_resize" in item:
                processed_item["image_resize"] = item["image_resize"]
            if "image_flip_horizontal" in item:
                processed_item["image_flip_horizontal"] = item["image_flip_horizontal"]
            if "image_flip_vertical" in item:
                processed_item["image_flip_vertical"] = item["image_flip_vertical"]
            if "image_rotate_180" in item:
                processed_item["image_rotate_180"] = item["image_rotate_180"]
            
            processed_batch.append(processed_item)
        
        # 发送请求
        response = requests.post(
            f"{self.base_url}/predict_batch",
            json={"batch": processed_batch},
            timeout=self.timeout * len(batch)
        )
        response.raise_for_status()
        
        return response.json()["results"]
    
    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        response = requests.get(
            f"{self.base_url}/latency_stats",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def reset_latency_stats(self) -> Dict[str, str]:
        """重置延迟统计"""
        response = requests.post(
            f"{self.base_url}/latency_stats/reset",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def _encode_images(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        image_format: str
    ) -> List:
        """编码图像"""
        encoded = []
        
        for img in images:
            # 加载图像
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            if image_format == "base64":
                # Base64 编码
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=95)
                encoded.append(base64.b64encode(buffered.getvalue()).decode())
            elif image_format == "numpy":
                # Numpy array (JSON 格式)
                encoded.append(np.array(img).tolist())
            else:
                raise ValueError(f"Unsupported image format: {image_format}")
        
        return encoded


# ==================== 便捷函数 ====================

def predict_once(
    images: List[Union[Image.Image, np.ndarray, str]],
    state: Union[List[float], np.ndarray],
    prompt: Optional[str] = None,
    server_url: str = "http://localhost:5000",
    **kwargs
) -> Dict[str, Any]:
    """
    单次预测便捷函数
    
    Args:
        images: 图像列表
        state: 状态向量
        prompt: 任务指令
        server_url: 服务器 URL
        **kwargs: 其他参数传递给 predict()
    
    Returns:
        预测结果
    """
    client = Sai0Client(server_url)
    return client.predict(images, state, prompt, **kwargs)


# ==================== 测试 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sai0_1 客户端测试')
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                       help='服务器 URL')
    parser.add_argument('--test', type=str, default='health',
                       choices=['health', 'info', 'config', 'predict'],
                       help='测试类型')
    args = parser.parse_args()
    
    client = Sai0Client(args.server)
    
    if args.test == 'health':
        print("健康检查:")
        print(client.health_check())
    
    elif args.test == 'info':
        print("服务器信息:")
        print(client.get_info())
    
    elif args.test == 'config':
        print("预处理配置:")
        print(client.get_preprocess_config())
    
    elif args.test == 'predict':
        # 创建测试图像
        test_images = [
            Image.new('RGB', (256, 256), color=(100, 150, 200)),
            Image.new('RGB', (256, 256), color=(200, 150, 100)),
        ]
        test_state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]  # 注意索引 6 是 0
        
        print("预测测试:")
        result = client.predict(
            images=test_images,
            state=test_state,
            prompt="pick up the object"
        )
        print(f"动作形状: {result['metadata']['action_shape']}")
        print(f"预处理后状态: {result['metadata']['preprocessed_state']}")
        print(f"时间统计: {result['timing']}")

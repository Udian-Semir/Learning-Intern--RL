#!/usr/bin/env python3
"""
从LeRobot数据集提取Qwen3-VL-2B-Instruct模型指定层的hidden states

使用方法:
    python extract_vlm_hidden_states.py --dataset_path /path/to/dataset --layer 14
    python extract_vlm_hidden_states.py --dataset_path /path/to/dataset --layer 14 --gpu 0
    python extract_vlm_hidden_states.py --dataset_path /path/to/dataset --layer 14 --start_idx 0 --end_idx 1000
"""

import os
import argparse
import sys

# 先解析GPU参数，在import torch之前设置CUDA_VISIBLE_DEVICES
def get_gpu_arg():
    """提前解析--gpu参数"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=str, default="0")
    args, _ = parser.parse_known_args()
    return args.gpu

# 在import torch之前设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_arg()

import json
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading


def parse_args():
    parser = argparse.ArgumentParser(description="从LeRobot数据集提取VLM hidden states")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/HuangWenlong/datasets/qwen/libero_github_convert_for_qwen2b-only-libero_spatial",
        help="LeRobot数据集路径"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Qwen3-VL模型ID"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="要提取的language_model transformer层号 (1-28，共28层)。例如: --layer 14 表示提取第14层transformer的输出"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="使用的GPU编号"
    )
    # What shout
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="起始帧索引（可选，用于断点续传）"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="结束帧索引（可选，用于断点续传）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理大小（默认为1）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认为数据集目录下的vlm_hidden_states）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细信息（input_ids和decode内容）"
    )
    parser.add_argument(
        "--add_action_prompt",
        action="store_true",
        help="在图像后添加action prompt: 'What action should the robot take to {instruction}?'"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据预加载的worker数量（默认为4）"
    )
    parser.add_argument(
        "--prefetch_size",
        type=int,
        default=8,
        help="预加载队列大小（默认为8）"
    )
    parser.add_argument(
        "--flip_images",
        action="store_true",
        help="水平翻转图像"
    )
    return parser.parse_args()


def load_dataset_info(dataset_path: str):
    """加载数据集元信息"""
    meta_path = Path(dataset_path) / "meta"
    
    # 读取info.json
    with open(meta_path / "info.json", "r") as f:
        info = json.load(f)
    
    # 读取tasks.jsonl
    tasks = {}
    with open(meta_path / "tasks.jsonl", "r") as f:
        for line in f:
            task = json.loads(line)
            tasks[task["task_index"]] = task["task"]
    
    return info, tasks


def get_all_frames_info(dataset_path: str, info: dict):
    """获取所有帧的信息列表"""
    frames_info = []
    data_path = Path(dataset_path) / "data"
    chunks_size = info["chunks_size"]
    total_episodes = info["total_episodes"]
    
    for episode_idx in range(total_episodes):
        chunk_idx = episode_idx // chunks_size
        parquet_path = data_path / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
        
        if not parquet_path.exists():
            continue
        
        df = pd.read_parquet(parquet_path)
        
        # 计算episode内的本地帧索引（从0开始）
        min_index = df["index"].min()
        
        for _, row in df.iterrows():
            frames_info.append({
                "episode_index": row["episode_index"],
                "frame_index": row["index"] - min_index,  # 本地帧索引（视频中的帧号）
                "global_index": row["vlm_hidden_state_index"] if "vlm_hidden_state_index" in row else row["index"],  # 用于命名
                "task_index": row["task_index"],
                "chunk_index": chunk_idx,
            })
    
    return frames_info


def get_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """从视频中提取指定帧"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取视频帧: {video_path}, frame {frame_idx}")
    
    # BGR转RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_two_images(dataset_path: str, episode_idx: int, frame_idx: int, chunk_idx: int, flip: bool = False) -> tuple:
    """获取agentview和wrist两张图像
    
    Args:
        flip: 是否水平翻转图像
    """
    videos_path = Path(dataset_path) / "videos"
    
    agentview_path = videos_path / f"chunk-{chunk_idx:03d}" / "observation.images.agentview" / f"episode_{episode_idx:06d}.mp4"
    wrist_path = videos_path / f"chunk-{chunk_idx:03d}" / "observation.images.wrist" / f"episode_{episode_idx:06d}.mp4"
    
    img_agentview = get_video_frame(str(agentview_path), frame_idx)
    img_wrist = get_video_frame(str(wrist_path), frame_idx)
    
    # 翻转图像（180度旋转：上下+左右都翻转）
    if flip:
        img_agentview = img_agentview[::-1, ::-1, :].copy()
        img_wrist = img_wrist[::-1, ::-1, :].copy()
    
    # 转换为PIL Image
    img_agentview = Image.fromarray(img_agentview)
    img_wrist = Image.fromarray(img_wrist)
    
    return img_agentview, img_wrist


def prepare_vlm_input(processor, img_agentview: Image.Image, img_wrist: Image.Image, task_description: str, add_action_prompt: bool = False):
    """准备VLM输入"""
    # 构建文本内容
    if add_action_prompt:
        text_content = f"What action should the robot take to {task_description.lower()}?"
    else:
        text_content = task_description
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_agentview},
                {"type": "image", "image": img_wrist},
                {"type": "text", "text": text_content}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    return inputs


class DataPrefetcher:
    """数据预加载器，使用多线程预先加载图像数据"""
    
    def __init__(self, frames_info, dataset_path, tasks, processor, add_action_prompt, 
                 output_dir, num_workers=4, prefetch_size=8, flip_images=False):
        self.frames_info = frames_info
        self.dataset_path = dataset_path
        self.tasks = tasks
        self.processor = processor
        self.add_action_prompt = add_action_prompt
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.flip_images = flip_images
        
        self.queue = Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        self.executor = None
        self.loader_thread = None
    
    def _load_single_frame(self, frame_info):
        """加载单帧数据"""
        global_idx = frame_info["global_index"]
        output_path = os.path.join(self.output_dir, f"hidden_state_{global_idx:06d}.npy")
        
        # 如果已存在，跳过
        if os.path.exists(output_path):
            return None
        
        try:
            task_description = self.tasks[frame_info["task_index"]]
            img_agentview, img_wrist = get_two_images(
                self.dataset_path,
                frame_info["episode_index"],
                frame_info["frame_index"],
                frame_info["chunk_index"],
                flip=self.flip_images
            )
            inputs = prepare_vlm_input(self.processor, img_agentview, img_wrist, 
                                       task_description, self.add_action_prompt)
            return {
                "global_idx": global_idx,
                "output_path": output_path,
                "inputs": inputs
            }
        except Exception as e:
            return {"error": str(e), "global_idx": global_idx}
    
    def _loader_worker(self):
        """后台加载线程"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            frame_iter = iter(self.frames_info)
            
            # 初始填充
            for _ in range(min(self.num_workers * 2, len(self.frames_info))):
                try:
                    frame_info = next(frame_iter)
                    futures.append(executor.submit(self._load_single_frame, frame_info))
                except StopIteration:
                    break
            
            while futures and not self.stop_event.is_set():
                # 等待第一个完成
                future = futures.pop(0)
                result = future.result()
                
                if result is not None:
                    self.queue.put(result)
                
                # 提交新任务
                try:
                    frame_info = next(frame_iter)
                    futures.append(executor.submit(self._load_single_frame, frame_info))
                except StopIteration:
                    pass
            
            # 结束信号
            self.queue.put(None)
    
    def start(self):
        """启动预加载"""
        self.loader_thread = threading.Thread(target=self._loader_worker, daemon=True)
        self.loader_thread.start()
    
    def stop(self):
        """停止预加载"""
        self.stop_event.set()
        if self.loader_thread:
            self.loader_thread.join(timeout=1)
    
    def __iter__(self):
        self.start()
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item
        self.stop()


def extract_hidden_state(model, inputs: dict, layer_idx: int, device: str, processor=None, verbose: bool = False) -> np.ndarray:
    """提取指定层的hidden state"""
    # 移动到GPU
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    # 打印verbose信息
    if verbose and processor is not None:
        print("\n" + "-" * 50)
        print("【Verbose模式】")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Input IDs: {inputs['input_ids']}")
        decoded_text = processor.decode(inputs['input_ids'][0], skip_special_tokens=False)
        print(f"Decoded内容:\n{decoded_text}")
        print("-" * 50)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    # hidden_states是一个tuple，包含29个tensor（embedding层 + 28个transformer层）
    # 索引0是embedding层输出，索引1-28是transformer层1-28的输出
    # 所以layer_idx=14对应的是hidden_states[14]（第14层的输出）
    # 注意：layer_idx=0对应embedding输出，layer_idx=1-28对应transformer layer 0-27
    
    # 实际上，如果用户想要第14层（从1开始计数），应该使用hidden_states[14]
    # 如果用户想要第14层（从0开始计数），应该使用hidden_states[15]
    # 这里我们假设用户说的第14层是transformer的第14层（从0开始计数），即hidden_states[15]
    # 但为了简单，我们直接使用layer_idx作为索引
    
    hidden_state = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
    
    # 取最后一个token的hidden state（通常用于生成任务）
    # 或者取所有token的平均
    # 这里我们保存完整的hidden state
    hidden_state = hidden_state.cpu().float().numpy()  # 转为float32以减少精度损失
    
    return hidden_state


def main():
    args = parse_args()
    
    # GPU已在import前设置，这里只是确认
    device = "cuda:0"
    
    print("=" * 70)
    print("VLM Hidden States 提取工具")
    print("=" * 70)
    print(f"数据集路径: {args.dataset_path}")
    print(f"模型: {args.model_id}")
    print(f"提取层索引: {args.layer}")
    print(f"使用GPU: {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    print("=" * 70)
    
    # 加载数据集信息
    print("\n正在加载数据集信息...")
    info, tasks = load_dataset_info(args.dataset_path)
    print(f"总episodes: {info['total_episodes']}")
    print(f"总帧数: {info['total_frames']}")
    print(f"任务数: {len(tasks)}")
    
    # 获取所有帧信息
    print("\n正在收集所有帧信息...")
    frames_info = get_all_frames_info(args.dataset_path, info)
    print(f"收集到 {len(frames_info)} 帧")
    
    # 确定处理范围
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(frames_info)
    frames_to_process = frames_info[start_idx:end_idx]
    print(f"将处理帧索引 [{start_idx}, {end_idx}), 共 {len(frames_to_process)} 帧")
    
    # 设置输出目录
    output_dir = args.output_dir if args.output_dir else os.path.join(args.dataset_path, "vlm_hidden_states")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载模型和处理器
    print("\n正在加载模型...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    print("正在加载处理器...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    # 验证层索引
    # Qwen3-VL-2B-Instruct的language_model有28层（transformer layer 0-27）
    # hidden_states有29个元素:
    #   - 索引0: embedding层输出
    #   - 索引1-28: transformer第1-28层的输出（对应layer 0-27）
    # 用户指定的layer是从1开始计数的transformer层号
    # 所以 --layer 14 表示第14个transformer层，对应 hidden_states[14]
    
    if args.layer < 1 or args.layer > 28:
        raise ValueError(f"层号必须在 [1, 28] 范围内，当前: {args.layer}")
    
    # 用户层号直接对应hidden_states的索引
    hidden_state_idx = args.layer
    
    print(f"\n将提取 language_model 的第 {args.layer} 层 transformer 输出")
    print(f"  - 对应 hidden_states[{hidden_state_idx}]")
    print(f"  - Qwen3-VL-2B-Instruct 共有 28 层 transformer")
    
    # 处理每一帧
    print("\n" + "=" * 70)
    print("开始提取hidden states...")
    print(f"使用 {args.num_workers} 个worker进行数据预加载，预加载队列大小: {args.prefetch_size}")
    if args.flip_images:
        print("已启用图像水平翻转")
    print("=" * 70)
    
    # 创建数据预加载器
    prefetcher = DataPrefetcher(
        frames_info=frames_to_process,
        dataset_path=args.dataset_path,
        tasks=tasks,
        processor=processor,
        add_action_prompt=args.add_action_prompt,
        output_dir=output_dir,
        num_workers=args.num_workers,
        prefetch_size=args.prefetch_size,
        flip_images=args.flip_images
    )
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 计算需要处理的总数（排除已存在的）
    total_to_process = sum(1 for f in frames_to_process 
                          if not os.path.exists(os.path.join(output_dir, f"hidden_state_{f['global_index']:06d}.npy")))
    
    with tqdm(total=total_to_process, desc="提取hidden states") as pbar:
        for item in prefetcher:
            if "error" in item:
                print(f"\n处理帧 {item['global_idx']} 时出错: {item['error']}")
                error_count += 1
                continue
            
            try:
                # 提取hidden state
                hidden_state = extract_hidden_state(
                    model, item["inputs"], hidden_state_idx, device, 
                    processor, args.verbose
                )
                
                # 保存
                np.save(item["output_path"], hidden_state)
                processed_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"\n处理帧 {item['global_idx']} 时出错: {e}")
                error_count += 1
    
    print("\n" + "=" * 70)
    print("提取完成！")
    print(f"成功处理: {processed_count} 帧")
    print(f"错误: {error_count} 帧")
    print(f"Hidden states已保存到: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

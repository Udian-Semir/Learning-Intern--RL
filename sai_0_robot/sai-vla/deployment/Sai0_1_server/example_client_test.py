"""
Sai0_1 客户端测试示例

演示如何使用客户端与推理服务器通信。
"""

import time
import numpy as np
from PIL import Image

from client import Sai0Client


def test_health_check(client: Sai0Client):
    """测试健康检查"""
    print("\n" + "=" * 50)
    print("测试: 健康检查")
    print("=" * 50)
    
    result = client.health_check()
    print(f"状态: {result['status']}")
    print(f"Pipeline 已加载: {result['pipeline_loaded']}")


def test_server_info(client: Sai0Client):
    """测试服务器信息"""
    print("\n" + "=" * 50)
    print("测试: 服务器信息")
    print("=" * 50)
    
    result = client.get_info()
    print(f"VLM 类型: {result['vlm_type']}")
    print(f"Action Head 类型: {result['action_head_type']}")
    print(f"设备: {result['device']}")
    print(f"图像预处理配置: {result['image_preprocess']}")
    print(f"状态预处理配置: {result['state_preprocess']}")


def test_preprocess_config(client: Sai0Client):
    """测试预处理配置"""
    print("\n" + "=" * 50)
    print("测试: 预处理配置")
    print("=" * 50)
    
    # 获取当前配置
    config = client.get_preprocess_config()
    print("当前配置:")
    print(f"  图像: {config['image_preprocess']}")
    print(f"  状态: {config['state_preprocess']}")
    
    # 更新配置
    print("\n更新配置...")
    new_config = client.update_preprocess_config(
        image_preprocess={
            "resize": [256, 256],
            "flip_horizontal": False,
            "flip_vertical": False,
            "rotate_180": False,
        },
        state_preprocess={
            6: {
                "zero_to_minus_one": True,
                "enable_normalization": False,
                "min_val": None,
                "max_val": None,
            }
        }
    )
    print("更新后配置:")
    print(f"  图像: {new_config['image_preprocess']}")
    print(f"  状态: {new_config['state_preprocess']}")


def test_single_predict(client: Sai0Client):
    """测试单次预测"""
    print("\n" + "=" * 50)
    print("测试: 单次预测")
    print("=" * 50)
    
    # 创建测试图像 (模拟 agentview 和 wrist)
    agentview = Image.new('RGB', (256, 256), color=(100, 150, 200))
    wrist = Image.new('RGB', (256, 256), color=(200, 150, 100))
    
    # 创建测试状态
    # 假设: [x, y, z, qw, qx, qy, qz, gripper]
    state = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0]  # gripper=0 会被转换为 -1
    
    print(f"输入状态: {state}")
    print(f"注意: 索引 7 (gripper) 值为 0，应该被转换为 -1")
    
    # 预测
    start_time = time.time()
    result = client.predict(
        images=[agentview, wrist],
        state=state,
        prompt="pick up the red cube"
    )
    total_time = time.time() - start_time
    
    print(f"\n预测结果:")
    print(f"  动作形状: {result['metadata']['action_shape']}")
    print(f"  预处理后状态: {result['metadata']['preprocessed_state']}")
    print(f"  预处理时间: {result['timing']['preprocess_time']*1000:.1f}ms")
    print(f"  推理时间: {result['timing']['inference_time']*1000:.1f}ms")
    print(f"  总时间: {result['timing']['total_time']*1000:.1f}ms")
    print(f"  客户端总时间: {total_time*1000:.1f}ms")
    
    # 打印第一步动作
    actions = np.array(result['actions'])
    print(f"\n第一步动作: {actions[0]}")


def test_predict_with_override(client: Sai0Client):
    """测试带预处理覆盖的预测"""
    print("\n" + "=" * 50)
    print("测试: 带预处理覆盖的预测")
    print("=" * 50)
    
    # 创建测试图像
    img = Image.new('RGB', (512, 512), color=(150, 150, 150))
    
    # 状态
    state = [0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]
    
    # 预测 (覆盖 resize 配置)
    result = client.predict(
        images=[img, img],
        state=state,
        prompt="move to target",
        image_resize=[128, 128],  # 覆盖全局配置
        image_flip_horizontal=True,  # 覆盖全局配置
    )
    
    print(f"使用覆盖配置: resize=[128,128], flip_horizontal=True")
    print(f"动作形状: {result['metadata']['action_shape']}")


def test_batch_predict(client: Sai0Client):
    """测试批量预测"""
    print("\n" + "=" * 50)
    print("测试: 批量预测")
    print("=" * 50)
    
    # 创建多个测试样本
    batch = []
    for i in range(3):
        img1 = Image.new('RGB', (256, 256), color=(100 + i*50, 150, 200))
        img2 = Image.new('RGB', (256, 256), color=(200, 150, 100 + i*50))
        state = [0.1 * (i+1), 0.2 * (i+1), 0.3, 1.0, 0.0, 0.0, 0.0, float(i % 2)]
        
        batch.append({
            "images": [img1, img2],
            "state": state,
            "prompt": f"task {i+1}"
        })
    
    # 批量预测
    start_time = time.time()
    results = client.predict_batch(batch)
    total_time = time.time() - start_time
    
    print(f"批次大小: {len(batch)}")
    print(f"总时间: {total_time*1000:.1f}ms")
    print(f"平均每样本: {total_time*1000/len(batch):.1f}ms")
    
    for i, result in enumerate(results):
        actions = np.array(result['actions'])
        print(f"\n样本 {i+1}:")
        print(f"  动作形状: {result['metadata']['action_shape']}")
        print(f"  第一步动作: {actions[0][:3]}...")


def test_latency_stats(client: Sai0Client):
    """测试延迟统计"""
    print("\n" + "=" * 50)
    print("测试: 延迟统计")
    print("=" * 50)
    
    # 先重置
    client.reset_latency_stats()
    print("已重置延迟统计")
    
    # 执行多次预测
    img = Image.new('RGB', (256, 256), color=(128, 128, 128))
    state = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5]
    
    print("\n执行 5 次预测...")
    for i in range(5):
        client.predict(
            images=[img, img],
            state=state,
            prompt="test task"
        )
    
    # 获取统计
    stats = client.get_latency_stats()
    print(f"\n延迟统计 (ms):")
    print(f"  平均: {stats['mean']:.2f}")
    print(f"  标准差: {stats['std']:.2f}")
    print(f"  最小: {stats['min']:.2f}")
    print(f"  最大: {stats['max']:.2f}")
    print(f"  样本数: {stats['count']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sai0_1 客户端测试')
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                       help='服务器 URL')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'health', 'info', 'config', 'predict', 
                               'override', 'batch', 'latency'],
                       help='测试类型')
    args = parser.parse_args()
    
    client = Sai0Client(args.server)
    
    print("=" * 60)
    print(f"Sai0_1 客户端测试")
    print(f"服务器: {args.server}")
    print("=" * 60)
    
    if args.test == 'all' or args.test == 'health':
        test_health_check(client)
    
    if args.test == 'all' or args.test == 'info':
        test_server_info(client)
    
    if args.test == 'all' or args.test == 'config':
        test_preprocess_config(client)
    
    if args.test == 'all' or args.test == 'predict':
        test_single_predict(client)
    
    if args.test == 'all' or args.test == 'override':
        test_predict_with_override(client)
    
    if args.test == 'all' or args.test == 'batch':
        test_batch_predict(client)
    
    if args.test == 'all' or args.test == 'latency':
        test_latency_stats(client)
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

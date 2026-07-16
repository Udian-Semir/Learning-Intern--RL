#!/usr/bin/env python3
"""查看 npy 文件的维度信息，支持对比两个文件"""

import numpy as np
import os
from pathlib import Path

# ============ 在这里修改路径 ============
NPY_PATH = "/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_spatial/libero_lerobot_spatial_sys1_qwen_2b_1_14_28/vlm_hidden_states/hidden_state_000001.npy"  # 可以是单个文件或目录
RECURSIVE = False  # 是否递归搜索子目录
SHOW_CONTENT = False  # 是否显示数据内容（仅对小数组有效）

# ============ 对比两个文件 ============
COMPARE_MODE = False  # 设为 True 开启对比模式
NPY_PATH_2 = "/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_eagle25-only-libero_spatial/libero_lerobot_spatial/vlm_hidden_states/hidden_state_000002.npy"
# ======================================


def check_npy_shape(file_path):
    """查看单个 npy 文件的维度"""
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"文件: {file_path}")
        print(f"  形状 (shape): {data.shape}")
        print(f"  数据类型 (dtype): {data.dtype}")
        print(f"  总元素数: {data.size}")
        print(f"  文件大小: {os.path.getsize(file_path) / 1024:.2f} KB")
        print()
        return data
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return None


def compare_npy_files(path1, path2):
    """对比两个 npy 文件"""
    print("=" * 60)
    print("对比两个 npy 文件")
    print("=" * 60)
    
    # 加载两个文件
    data1 = check_npy_shape(path1)
    data2 = check_npy_shape(path2)
    
    if data1 is None or data2 is None:
        print("无法进行对比：文件加载失败")
        return
    
    print("-" * 60)
    print("对比结果:")
    print("-" * 60)
    
    # 检查形状是否相同
    if data1.shape != data2.shape:
        print(f"  ⚠️  形状不同!")
        print(f"      文件1: {data1.shape}")
        print(f"      文件2: {data2.shape}")
        print(f"  无法逐元素对比，因为形状不同")
        return
    
    print(f"  ✓ 形状相同: {data1.shape}")
    
    # 检查数据类型
    if data1.dtype != data2.dtype:
        print(f"  ⚠️  数据类型不同: {data1.dtype} vs {data2.dtype}")
    else:
        print(f"  ✓ 数据类型相同: {data1.dtype}")
    
    # 计算不同元素数量
    total_elements = data1.size
    
    # 处理 NaN 值
    nan_mask1 = np.isnan(data1) if np.issubdtype(data1.dtype, np.floating) else np.zeros_like(data1, dtype=bool)
    nan_mask2 = np.isnan(data2) if np.issubdtype(data2.dtype, np.floating) else np.zeros_like(data2, dtype=bool)
    
    # 不同的元素（考虑 NaN）
    if np.issubdtype(data1.dtype, np.floating):
        # 对于浮点数，使用 np.isclose 处理精度问题
        different_mask = ~np.isclose(data1, data2, rtol=1e-5, atol=1e-8, equal_nan=True)
    else:
        different_mask = data1 != data2
    
    num_different = np.sum(different_mask)
    num_same = total_elements - num_different
    
    print()
    print(f"  总元素数: {total_elements:,}")
    print(f"  相同元素: {num_same:,} ({num_same/total_elements*100:.4f}%)")
    print(f"  不同元素: {num_different:,} ({num_different/total_elements*100:.4f}%)")
    
    # 如果有不同，显示一些统计信息
    if num_different > 0:
        diff = np.abs(data1.astype(float) - data2.astype(float))
        valid_diff = diff[~np.isnan(diff)]
        if len(valid_diff) > 0:
            print()
            print("  差异统计:")
            print(f"    最大差异: {np.max(valid_diff):.6e}")
            print(f"    最小差异: {np.min(valid_diff[valid_diff > 0]):.6e}" if np.any(valid_diff > 0) else "    最小差异: 0")
            print(f"    平均差异: {np.mean(valid_diff):.6e}")
            print(f"    差异标准差: {np.std(valid_diff):.6e}")
    else:
        print()
        print("  ✓ 两个文件完全相同!")
    
    print("=" * 60)


def main():
    if COMPARE_MODE:
        # 对比模式
        compare_npy_files(NPY_PATH, NPY_PATH_2)
    else:
        # 查看模式
        path = Path(NPY_PATH)

        if path.is_file():
            # 单个文件
            if path.suffix == ".npy":
                data = check_npy_shape(str(path))
                if SHOW_CONTENT and data is not None and data.size <= 100:
                    print("数据内容:")
                    print(data)
            else:
                print(f"错误: {path} 不是 npy 文件")
        elif path.is_dir():
            # 目录
            if RECURSIVE:
                npy_files = list(path.rglob("*.npy"))
            else:
                npy_files = list(path.glob("*.npy"))
            
            if not npy_files:
                print(f"在 {path} 中没有找到 npy 文件")
                return
            
            print(f"找到 {len(npy_files)} 个 npy 文件:\n")
            for npy_file in sorted(npy_files):
                data = check_npy_shape(str(npy_file))
                if SHOW_CONTENT and data is not None and data.size <= 100:
                    print("数据内容:")
                    print(data)
        else:
            print(f"错误: 路径 {path} 不存在")


if __name__ == "__main__":
    main()

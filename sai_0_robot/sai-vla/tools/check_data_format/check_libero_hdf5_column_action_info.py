#!/usr/bin/env python3
"""检查LIBERO原始数据中action和state的关系"""

# python /home/sythoid_01/文档/Huangwenlong/sai0-vla/tools/check_libero_action.py \
#     --hdf5 /path/to/your/libero_file.hdf5 \
#     --demo 0 \
#     --frames 10

import h5py
import numpy as np
import argparse
from pathlib import Path


def check_action_state_relationship(hdf5_path: str, demo_idx: int = 0, num_frames: int = 5):
    """分析action和state之间的关系"""
    
    with h5py.File(hdf5_path, 'r') as f:
        demo = f['data'][f'demo_{demo_idx}']
        
        actions = demo['actions'][:]  # (T, 7)
        robot_states = demo['robot_states'][:]  # (T, 9)
        
        print(f"Actions shape: {actions.shape}")
        print(f"Robot states shape: {robot_states.shape}")
        print(f"\n{'='*80}")
        
        # State定义: [gripper_qpos_left, gripper_qpos_right, ee_pos_x, ee_pos_y, ee_pos_z, 
        #             ee_quat_w, ee_quat_x, ee_quat_y, ee_quat_z]
        # Action定义: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
        
        print("\n分析前几帧的action和state关系：")
        print("-"*80)
        
        for t in range(min(num_frames, len(actions)-1)):
            state_t = robot_states[t]
            state_t1 = robot_states[t+1]
            action_t = actions[t]
            
            # 提取ee位置 (索引 2,3,4)
            ee_pos_t = state_t[2:5]
            ee_pos_t1 = state_t1[2:5]
            
            # 计算实际delta
            actual_delta = ee_pos_t1 - ee_pos_t
            
            print(f"\n帧 {t} -> {t+1}:")
            print(f"  State ee_pos[t]:   [{ee_pos_t[0]:.6f}, {ee_pos_t[1]:.6f}, {ee_pos_t[2]:.6f}]")
            print(f"  State ee_pos[t+1]: [{ee_pos_t1[0]:.6f}, {ee_pos_t1[1]:.6f}, {ee_pos_t1[2]:.6f}]")
            print(f"  实际delta:         [{actual_delta[0]:.6f}, {actual_delta[1]:.6f}, {actual_delta[2]:.6f}]")
            print(f"  Action[0:3]:       [{action_t[0]:.6f}, {action_t[1]:.6f}, {action_t[2]:.6f}]")
            print(f"  Action[3:7]:       [{action_t[3]:.6f}, {action_t[4]:.6f}, {action_t[5]:.6f}, {action_t[6]:.2f}]")
            
            # 计算比例
            if np.any(np.abs(actual_delta) > 1e-8):
                ratio = action_t[:3] / (actual_delta + 1e-10)
                print(f"  Action/Delta比例:  [{ratio[0]:.2f}, {ratio[1]:.2f}, {ratio[2]:.2f}]")
        
        # 统计action的范围
        print(f"\n{'='*80}")
        print("Action统计信息：")
        print(f"  Action min: {actions.min(axis=0)}")
        print(f"  Action max: {actions.max(axis=0)}")
        print(f"  Action mean: {actions.mean(axis=0)}")
        print(f"  Action std: {actions.std(axis=0)}")
        
        # 检查action是否有特殊的离散值
        print(f"\n第7维(gripper)的唯一值: {np.unique(actions[:, 6])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, required=True, help="Path to LIBERO HDF5 file")
    parser.add_argument("--demo", type=int, default=0, help="Demo index")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to analyze")
    
    args = parser.parse_args()
    check_action_state_relationship(args.hdf5, args.demo, args.frames)

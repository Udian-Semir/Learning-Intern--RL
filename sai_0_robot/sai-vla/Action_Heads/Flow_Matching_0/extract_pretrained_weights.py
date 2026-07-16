"""
从 GR00T-N1.5 预训练模型中提取 Action Head 权重
"""
import argparse
import json
from pathlib import Path
from safetensors import safe_open
import torch


def extract_action_head_weights(pretrained_dir: Path, output_path: Path):
    """
    从预训练模型中提取 action_head 相关权重
    
    Args:
        pretrained_dir: 预训练模型目录
        output_path: 输出权重文件路径
    """
    # 读取索引文件找到权重分布
    index_file = pretrained_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # 找到所有 action_head 相关的权重
    action_head_weights = {}
    files_to_load = set()
    
    for key, file in weight_map.items():
        if "action_head" in key:
            files_to_load.add(file)
    
    print(f"Found {len(files_to_load)} files containing action_head weights")
    
    # 从 safetensors 文件中加载权重
    for file in files_to_load:
        file_path = pretrained_dir / file
        print(f"Loading from {file}")
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "action_head" in key:
                    # 去掉前缀 "action_head."
                    new_key = key.replace("action_head.", "")
                    action_head_weights[new_key] = f.get_tensor(key)
                    print(f"  Extracted: {key} -> {new_key}")
    
    # 保存权重
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(action_head_weights, output_path)
    print(f"\nSaved {len(action_head_weights)} weights to {output_path}")
    
    # 打印权重统计
    total_params = sum(w.numel() for w in action_head_weights.values())
    print(f"Total parameters: {total_params:,}")
    
    # 保存模型结构到txt文件（类似print(model)的输出）
    structure_file = output_path.parent / (output_path.stem + "_structure.txt")
    with open(structure_file, "w") as f:
        # 构建类似 nn.Module 的 print 输出格式
        f.write("FlowmatchingActionHead(\n")
        
        # 按模块分组显示权重
        modules = {}
        for key in sorted(action_head_weights.keys()):
            # 提取模块名（第一个点之前的部分）
            if "." in key:
                module_name = key.split(".")[0]
            else:
                module_name = "root"
            
            if module_name not in modules:
                modules[module_name] = []
            modules[module_name].append(key)
        
        # 输出每个模块
        for module_name in sorted(modules.keys()):
            if module_name == "root":
                continue
            
            keys = modules[module_name]
            # 统计该模块的参数量
            module_params = sum(action_head_weights[k].numel() for k in keys)
            
            f.write(f"  ({module_name}): Module with {len(keys)} parameters, total params: {module_params:,}\n")
            
            # 列出该模块的所有权重
            for key in keys:
                tensor = action_head_weights[key]
                f.write(f"    {key}: {tuple(tensor.shape)}\n")
        
        f.write(")\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total trainable parameters: {total_params:,}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved model structure to {structure_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default="/home/ssd/Documents/wenlonghuang/models/from_huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e",
        help="预训练模型目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pretrained_action_head.pt",
        help="输出权重文件路径"
    )
    args = parser.parse_args()
    
    pretrained_dir = Path(args.pretrained_dir)
    # 确保输出路径是相对于脚本所在目录
    script_dir = Path(__file__).parent
    output_path = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    extract_action_head_weights(pretrained_dir, output_path)


if __name__ == "__main__":
    main()

# python utils/pt2safetensors/transfer.py \
#     --input /path/to/model.pt \
#     --output /path/to/output_dir \
#     --num_shards 4

import argparse
import math
import os
from collections import OrderedDict

import torch
from safetensors.torch import save_file


def split_state_dict(state_dict: dict, num_shards: int) -> list[OrderedDict]:
    keys = list(state_dict.keys())
    total = len(keys)
    shard_size = math.ceil(total / num_shards)

    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = min(start + shard_size, total)
        shard = OrderedDict()
        for key in keys[start:end]:
            shard[key] = state_dict[key]
        shards.append(shard)
    return shards


def generate_index(shards: list[OrderedDict], filenames: list[str]) -> dict:
    weight_map = {}
    total_size = 0
    for shard, fname in zip(shards, filenames):
        for key, tensor in shard.items():
            weight_map[key] = fname
            total_size += tensor.numel() * tensor.element_size()
    return {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }


def main():
    parser = argparse.ArgumentParser(description="将 .pt 文件拆分为多个 safetensors 分片")
    parser.add_argument("--input", type=str, required=True, help="输入 .pt 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录路径")
    parser.add_argument("--num_shards", type=int, required=True, help="拆分的分片数量")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"正在加载 {args.input} ...")
    state_dict = torch.load(args.input, map_location="cpu", weights_only=True)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    print(f"共 {len(state_dict)} 个张量，拆分为 {args.num_shards} 个分片")

    shards = split_state_dict(state_dict, args.num_shards)

    filenames = []
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{args.num_shards:05d}.safetensors"
        filenames.append(fname)
        path = os.path.join(args.output, fname)
        save_file(shard, path)
        print(f"  已保存 {fname} ({len(shard)} 个张量)")

    import json
    index = generate_index(shards, filenames)
    index_path = os.path.join(args.output, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  已保存索引文件 model.safetensors.index.json")

    print("完成!")


if __name__ == "__main__":
    main()

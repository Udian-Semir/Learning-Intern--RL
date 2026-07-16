#!/usr/bin/env python3
"""
Safetensors Model Structure Analyzer
=====================================
分析 .safetensors 文件的详细模型结构并保存为 txt 文件
生成类似 PyTorch print(model) 的模型架构总结

用法:
    python check_safetensors.py --dir /path/to/safetensors/directory
    python check_safetensors.py  # 交互式输入目录路径

    python check_safetensors.py --dir /home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
    python check_safetensors.py --dir /home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/d0814e7ecb19202e7c8468b46098b0b7ef3a6d61 --output model_structure_GR00T-N1.5-3B_20251229_100000.txt
"""

import os
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, OrderedDict

try:
    from safetensors import safe_open
except ImportError:
    print("错误: 需要安装 safetensors 库")
    print("请运行: pip install safetensors")
    sys.exit(1)


def format_size(num_bytes: int) -> str:
    """格式化字节大小为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def format_params(num_params: int) -> str:
    """格式化参数数量"""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    return str(num_params)


class ModelArchitectureInferrer:
    """从张量名称和形状推断模型架构"""
    
    def __init__(self):
        self.modules = OrderedDict()  # path -> module_info
        self.tensor_registry = {}  # tensor_name -> tensor_info
    
    def add_tensor(self, name: str, shape: list, dtype: str, num_params: int):
        """添加张量并推断其所属模块"""
        self.tensor_registry[name] = {
            'shape': shape,
            'dtype': dtype,
            'num_params': num_params
        }
        
        # 解析张量路径
        parts = name.split('.')
        
        # 最后一部分通常是 weight/bias
        param_type = parts[-1] if parts else ''
        
        # 构建模块路径
        if param_type in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
            module_path = '.'.join(parts[:-1])
        else:
            module_path = '.'.join(parts)
        
        if module_path not in self.modules:
            self.modules[module_path] = {
                'tensors': [],
                'inferred_type': None,
                'params': {},
            }
        
        self.modules[module_path]['tensors'].append(name)
        self.modules[module_path]['params'][param_type] = {
            'shape': shape,
            'dtype': dtype,
            'num_params': num_params
        }
    
    def infer_layer_type(self, module_path: str, params: dict) -> str:
        """推断层类型"""
        path_lower = module_path.lower()
        
        # 获取权重形状
        weight_shape = params.get('weight', {}).get('shape', [])
        has_bias = 'bias' in params
        
        # 基于路径名称推断
        if 'embed_tokens' in path_lower or 'token_embedding' in path_lower:
            if len(weight_shape) == 2:
                return f"Embedding({weight_shape[0]}, {weight_shape[1]})"
        
        if 'position_embedding' in path_lower or 'pos_embed' in path_lower:
            if len(weight_shape) == 2:
                return f"Embedding({weight_shape[0]}, {weight_shape[1]})"
        
        if 'patch_embedding' in path_lower or 'patch_embed' in path_lower:
            if len(weight_shape) == 4:  # Conv2d
                out_c, in_c, kh, kw = weight_shape
                return f"Conv2d({in_c}, {out_c}, kernel_size=({kh}, {kw}))"
            elif len(weight_shape) == 5:  # Conv3d
                out_c, in_c, kd, kh, kw = weight_shape
                return f"Conv3d({in_c}, {out_c}, kernel_size=({kd}, {kh}, {kw}))"
        
        if 'layernorm' in path_lower or 'layer_norm' in path_lower or 'norm' in path_lower:
            if len(weight_shape) == 1:
                eps = "1e-05"
                if 'rms' in path_lower:
                    return f"RMSNorm(({weight_shape[0]},), eps={eps})"
                return f"LayerNorm(({weight_shape[0]},), eps={eps}, elementwise_affine=True)"
        
        if 'q_proj' in path_lower or 'k_proj' in path_lower or 'v_proj' in path_lower or 'o_proj' in path_lower:
            if len(weight_shape) == 2:
                return f"Linear(in_features={weight_shape[1]}, out_features={weight_shape[0]}, bias={has_bias})"
        
        if 'qkv' in path_lower:
            if len(weight_shape) == 2:
                return f"Linear(in_features={weight_shape[1]}, out_features={weight_shape[0]}, bias={has_bias})"
        
        if 'mlp' in path_lower or 'ffn' in path_lower:
            if 'gate_proj' in path_lower or 'up_proj' in path_lower or 'down_proj' in path_lower:
                if len(weight_shape) == 2:
                    return f"Linear(in_features={weight_shape[1]}, out_features={weight_shape[0]}, bias={has_bias})"
            if 'fc1' in path_lower or 'fc2' in path_lower or 'dense' in path_lower:
                if len(weight_shape) == 2:
                    return f"Linear(in_features={weight_shape[1]}, out_features={weight_shape[0]}, bias={has_bias})"
        
        if 'lm_head' in path_lower or 'output' in path_lower:
            if len(weight_shape) == 2:
                return f"Linear(in_features={weight_shape[1]}, out_features={weight_shape[0]}, bias={has_bias})"
        
        if 'q_norm' in path_lower or 'k_norm' in path_lower:
            if len(weight_shape) == 1:
                return f"RMSNorm(({weight_shape[0]},))"
        
        # 通用推断
        if len(weight_shape) == 2:
            return f"Linear(in_features={weight_shape[1]}, out_features={weight_shape[0]}, bias={has_bias})"
        elif len(weight_shape) == 4:
            out_c, in_c, kh, kw = weight_shape
            return f"Conv2d({in_c}, {out_c}, kernel_size=({kh}, {kw}))"
        elif len(weight_shape) == 1:
            return f"Parameter({weight_shape[0]})"
        
        return f"Module(tensors={len(params)})"
    
    def build_hierarchy(self) -> dict:
        """构建层级结构"""
        hierarchy = {}
        
        for module_path, info in self.modules.items():
            parts = module_path.split('.')
            current = hierarchy
            
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {
                        '_children': {},
                        '_is_leaf': False,
                        '_path': '.'.join(parts[:i+1]),
                        '_params': 0,
                    }
                current = current[part]['_children']
            
            # 标记为叶子节点
            if parts:
                temp = hierarchy
                for part in parts[:-1]:
                    temp = temp[part]['_children']
                if parts[-1] in temp:
                    temp[parts[-1]]['_is_leaf'] = True
                    temp[parts[-1]]['_type'] = self.infer_layer_type(module_path, info['params'])
                    temp[parts[-1]]['_total_params'] = sum(
                        p['num_params'] for p in info['params'].values()
                    )
        
        return hierarchy
    
    def count_repeated_layers(self, hierarchy: dict) -> dict:
        """统计重复层的数量"""
        repeated = defaultdict(list)
        
        def find_numeric_children(node: dict, parent_path: str = ""):
            for key, value in node.items():
                if key.startswith('_'):
                    continue
                
                current_path = f"{parent_path}.{key}" if parent_path else key
                
                # 检查是否是数字索引 (如 layers.0, layers.1)
                if key.isdigit():
                    # 找到父级名称
                    parent_key = parent_path.split('.')[-1] if parent_path else ""
                    repeated[parent_path].append(int(key))
                
                if '_children' in value:
                    find_numeric_children(value['_children'], current_path)
        
        find_numeric_children(hierarchy)
        return repeated
    
    def generate_architecture_string(self) -> str:
        """生成类似 PyTorch 的模型架构字符串"""
        hierarchy = self.build_hierarchy()
        repeated = self.count_repeated_layers(hierarchy)
        
        lines = []
        
        def format_node(node: dict, indent: int = 0, parent_path: str = "", skip_indices: set = None):
            if skip_indices is None:
                skip_indices = set()
            
            prefix = "  " * indent
            
            # 按键排序，数字键按数值排序
            keys = [k for k in node.keys() if not k.startswith('_')]
            
            def sort_key(k):
                if k.isdigit():
                    return (0, int(k))
                return (1, k)
            
            keys.sort(key=sort_key)
            
            for key in keys:
                value = node[key]
                current_path = f"{parent_path}.{key}" if parent_path else key
                
                # 检查是否是重复层的一部分
                if key.isdigit() and parent_path in repeated:
                    indices = sorted(repeated[parent_path])
                    if int(key) == indices[0]:
                        # 第一个，显示范围
                        if len(indices) > 1:
                            range_str = f"(0-{indices[-1]}): {len(indices)} x "
                        else:
                            range_str = "(0): "
                        
                        # 获取子模块的结构类型名称
                        child_type = self._infer_block_type(current_path)
                        lines.append(f"{prefix}{range_str}{child_type}(")
                        
                        # 递归处理子节点
                        if '_children' in value and value['_children']:
                            format_node(value['_children'], indent + 1, current_path)
                        
                        lines.append(f"{prefix})")
                        
                        # 跳过后续的数字索引
                        skip_indices.update(indices[1:])
                        continue
                    elif int(key) in skip_indices:
                        continue
                
                # 普通节点
                if value.get('_is_leaf'):
                    layer_type = value.get('_type', 'Module()')
                    total_params = value.get('_total_params', 0)
                    lines.append(f"{prefix}({key}): {layer_type}")
                else:
                    # 非叶子节点，是容器
                    container_type = self._infer_container_type(key, current_path)
                    lines.append(f"{prefix}({key}): {container_type}(")
                    
                    if '_children' in value and value['_children']:
                        format_node(value['_children'], indent + 1, current_path)
                    
                    lines.append(f"{prefix})")
        
        # 推断模型根类型
        root_type = self._infer_model_type()
        lines.append(f"{root_type}(")
        format_node(hierarchy, indent=1)
        lines.append(")")
        
        return '\n'.join(lines)
    
    def _infer_model_type(self) -> str:
        """推断模型类型"""
        paths = list(self.modules.keys())
        path_str = ' '.join(paths).lower()
        
        if 'eagle' in path_str:
            return "EagleModel"
        if 'qwen' in path_str:
            if 'vision' in path_str:
                return "Qwen2VLForConditionalGeneration"
            return "Qwen2ForCausalLM"
        if 'llama' in path_str:
            return "LlamaForCausalLM"
        if 'mistral' in path_str:
            return "MistralForCausalLM"
        if 'siglip' in path_str:
            return "SiglipVisionModel"
        
        return "Model"
    
    def _infer_container_type(self, key: str, path: str) -> str:
        """推断容器类型"""
        key_lower = key.lower()
        
        if key_lower == 'model':
            return "Model"
        if key_lower == 'backbone':
            return "Backbone"
        if key_lower in ['layers', 'blocks', 'encoder', 'decoder']:
            return "ModuleList"
        if 'vision' in key_lower:
            return "VisionModel"
        if 'language' in key_lower:
            return "LanguageModel"
        if key_lower == 'self_attn' or key_lower == 'attn' or key_lower == 'attention':
            return "Attention"
        if key_lower == 'mlp' or key_lower == 'ffn':
            return "MLP"
        if 'embed' in key_lower:
            return "Embeddings"
        
        return key.replace('_', ' ').title().replace(' ', '')
    
    def _infer_block_type(self, path: str) -> str:
        """推断块类型"""
        path_lower = path.lower()
        
        if 'encoder.layers' in path_lower or 'vision' in path_lower:
            return "VisionBlock"
        if 'decoder.layers' in path_lower:
            return "DecoderBlock"
        if 'language_model' in path_lower and 'layers' in path_lower:
            return "TransformerBlock"
        if 'blocks' in path_lower:
            return "Block"
        
        return "Block"


def analyze_safetensors_file(filepath: str) -> dict:
    """分析单个 safetensors 文件"""
    result = {
        'filename': os.path.basename(filepath),
        'filepath': filepath,
        'file_size': os.path.getsize(filepath),
        'tensors': [],
        'total_params': 0,
        'total_bytes': 0,
        'dtype_stats': defaultdict(lambda: {'count': 0, 'params': 0, 'bytes': 0}),
    }
    
    try:
        with safe_open(filepath, framework="pt") as f:
            metadata = f.metadata()
            result['metadata'] = metadata if metadata else {}
            
            for key in f.keys():
                tensor = f.get_tensor(key)
                shape = list(tensor.shape)
                dtype = str(tensor.dtype).replace('torch.', '')
                num_params = tensor.numel()
                
                # 计算字节大小
                if 'float16' in dtype.lower() or 'half' in dtype.lower():
                    dtype_bytes = 2
                elif 'bfloat16' in dtype.lower():
                    dtype_bytes = 2
                elif 'float32' in dtype.lower() or 'float' == dtype.lower():
                    dtype_bytes = 4
                elif 'float64' in dtype.lower() or 'double' in dtype.lower():
                    dtype_bytes = 8
                elif 'int8' in dtype.lower():
                    dtype_bytes = 1
                elif 'int16' in dtype.lower():
                    dtype_bytes = 2
                elif 'int32' in dtype.lower() or 'int' == dtype.lower():
                    dtype_bytes = 4
                elif 'int64' in dtype.lower() or 'long' in dtype.lower():
                    dtype_bytes = 8
                else:
                    dtype_bytes = tensor.element_size() if hasattr(tensor, 'element_size') else 4
                
                tensor_bytes = num_params * dtype_bytes
                
                tensor_info = {
                    'name': key,
                    'shape': shape,
                    'dtype': dtype,
                    'num_params': num_params,
                    'size_bytes': tensor_bytes,
                }
                
                result['tensors'].append(tensor_info)
                result['total_params'] += num_params
                result['total_bytes'] += tensor_bytes
                
                result['dtype_stats'][dtype]['count'] += 1
                result['dtype_stats'][dtype]['params'] += num_params
                result['dtype_stats'][dtype]['bytes'] += tensor_bytes
                
    except Exception as e:
        result['error'] = str(e)
    
    return result


def analyze_directory(directory: str) -> list:
    """分析目录中的所有 safetensors 文件"""
    safetensors_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.safetensors'):
                safetensors_files.append(os.path.join(root, file))
    
    safetensors_files.sort()
    
    results = []
    for filepath in safetensors_files:
        print(f"正在分析: {filepath}")
        result = analyze_safetensors_file(filepath)
        results.append(result)
    
    return results


def generate_model_architecture(results: list) -> str:
    """生成模型架构总结"""
    inferrer = ModelArchitectureInferrer()
    
    # 收集所有张量
    for result in results:
        for tensor in result['tensors']:
            inferrer.add_tensor(
                tensor['name'],
                tensor['shape'],
                tensor['dtype'],
                tensor['num_params']
            )
    
    return inferrer.generate_architecture_string()


def generate_report(results: list, directory: str) -> str:
    """生成详细报告"""
    lines = []
    
    # 报告头部
    lines.append("=" * 100)
    lines.append("SAFETENSORS 模型结构分析报告")
    lines.append("=" * 100)
    lines.append(f"分析目录: {directory}")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"文件数量: {len(results)}")
    lines.append("")
    
    # 汇总统计
    total_params_all = sum(r['total_params'] for r in results)
    total_bytes_all = sum(r['total_bytes'] for r in results)
    total_tensors = sum(len(r['tensors']) for r in results)
    
    lines.append("-" * 100)
    lines.append("总体统计")
    lines.append("-" * 100)
    lines.append(f"  总张量数量: {total_tensors:,}")
    lines.append(f"  总参数数量: {total_params_all:,} ({format_params(total_params_all)})")
    lines.append(f"  总模型大小: {format_size(total_bytes_all)}")
    lines.append("")
    
    # 汇总 dtype 统计
    all_dtype_stats = defaultdict(lambda: {'count': 0, 'params': 0, 'bytes': 0})
    for r in results:
        for dtype, stats in r['dtype_stats'].items():
            all_dtype_stats[dtype]['count'] += stats['count']
            all_dtype_stats[dtype]['params'] += stats['params']
            all_dtype_stats[dtype]['bytes'] += stats['bytes']
    
    lines.append("数据类型分布:")
    for dtype, stats in sorted(all_dtype_stats.items()):
        lines.append(f"  {dtype}:")
        lines.append(f"    张量数: {stats['count']:,}")
        lines.append(f"    参数数: {stats['params']:,} ({format_params(stats['params'])})")
        lines.append(f"    大小: {format_size(stats['bytes'])}")
    lines.append("")
    
    # ========== 模型架构总结 ==========
    lines.append("=" * 100)
    lines.append("模型架构 (Model Architecture)")
    lines.append("=" * 100)
    lines.append("")
    
    architecture_str = generate_model_architecture(results)
    lines.append(architecture_str)
    lines.append("")
    
    # ========== 层级参数统计 ==========
    lines.append("=" * 100)
    lines.append("层级参数统计 (Layer Parameters)")
    lines.append("=" * 100)
    
    # 统计每个主要模块的参数
    module_params = defaultdict(lambda: {'count': 0, 'params': 0, 'bytes': 0})
    
    for result in results:
        for tensor in result['tensors']:
            parts = tensor['name'].split('.')
            # 取前3级作为模块
            if len(parts) >= 3:
                module_key = '.'.join(parts[:3])
            elif len(parts) >= 2:
                module_key = '.'.join(parts[:2])
            else:
                module_key = parts[0]
            
            module_params[module_key]['count'] += 1
            module_params[module_key]['params'] += tensor['num_params']
            module_params[module_key]['bytes'] += tensor['size_bytes']
    
    lines.append("")
    lines.append(f"{'模块':<60} {'张量数':<10} {'参数量':<20} {'大小':<15}")
    lines.append("-" * 100)
    
    for module, stats in sorted(module_params.items(), key=lambda x: -x[1]['params']):
        lines.append(
            f"{module:<60} {stats['count']:<10} "
            f"{stats['params']:>15,} ({format_params(stats['params']):>8}) "
            f"{format_size(stats['bytes']):>12}"
        )
    
    lines.append("")
    
    # ========== 每个文件的详细信息 ==========
    for idx, result in enumerate(results, 1):
        lines.append("=" * 100)
        lines.append(f"文件 [{idx}/{len(results)}]: {result['filename']}")
        lines.append("=" * 100)
        lines.append(f"路径: {result['filepath']}")
        lines.append(f"文件大小: {format_size(result['file_size'])}")
        lines.append(f"张量数量: {len(result['tensors']):,}")
        lines.append(f"总参数量: {result['total_params']:,} ({format_params(result['total_params'])})")
        lines.append(f"模型大小: {format_size(result['total_bytes'])}")
        
        if result.get('metadata'):
            lines.append("")
            lines.append("元数据:")
            for k, v in result['metadata'].items():
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                lines.append(f"  {k}: {v_str}")
        
        if result.get('error'):
            lines.append(f"错误: {result['error']}")
            continue
        
        lines.append("")
        lines.append("-" * 100)
        lines.append("张量详情:")
        lines.append("-" * 100)
        lines.append(f"{'序号':<8} {'名称':<70} {'形状':<30} {'类型':<12} {'参数量':<15} {'大小':<12}")
        lines.append("-" * 100)
        
        for i, tensor in enumerate(result['tensors'], 1):
            name = tensor['name']
            if len(name) > 68:
                name = "..." + name[-65:]
            shape_str = str(tensor['shape'])
            if len(shape_str) > 28:
                shape_str = shape_str[:25] + "..."
            
            lines.append(
                f"{i:<8} {name:<70} {shape_str:<30} {tensor['dtype']:<12} "
                f"{tensor['num_params']:<15,} {format_size(tensor['size_bytes']):<12}"
            )
        
        lines.append("")
    
    # 报告尾部
    lines.append("=" * 100)
    lines.append("报告结束")
    lines.append("=" * 100)
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='分析 .safetensors 文件的模型结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    python check_safetensors.py --dir /path/to/model
    python check_safetensors.py -d /home/user/.cache/huggingface/hub/models--xxx/snapshots/xxx
        '''
    )
    parser.add_argument(
        '-d', '--dir',
        type=str,
        help='包含 .safetensors 文件的目录路径'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出文件名 (默认: model_structure_<timestamp>.txt)'
    )
    
    args = parser.parse_args()
    
    # 获取目录路径
    if args.dir:
        directory = args.dir
    else:
        directory = input("请输入包含 .safetensors 文件的目录路径: ").strip()
    
    # 验证目录
    if not os.path.exists(directory):
        print(f"错误: 目录不存在: {directory}")
        sys.exit(1)
    
    if not os.path.isdir(directory):
        print(f"错误: 路径不是目录: {directory}")
        sys.exit(1)
    
    print(f"\n开始分析目录: {directory}")
    print("-" * 50)
    
    # 分析文件
    results = analyze_directory(directory)
    
    if not results:
        print("错误: 在指定目录中未找到 .safetensors 文件")
        sys.exit(1)
    
    # 生成报告
    report = generate_report(results, directory)
    
    # 确定输出路径 (保存在脚本所在目录)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.output:
        output_filename = args.output
    else:
        dir_name = os.path.basename(os.path.normpath(directory))
        if not dir_name or dir_name == '.':
            dir_name = "model"
        dir_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in dir_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"model_structure_{dir_name}_{timestamp}.txt"
    
    output_path = os.path.join(script_dir, output_filename)
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("-" * 50)
    print(f"\n分析完成!")
    print(f"  文件数量: {len(results)}")
    print(f"  总张量数: {sum(len(r['tensors']) for r in results):,}")
    print(f"  总参数量: {sum(r['total_params'] for r in results):,}")
    print(f"  报告已保存至: {output_path}")


if __name__ == "__main__":
    main()

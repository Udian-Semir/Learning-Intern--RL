# Sai0-VLA 代码交接文档

> 项目: **Sai0-VLA (Vision-Language-Action Model)**
> 仓库根目录: `/home/dev/文档/huangwenlong/sai0-vla`
> 文档更新日期: 2026-04
> 文档目的: 向下一位维护者完整移交整个代码库 — 描述**代码结构、模块职责、模块间串联关系、训练/推理/部署/评估全链路、关键配置入口**。

---

## 目录

1. [项目定位与核心思想](#1-项目定位与核心思想)
2. [顶层目录速览](#2-顶层目录速览)
3. [系统架构全景图](#3-系统架构全景图)
4. [核心抽象：三层解耦设计](#4-核心抽象三层解耦设计)
5. [模块逐层剖析](#5-模块逐层剖析)
   - 5.1 [`VLMs/S0_1`（视觉-语言 backbone）](#51-vlmss0_1--vlm-backbone)
   - 5.2 [`Action_Heads`（动作头）](#52-action_heads--动作头)
   - 5.3 [`Adapter/Pons`（适配器）](#53-adapterpons--特征压缩适配器)
   - 5.4 [`VLAs/Sai0_1`（端到端编排层）](#54-vlassai0_1--端到端编排层)
   - 5.5 [`utils`（数据/特征工具链）](#55-utils--数据与特征工具链)
   - 5.6 [`deployment`（部署与推理服务）](#56-deployment--http-推理服务)
   - 5.7 [`eval`（策略闭环评估）](#57-eval--闭环评估)
   - 5.8 [`tools`（辅助脚本）](#58-tools--辅助脚本)
6. [三大数据/特征流](#6-三大数据特征流)
7. [训练全流程串联](#7-训练全流程串联)
8. [推理与部署全流程串联](#8-推理与部署全流程串联)
9. [评估全流程串联](#9-评估全流程串联)
10. [关键配置点与默认值](#10-关键配置点与默认值)
11. [模块依赖矩阵](#11-模块依赖矩阵)
12. [常见运维/接手注意事项](#12-常见运维接手注意事项)
13. [文件级速查表](#13-文件级速查表)

---

## 1. 项目定位与核心思想

Sai0-VLA 是一个**视觉-语言-动作（VLA）机器人策略框架**，目标是在给定
`(多视角图像, 语言指令, 机器人本体状态)` 的输入下，输出机器人未来 `N` 步的动作序列（action chunking）。

它的架构哲学是 **"VLM 提供语义特征 + Action Head 解码成动作"** 的两段式解耦，并在其上再封装端到端的统一推理/训练接口：

- **VLM Backbone**（前端感知）：由 Qwen3-VL / Eagle 2.5 VL / Cosmos-Reason-2B 等通用多模态大模型承担，**冻结不训练**，只抽取某些中间/末尾层的 hidden states 作为视觉-语言联合特征。
- **Action Head**（动作解码器）：有 Flow Matching、OFT (L1 Regression)、ParaCAT（离散化三分类）等多种可插拔实现，**这才是真正训练的对象**。
- **Sai0 端到端层**：负责把 VLM 和 Action Head 粘合成一个统一模型（训练/推理/配置/权重加载）。
- **数据层**：统一使用 **LeRobot 标准格式** 的数据集，VLM hidden states 可以**离线预提取**（大幅加速训练）或**在线实时提取**（部署时使用）。

> ⚠️ **目录版本说明**：仓库中同时存在两套版本命名：
> - 旧版：`VLAs/Sai0`、`VLMs/S0`、`Action_Heads/Flow_Matching_{0,1}`（README 描述的版本）
> - 新版：`VLAs/Sai0_1`、`VLMs/S0_1`（实际代码里使用的版本，已统一迁移）
>
> **当前代码实际使用的是 `_1` 版本**。`README.md` 顶部的项目结构示意图是旧的，**以本文档为准**。

---

## 2. 顶层目录速览

```
sai0-vla/
├── README.md                        旧版使用说明（部分已过期）
├── HANDOVER.md                      ← 本文档
├── environment.yml / install_env.sh 环境安装（conda 环境名: qwen_eagle_hwl）
├── requirements.txt                 pip 依赖（与 conda 环境叠加）
├── start_server.sh                  一键启动推理服务器的 shell 脚本
├── .gitignore
│
├── VLMs/
│   └── S0_1/                        VLM Backbone 子系统
│       └── backbone/
│           ├── model_selector.py    统一工厂 + CLI 入口 + HiddenStateExtractor
│           ├── qwen3_vl/            Qwen3-VL 后端
│           ├── eagle2_5_vl/         Eagle 2.5 (GR00T-N1.5-3B) 后端
│           └── cosmos_reason_2b_vl/ Cosmos Reason 2B 后端
│
├── Action_Heads/
│   ├── Flow_Matching_0/             GR00T N1.5 原始 Flow Matching 头（加载官方预训练权重微调）
│   ├── Flow_Matching_1/             自定义 Flow Matching 头（支持任意 VLM 维度 / 多层 hidden）
│   ├── OFT1_0/                      OFT 头（Transformer + L1 回归 或 Diffusion）
│   ├── ParaCAT/                     离散化三分类动作头（配合 Pons Adapter）
│   ├── batch_train.sh               批量训练调度脚本
│   └── batch_train_scheduled.sh     定时批量训练脚本
│
├── Adapter/
│   └── Pons/                        Pons Adapter（把多层 VLM seq 压缩成固定长度 Query token）
│
├── VLAs/
│   └── Sai0_1/                      端到端 VLA 模型（核心编排层）
│       ├── config.py                Sai0Config/VLMConfig/ActionHeadConfig/DataConfig/TrainingConfig
│       ├── sai0_model.py            Sai0Model（把 VLM + Action Head 组合起来的 nn.Module）
│       ├── data_utils.py            训练数据流水线封装
│       └── inference.py             Sai0Inference / RealtimeInference 推理接口
│
├── utils/
│   ├── lerobot_dataset_loader.py    LeRobot 数据集通用加载器（非常核心，~2000 行）
│   ├── discrete.py                  动作离散化/反离散化（ParaCAT 用）
│   ├── hand_processor.py            手部数据处理
│   ├── euler_unwrap.py              欧拉角解缠绕
│   ├── polyfit_chunk.py             动作轨迹多项式拟合
│   ├── extract_vlm_hidden_state/    VLM hidden states 离线提取工具（含 CLI & 多 GPU 脚本）
│   ├── dataset_format_transform/    数据集格式转换脚本
│   ├── download/                    数据集下载脚本
│   ├── pt2safetensors/              权重格式转换
│   ├── convert_libero_plus_to_lerobot_v2.py  LIBERO++ → LeRobot v2 转换
│   ├── migrate_vlm_npy_to_chunk_npz.py       单帧 .npy → chunk .npz 迁移工具
│   └── webdataset_utils.py          WebDataset 支持
│
├── deployment/
│   ├── Sai0_1_server/               主推理服务器（FastAPI + 队列 + API Key + 限流）
│   ├── tele/                        ParaCAT 遥操作推理服务
│   ├── tele_parad/                  Pons+ParaCAT 遥操作服务（含累积误差分析）
│   └── tele_parad_new/              新版 VLM + OFT 遥操作服务（兼容旧客户端 API）
│
├── eval/
│   ├── libero/                      单独针对不同 Action Head 的 LIBERO 评估脚本（旧）
│   │   ├── Flow_Matching_0/
│   │   ├── Flow_Matching_1/
│   │   ├── N1_5/                    评估 GR00T N1.5 官方 checkpoint
│   │   └── OFT1_0/
│   └── Sai0_1/                      基于 Sai0_1 统一模块的评估（新）
│       ├── libero/                  LIBERO 仿真闭环
│       │   ├── Flow_Matching_0/
│       │   ├── Flow_Matching_1/
│       │   ├── OFT1_0/
│       │   └── ParaCAT/
│       └── tele/                    真机遥操作评估
│           └── ParaCAT/
│
└── tools/                           大量独立分析脚本（错误曲线、轨迹绘图、动作分析等）
    ├── calc_discrete/                离散化基线/误差/跟踪误差计算（calc_discrete_*.py）
    ├── check_data_format/            数据格式校验（check_libero_hdf5 / check_parquet / check_npy / check_safetensors）
    ├── plot_action/                  动作/状态可视化（plot_action_columns / plot_action_polyfit / plot_state_columns / plot_state_action_compare / plot_3d_trajectory）
    ├── evaluate_checkpoint/          checkpoint 轨迹评估（evaluate_oft_action_trajectory / plot_cumulative_error_per_dim）
    ├── openloop_eval/                开环评估（openloop_eval.py + run.sh）
    ├── model_structures/             Eagle 等模型结构 dump
    ├── outputs/                      所有脚本默认产物目录（PNG/NPZ）
    └── test_oft_cumulative_error.py  OFT 端到端累积误差测试
    ├── check_*.py                   数据格式 sanity check（parquet / hdf5 / npy / safetensors）
    └── test_oft_cumulative_error.py OFT 累计误差测试
```

---

## 3. 系统架构全景图

### 3.1 分层架构图

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        用户 / 客户端 / 仿真环境                                │
│     LIBERO 仿真环境 | 真实机械臂 | Python SDK | curl / HTTP / WebSocket        │
└──────────────────┬───────────────────────────────────────────┬───────────────┘
                   │                                           │
        HTTP 推理请求                                  本地 Python 调用
                   │                                           │
┌──────────────────▼───────────────────┐       ┌───────────────▼───────────────┐
│   deployment/ (5 个 FastAPI 服务)    │       │   eval/  (闭环评估脚本)         │
│   - Sai0_1_server  生产推理服务      │       │   - LIBERO (libero_spatial/    │
│   - tele           ParaCAT 遥操作    │       │     object / goal / 10 / 90)  │
│   - tele_parad     Pons+ParaCAT      │       │   - 真机遥操作                 │
│   - tele_parad_new VLM+OFT 兼容服务  │       │                                │
└──────────────────┬───────────────────┘       └───────────────┬───────────────┘
                   │                                           │
                   └─────────────┬─────────────────────────────┘
                                 │
                                 ▼
                 ╔═══════════════════════════════════╗
                 ║        VLAs/Sai0_1   (编排层)      ║
                 ║  ┌─────────────────────────────┐  ║
                 ║  │ Sai0Model (nn.Module)       │  ║
                 ║  │  ├─ vlm_backbone (lazy)     │  ║
                 ║  │  └─ action_head  (lazy)     │  ║
                 ║  │                             │  ║
                 ║  │ Sai0Config: 统一配置        │  ║
                 ║  │  ├─ VLMConfig               │  ║
                 ║  │  ├─ ActionHeadConfig        │  ║
                 ║  │  ├─ DataConfig              │  ║
                 ║  │  └─ TrainingConfig          │  ║
                 ║  │                             │  ║
                 ║  │ Sai0Inference / Realtime…   │  ║
                 ║  │ create_sai0_dataloader(…)   │  ║
                 ║  └─────────────────────────────┘  ║
                 ╚════════╦════════════════╦═════════╝
                          │                │
                ┌─────────▼────────┐  ┌────▼──────────────────────┐
                │  VLMs/S0_1        │  │   Action_Heads             │
                │  ┌──────────────┐ │  │  ┌─────────────────────┐  │
                │  │ model_selector│ │  │  │ Flow_Matching_0     │  │
                │  │ (工厂+CLI)    │ │  │  │ (GR00T N1.5 原生)  │  │
                │  └──────┬───────┘ │  │  └─────────────────────┘  │
                │         │         │  │  ┌─────────────────────┐  │
                │  ┌──────▼────────┐│  │  │ Flow_Matching_1     │  │
                │  │ qwen3_vl      ││  │  │ (自定义 VLM 维度)  │  │
                │  │ eagle2_5_vl   ││  │  └─────────────────────┘  │
                │  │ cosmos_reason ││  │  ┌─────────────────────┐  │
                │  └───────────────┘│  │  │ OFT1_0              │  │
                │   冻结，只抽特征   │  │  │ (Transformer+L1/   │  │
                └─────────────▲─────┘  │  │  Diffusion)         │  │
                              │        │  └─────────┬───────────┘  │
                              │        │            │              │
                              │        │  ┌─────────▼───────────┐  │
                              │        │  │ ParaCAT             │  │
                              │        │  │ (三分类离散动作)   │  │
                              │        │  └────────▲────────────┘  │
                              │        │           │ 通常前置      │
                              │        │  ┌────────▼────────────┐  │
                              │        │  │ Adapter/Pons        │  │
                              │        │  │ (压缩 VLM seq)     │  │
                              │        │  └─────────────────────┘  │
                              │        └───────────────────────────┘
                              │
                ┌─────────────▼──────────────────────────────────┐
                │                    utils/                       │
                │  ┌──────────────────────────────────────────┐  │
                │  │ lerobot_dataset_loader.py                │  │
                │  │  → 读取 LeRobot parquet+mp4+npy/npz      │  │
                │  │  → 输出 vlm_hidden_states / state /      │  │
                │  │     action / images                      │  │
                │  └──────────────────────────────────────────┘  │
                │  ┌──────────────────────────────────────────┐  │
                │  │ extract_vlm_hidden_state/                │  │
                │  │  (qwen / eagle / cosmos 离线提取器)      │  │
                │  └──────────────────────────────────────────┘  │
                │  + discrete / euler / polyfit / webdataset     │
                └───────────────────────▲────────────────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     │         LeRobot 格式数据集           │
                     │  data/chunk-XXX/*.parquet            │
                     │  videos/chunk-XXX/*.mp4              │
                     │  vlm_hidden_states/*.npy or .npz     │
                     │  meta/info.json / stats.json /       │
                     │       tasks.jsonl / episodes.jsonl   │
                     └──────────────────────────────────────┘
```

### 3.2 一次推理请求的数据流

```
┌──────────┐   ①POST /v1/act     ┌──────────────────────────────┐
│ Client   │ ───────────────────▶│  deployment/Sai0_1_server    │
│(机械臂/  │     {images, state, │  - 鉴权 / 限流 / 队列          │
│ 仿真器)  │      prompt}        │  - 图像预处理 (resize/flip)   │
└──────────┘                     │  - state 预处理 (归一化/0→-1)│
                                 └────────────┬─────────────────┘
                                              │②
                                              ▼
                             ┌─────────────────────────────────┐
                             │   VLAs/Sai0_1/RealtimeInference │
                             │   → model.predict(...)           │
                             └────────────┬────────────────────┘
                                          │③
                           ┌──────────────┴──────────────┐
                           │④                            │⑤
                           ▼                             ▼
                ┌────────────────────┐      ┌────────────────────────┐
                │ VLMs/S0_1          │      │ Action_Heads/...       │
                │ backbone.get_      │      │ (Flow Matching /       │
                │ hidden_states()    │ ───▶ │  OFT /                 │
                │                    │ List │  ParaCAT)              │
                │ List[(B,S,D)]      │ of   │                        │
                │ 按 layers 选层      │ Tensor│ 结合 state →          │
                └────────────────────┘      │ Actions (N,action_dim) │
                                            └───────────┬────────────┘
                                                        │⑥
                                                        ▼
                                 ┌────────────────────────────────────┐
                                 │  服务器层 action 反归一化（stats.json）│
                                 │  → JSON 响应 {actions, timing,…}   │
                                 └──────────────────┬─────────────────┘
                                                    │⑦
┌──────────┐   ⑧响应 JSON                           │
│ Client   │ ◀──────────────────────────────────────┘
└──────────┘
```

---

## 4. 核心抽象：三层解耦设计

整个代码库的可读性依赖于下面这三个**稳定接口契约**，请优先理解它们：

### 4.1 `HiddenStateOutput`（VLM → Action Head 的数据契约）

**定义位置**：`VLMs/S0_1/backbone/qwen3_vl/backbone.py`（每个 backbone 子包内都有同构定义）

```python
@dataclass
class HiddenStateOutput:
    hidden_states: List[torch.Tensor]   # 每层 (B, seq_len, hidden_dim)
    layer_indices: List[int]            # 对应层号
    metadata: Dict[str, Any]            # token 切分等信息
    # 属性: num_layers / hidden_dim / seq_len
    # 方法: to_stacked_tensor() → (num_layers, B, seq_len, hidden_dim)
```

**核心方法**：`backbone.get_hidden_states(images, instruction) -> HiddenStateOutput`

所有下游 Action Head 都只认这一个输入。

### 4.2 `Sai0Config`（配置契约）

**定义位置**：`VLAs/Sai0_1/config.py`

```python
Sai0Config
 ├── vlm:          VLMConfig          (model_type / model_path / layers / dtype / prompt…)
 ├── action_head:  ActionHeadConfig   (head_type / 维度 / 预训练权重路径)
 ├── data:         DataConfig         (dataset_path / batch / action_chunks / image_keys)
 ├── training:     TrainingConfig     (lr / steps / amp / wandb …)
 └── mode:         "train" | "eval" | "inference"
```

工厂类方法：
- `Sai0Config.for_qwen3_libero(dataset_path, pretrained_weights=...)`
- `Sai0Config.for_eagle_libero(dataset_path, pretrained_weights=...)`
- `VLMConfig.for_qwen3_vl_2b()` / `VLMConfig.for_eagle2_5_vl()`
- `ActionHeadConfig.for_flow_matching_0/1()` / `ActionHeadConfig.for_oft()`

### 4.3 LeRobot 批次契约（数据 → 模型的数据契约）

**定义位置**：`utils/lerobot_dataset_loader.py` 的 `collate_fn`

```python
batch = {
    "images": {                          # （训练时通常不用，除非实时 VLM）
        "observation.images.<cam>": (B, H, W, 3) float32 [0,255]
    },
    "vlm_hidden_states": (B, num_layers, seq_len, hidden_dim),   # 预提取
    "observation_state": (B, state_dim),
    "actions": (B, num_action_chunks, action_dim),
    "task_description": List[str],
    "episode_index" / "frame_index" / "vlm_index": (B,)
}
```

这是所有训练代码使用的统一 batch 格式。任何 Action Head 的 `train_multigpu.py` 都会把它转换成自己需要的 `BatchFeature`。

---

## 5. 模块逐层剖析

### 5.1 `VLMs/S0_1` — VLM Backbone

职责：为任意上游调用者提供 **"图像+指令 → 多层 hidden states"** 的统一接口。

#### 目录结构

```
VLMs/S0_1/
├── __init__.py                      暴露 create_backbone 等常用 API
└── backbone/
    ├── __init__.py
    ├── model_selector.py            ← 🔑 统一工厂 + CLI + HiddenStateExtractor (离线提取)
    ├── qwen3_vl/
    │   ├── backbone.py              Qwen3VLBackbone 主类（705 行）
    │   ├── config.py                Qwen3VLConfig 数据类 + YAML 加载
    │   └── prompt_config.yaml       prompt 模板、content_order、flip 等默认值
    ├── eagle2_5_vl/
    │   ├── backbone.py              Eagle25VLBackbone（967 行，支持 thumbnail/multi-tile）
    │   ├── config.py
    │   ├── ori/                     Eagle 官方模型相关代码的镜像
    │   └── prompt_config.yaml
    └── cosmos_reason_2b_vl/
        ├── backbone.py              CosmosReason2BVLBackbone（1208 行）
        ├── config.py
        ├── ori/                     Eagle-Block2A-2B-v2 模型文件
        └── prompt_config.yaml
```

#### 关键接口

```python
from VLMs.S0_1.backbone import create_vlm_backbone

backbone = create_vlm_backbone(
    model_type="qwen3_vl",              # 或 "eagle2_5_vl" / "cosmos_reason_2b_vl"
    model_path="Qwen/Qwen3-VL-2B-Instruct",
    device="cuda:0",
    layers=[14],                        # 要抽的 transformer 层号（1-based）
    prompt_template="action",           # 或 "simple" / "detailed" / 自定义模板
    content_order="images_first",       # 图像/文本排列顺序
    flip_images=True,                   # 某些仿真环境图像颠倒，这里翻回来
    dtype="bfloat16",
)

out = backbone.get_hidden_states(
    images=[pil_image_1, pil_image_2],
    instruction="pick up the red apple",
)
# out.hidden_states: List[Tensor(1, S, D)]
# out.to_stacked_tensor(): Tensor(num_layers, 1, S, D)
```

#### 层号索引陷阱（⚠️ 接手必读）

详见 `utils/extract_vlm_hidden_state/README.md`。以 Qwen3-VL-2B 为例：

| 用户输入 `--layers` | `hidden_states` 下标 | 实际 transformer 层 |
|---|---|---|
| `1` | `[1]` | embedding 后的第 0 层 |
| `14` | `[14]` | 第 13 层 |
| `28` | `[28]` | 最后一层（layer 27）|

因为 `hidden_states[0]` 是 embedding 输出，所以**用户输入的层号 = transformer 层号 + 1**。Eagle 则支持 `-1` 表示最后一层。

#### `model_selector.py` 的双重身份

这个文件同时做了两件事：

1. **Python 工厂**：`create_vlm_backbone(...)` → 统一构造 backbone（被 `Sai0Model` 调用）。
2. **CLI 离线提取工具**：支持两种 mode：
   - `--mode pipeline`：只创建 backbone（供调试 / import 测试用）
   - `--mode hidden_state`：内部使用 `HiddenStateExtractor` 遍历 LeRobot 数据集，按 **per-frame** (`hidden_state_XXXXXX.npy`) 或 **per-episode → chunk.npz** 格式保存所有帧的 hidden states。

`HiddenStateExtractor` 关键点：
- 多线程预加载 (`_loader_worker` + `ThreadPoolExecutor`) 解耦 IO/推理。
- 断点续传：已存在的文件自动跳过。
- chunk-npz 模式：每 `chunks_size` 个 episode 打包成一个 `chunk-XXX.npz`，降低小文件数。

### 5.2 `Action_Heads` — 动作头

一个模块就是**一种动作解码方法**。每个子模块结构高度同构：

```
<Head>/
├── models/action_head/*.py          纯 PyTorch 模型代码
├── config.py                         模型超参配置
├── constants.py  (OFT 专用)          全局常量
├── train_multigpu.py                 DDP 训练脚本（2000+ 行，含数据、loss、保存逻辑）
├── infer_once.py                     最小推理示例（用于单元验证）
├── scripts/train/{qwen,eagle,cosmos}/*.sh   按 VLM 分类的训练启动脚本
└── README.md                         该 head 的架构与训练说明
```

#### 5.2.1 `Flow_Matching_0` —— GR00T N1.5 原始架构

| 关键点 | 值 |
|---|---|
| 动作维度 padding | `max_action_dim=32`, `max_state_dim=64`（固定，不可改，否则无法加载预训练权重）|
| Action Horizon | 16 |
| Backbone 维度 | `input_embedding_dim=1536`, `backbone_embedding_dim=2048` |
| 噪声 schedule | Beta(α=1.5, β=1.0) |
| 推理步数 | `num_inference_timesteps=4` |
| 核心类 | `FlowmatchingActionHead` 在 `models/action_head/flow_matching_action_head.py` |
| 工具脚本 | `extract_pretrained_weights.py` 从 GR00T-N1.5-3B 官方权重中提取 action head 部分 |
| 训练脚本 | `train_with_pretrained_action_head_weight_multigpu.py`（2314 行）|

数据流（训练）：

```
VLM hidden states (B, num_layers, S, 2048)
    └─ 取最后一层 ──► backbone_features (B, S, 2048)
state (B, state_dim) ── pad ──► (B, 1, 64)       quat→axisangle (如需)
action (B, 16, action_dim) ── pad ──► (B, 16, 32)

┌─────────────────────── FlowmatchingActionHead ────────────────────────┐
│  state_encoder / action_encoder (带 sin-pos-embed)                     │
│    + future tokens (可学习)                                             │
│     → concat: [1 state, n future, 16 action] tokens                    │
│                                                                        │
│  DiT (16 layers, 32 heads, cross-attn to VL) + interleave self-attn    │
│  VL self-attention (4 层，对 backbone_features 做自注意力)              │
│  category_specific_mlp 按 embodiment_id=31 解码                         │
│                                                                        │
│  noise_scheduler: Beta(1.5, 1.0) 预测 velocity                         │
└────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.2 `Flow_Matching_1` —— 自定义 Flow Matching（推荐主力）

与 `Flow_Matching_0` 同构，但把 VLM 维度参数化：

- `vlm_output_dim`、`action_backbone_dim`、`vl_self_attention_head_dim/num_heads` 全部可配
- 可以直接接不同 VLM（Qwen 2B=1536 / 4B=2560 / 7B=3584 / Eagle=2048）
- `merge_hidden_states.py` 提供了把多层 hidden states 在 seq 维度拼接的工具
- 训练脚本：`train_multigpu.py`（2344 行）

#### 5.2.3 `OFT1_0` —— Optimal Flow Transport 风格

基于 `vlm2oft_pipeline.py`（见 `Action_Heads/OFT1_0/vlm2oft_pipeline.py`）：

```
VLM hidden layers: List[L × (B, S, D)]
            │ concat(dim=1)
            ▼
      (B, L·S, D)
            │ concat 上 ProprioProjector(proprio)  (B, 1, D)
            ▼
      (B, L·S + 1, D)
            │ Transformer × N (默认 4)
            ▼
      (B, L·S + 1, D)
            │ 取最后一个位置（proprio token）→ (B, D)
            ▼
      L1RegressionActionHead (MLP: D → hidden_dim → NUM_CHUNKS·ACTION_DIM)
            ▼
      (B, 1, NUM_CHUNKS·ACTION_DIM) → reshape (B, NUM_CHUNKS, ACTION_DIM)
```

关键常量（`Action_Heads/OFT1_0/constants.py`，在运行时由 `Sai0Config` 覆盖）：
- `LLM_OUTPUT_DIM_MLP_INPUT_DIM`, `NUM_VLM_HIDDEN_LAYERS`, `PROPRIO_DIM`, `ACTION_DIM`, `NUM_ACTIONS_CHUNK`, `USE_DIFFUSION`

⚠️ **OFT 是目前部署默认选型**，见 `deployment/Sai0_1_server/config.yaml`。

#### 5.2.4 `ParaCAT` —— 离散三分类动作头

完整架构图见 `Action_Heads/ParaCAT/architecture.md`。要点：

1. 前置 **Pons Adapter**（见 5.3）把多层 VLM `List[(B, S, D)]` 压成 `(B, Q, D)`。
2. **ParaCAT Head**：可学习 action query `(1, chunk×action_dim, D)` → cross-attn(Pons 输出) → self-attn × M → MLP → `(B, chunk, action_dim, 3)`。
3. 3 分类输出：`{后退, 不动, 前进} = {-1, 0, +1}`，再 × delta 得到连续动作，配合 `utils/discrete.py` 反离散化。
4. 两个训练入口：
   - `train_multigpu_only_paracat.py` 只训 ParaCAT
   - `train_multigpu_pons_paracat.py` 联合训练 Pons + ParaCAT

### 5.3 `Adapter/Pons` — 特征压缩适配器

```
Adapter/Pons/
├── pons_adapter.py       PonsAdapter + PonsCrossAttentionBlock
├── train_multigpu.py     独立训练（作为特征压缩器）
└── scripts/train_pons.sh
```

**核心**：可学习的 `query tokens (1, Q, D)` + 位置编码 → 对 merged VLM states 做 Cross-Attention（N 个 block）→ `(B, Q, D)`。

把 **可变长度、多层 VLM 序列** 压缩为**固定长度特征**，是 ParaCAT 的必要前置，也可以被任何 Action Head 替换使用。

### 5.4 `VLAs/Sai0_1` — 端到端编排层

整个工程的"接线板"，对外暴露一套稳定 API。**所有服务和评估脚本都通过它间接调用 VLM / Action Head**，避免了直接 import 底层模块导致的耦合。

#### 4 个核心文件

| 文件 | 行数 | 职责 |
|---|---|---|
| `config.py` | 586 | `Sai0Config / VLMConfig / ActionHeadConfig / DataConfig / TrainingConfig`，支持 dict/json 互转 & 预设工厂方法 |
| `sai0_model.py` | 558 | `Sai0Model(nn.Module)`：懒加载 VLM + Action Head；三类 head（FM0/FM1/OFT）都有独立 `_create_*_head` 与 `_predict_*` 分支 |
| `data_utils.py` | 523 | 封装 `lerobot_dataset_loader`，提供 `create_sai0_dataloader` + `MinMaxNormalizer` + `sai0_collate_fn` + `quat2axisangle_torch` |
| `inference.py` | 527 | `Sai0Inference.from_checkpoint(...)`、`RealtimeInference`、`quick_inference`、`evaluate_checkpoint` |

#### Sai0Model 两种模式

```python
# ① 训练模式（使用预提取 VLM 特征，不需要 vlm_backbone）
loss = model.compute_loss(backbone_output, action_head_inputs)
# backbone_output.backbone_features = (B, S, D)   ← 来自 dataloader['vlm_hidden_states'][:, L, :, :]
# action_head_inputs.state = (B, 1, 64)
# action_head_inputs.action = (B, 16, 32)
# action_head_inputs.embodiment_id = (B,)

# ② 推理模式（实时提取 VLM 特征）
actions = model.predict(images=[img1, img2],
                       instruction="pick up the apple",
                       state=torch.tensor([...]))
# 内部根据 action_head.head_type 自动走 _predict_flow_matching 或 _predict_oft
```

#### 三种 Action Head 的推理路径差异

| head_type | 调用方式 | 关键差异 |
|---|---|---|
| `flow_matching_0/1` | `action_head.get_action(backbone_output, action_input)` | 需要 4 步去噪；state padding 到 max_state_dim |
| `oft_1_0` | `action_head(vlm_hidden_states=List, proprioception=state)` | 一次前向出结果；需要整个 List of hidden states；proprio 只取前 PROPRIO_DIM 维 |

这两个分支在 `sai0_model.py` 的 `_predict_flow_matching` / `_predict_oft` 里清晰分开。

### 5.5 `utils` — 数据与特征工具链

#### 5.5.1 `lerobot_dataset_loader.py`（~2013 行，最核心的工具文件）

职责：
- 读取 LeRobot 格式的 `parquet`（state/action/metadata）+ `mp4`（图像）+ `npy/npz`（VLM hidden states）
- 做 **action chunking**（窗口内拼接 N 步动作）
- 做 **欧拉角/四元数/轴角** 的互转（`euler_to_quat_numpy`、`quat_to_axisangle_numpy` 等）
- 提供 **共享内存缓存**（多 DDP 进程共享 VLM states）：`preload_vlm_cache_distributed` + `shared_memory`
- 提供 **视频 reader 缓存**（`max_cached_video_readers`）
- 支持 WebDataset 格式（见 `webdataset_utils.py`）

核心函数：
- `create_lerobot_dataloader(dataset_path, batch_size, num_workers, ...)` → `DataLoader`
- `preload_vlm_cache_distributed(...)` → 把 VLM states 一次性 mmap 到共享内存
- `collate_fn` → 把 dict 列表打成 batch，并自动 pad

批次字段见第 4.3 节。

#### 5.5.2 `extract_vlm_hidden_state/S0_1/`（离线特征提取）

```
S0_1/
├── qwen/
│   ├── qwen_extract_vlm_hidden_states.py          单 GPU
│   └── qwen_multi_dataset_multi_gpu_extraction.py 多数据集多 GPU 并行
├── eagle/eagle_extract_vlm_hidden_states.py
├── cosmos/cosmos_extract_vlm_hidden_states.py
└── scripts/
    ├── run_qwen_extraction.sh
    ├── run_eagle_extraction.sh
    ├── run_cosmos_extraction.sh
    └── run_qwen_multi_dataset_multi_gpu_save_per_episode.sh   ← 生产脚本
```

这些脚本**本质上是 `VLMs/S0_1/backbone/model_selector.py --mode hidden_state` 的包装**，但提供：
- 更好的多 GPU 分片（按 episode 划分给不同 rank）
- 更方便的目录结构（chunk-XXX.npz）
- 跨数据集批量提取

#### 5.5.3 其他 utils

| 文件 | 职责 |
|---|---|
| `discrete.py` | 动作离散化/反离散化（三分类 ↔ 连续 delta），ParaCAT 用 |
| `hand_processor.py` | 灵巧手/手指数据处理 |
| `euler_unwrap.py` | 跨帧欧拉角解缠绕，防止 ±π 跳变 |
| `polyfit_chunk.py` | 动作轨迹多项式拟合（训练前预处理，平滑动作）|
| `webdataset_utils.py` | WebDataset tar-stream 支持 |
| `convert_libero_plus_to_lerobot_v2.py` | LIBERO-Plus → LeRobot v2 数据集转换 |
| `migrate_vlm_npy_to_chunk_npz.py` | 把 per-frame `.npy` 迁移成 per-episode chunk `.npz` |
| `pt2safetensors/transfer.py` | `.pt` ↔ `.safetensors` 权重转换 |
| `download/libero_plus.py` | LIBERO-Plus 下载 |
| `dataset_format_transform/final_format_lerobot/` | 通用数据集 → LeRobot 最终格式 |

### 5.6 `deployment` — HTTP 推理服务

**四个独立服务**，针对不同场景：

#### 5.6.1 `Sai0_1_server/` — 主推理服务（生产级）

```
Sai0_1_server/
├── server.py              FastAPI 应用（1220 行，含 lifespan/限流/队列）
├── client.py              Python 客户端 SDK
├── client_test.py         命令行测试工具
├── example_client_test.py 4 种使用方式示例
├── config.yaml            服务器配置（VLM/head/ckpt/task_suites/preprocess）
├── auth.py                API Key 鉴权
├── queue_worker.py        GPU 推理队列（避免并发冲突）
├── dashboard.html         实时监控面板
├── IMAGE_FORMATS.md       图像格式说明
└── logs/                  server.log / metrics
```

**关键特性**：
- 支持 **多 task_suite**（一个服务托管多个 checkpoint，路由按 `task_suite` 字段）
- 支持 **base64** 和 **numpy JSON** 两种图像格式
- 图像预处理：`resize / flip_h / flip_v / rotate_180`
- state 预处理：按索引配置 `zero_to_minus_one / enable_normalization / min_val / max_val`
- 自动加载 `dataset_path/meta/stats.json` 做 action 反归一化
- 请求队列（`queue_worker.py`）+ 限流（`slowapi`）+ API Key（`auth.py`）
- 端点：`/predict` `/predict_batch` `/v1/act` `/health` `/info` `/debug` `/docs`

**架构：**

```
Client ─HTTP POST /v1/act─► FastAPI
                               ├─ Auth (Bearer API Key, auth.py)
                               ├─ Limiter (slowapi, rate_limit.v1_act)
                               ├─ Queue (queue_worker.InferenceQueue)
                               └─ Inference (VLAs.Sai0_1.RealtimeInference)
                                     ├─ preprocess_image(...)  (resize/flip/rotate)
                                     ├─ preprocess_state(...)  (zero→-1, min/max norm)
                                     ├─ Sai0Model.predict(...)
                                     └─ unnormalize_action(...)  (用 stats.json)
                               Response: {actions, timing, metadata}
```

#### 5.6.2 `tele/` — ParaCAT 遥操作服务

独立的 FastAPI，组合 `create_vlm_backbone + PonsAdapter + ParaCATActionHead`，不经过 Sai0_1 层。针对真机遥操作场景。

#### 5.6.3 `tele_parad/` — Pons + ParaCAT 遥操作（带误差分析）

与 `tele` 类似，但额外带累积误差可视化（`cumulative_error_plots*`），用于调优。

#### 5.6.4 `tele_parad_new/` — VLM + OFT 的兼容层服务

```
tele_parad_new/
├── server.py           新服务实现
├── open_loop_eval.py   开环评估
├── preprocessing/      额外预处理
└── config.yaml
```

**API 接口兼容旧的 `deploy_dataset197_jointangle.py`**，可以直接给旧客户端 `robot_control_no_hand_status_joint_roi_new_bgr2rgb.py` 使用。内部结构：`create_vlm_backbone + create_vlm2oft_pipeline`。

### 5.7 `eval` — 闭环评估

```
eval/
├── libero/              ← 旧版：按 head 划分的 LIBERO 评估
│   ├── Flow_Matching_0/ eval_benchmark.py / eval_trainingset.py / eval_eagle_action_head.py
│   ├── Flow_Matching_1/ eval_benchmark.py / eval_trainingset.py
│   ├── N1_5/            eval_n1_5.py  (直接用 GR00T N1.5 官方接口评估)
│   └── OFT1_0/          eval_not_sai0_pipeline_qwen.py
└── Sai0_1/              ← 新版：基于 Sai0_1 模块的统一评估
    ├── libero/
    │   ├── Flow_Matching_0/eval_Sai0_1.py
    │   ├── Flow_Matching_1/eval_Sai0_1.py
    │   ├── OFT1_0/eval_Sai0_1.py
    │   └── ParaCAT/eval_Sai0_1.py
    └── tele/ParaCAT/
```

**每个 `eval_Sai0_1.py` 的套路**：

1. 读取 checkpoint（`.pt`）+ 对应 `dataset_path/meta/stats.json`
2. 构造 `Sai0Config` / 实例化 `Sai0Inference`
3. 调用 LIBERO 仿真环境（`libero_spatial/object/goal/10/90/100`）
4. 对每个 `(task, rollout)` 循环：`obs → images → model.predict → env.step`
5. 记录 `success_rate / avg_steps` 到 JSON + 可选保存视频

#### LIBERO benchmark 配置

| benchmark | max_steps | 任务数 |
|---|---|---|
| `libero_spatial` | 220 | 10 |
| `libero_object` | 280 | 10 |
| `libero_goal` | 600 | 10 |
| `libero_10` | 1000 | 10（综合）|
| `libero_90` | 400 | 90 |
| `libero_100` | 600 | 100 |

### 5.8 `tools` — 辅助脚本

纯独立脚本，不参与主流程，但在数据调试和结果分析时经常用到：

命名规范：全部 `snake_case`，按功能分子目录，产物统一落到 `tools/outputs/<脚本名>/`。

| 类别 | 子目录 | 脚本 |
|---|---|---|
| **离散化基线分析** | `calc_discrete/` | `calc_discrete_ce_baseline.py` / `calc_discrete_error{,_batch}.py` / `calc_discrete_tracking{,_batch}.py` |
| **数据格式校验** | `check_data_format/` | `check_libero_hdf5_column_action_info.py` / `check_parquet_column_action{,_specified_delta}.py` / `check_npy_shape.py` / `check_safetensors.py` |
| **动作/状态可视化** | `plot_action/` | `plot_action_columns.py` / `plot_action_polyfit.py` / `plot_state_columns.py` / `plot_state_action_compare.py` / `plot_3d_trajectory.py` |
| **Checkpoint 评估** | `evaluate_checkpoint/` | `evaluate_oft_action_trajectory.py` / `plot_cumulative_error_per_dim.py` |
| **开环评估** | `openloop_eval/` | `openloop_eval.py` + `run.sh` |
| **OFT 端到端测试** | (根目录) | `test_oft_cumulative_error.py` |
| **模型结构 dump** | `model_structures/` | `model_structure_<hash8>_<date>.txt`（Eagle 权重结构快照） |
| **脚本产物** | `outputs/` | 各脚本默认输出落到 `outputs/<脚本名>/` |

---

## 6. 三大数据/特征流

### 6.1 离线 VLM 特征预提取流

```
 LeRobot 数据集
  └── data/chunk-XXX/*.parquet
  └── videos/chunk-XXX/*.mp4
  └── meta/tasks.jsonl
        │
        ▼
utils/extract_vlm_hidden_state/S0_1/<vlm>/<vlm>_extract_vlm_hidden_states.py
        │  内部调用 VLMs.S0_1.backbone.create_vlm_backbone
        │  逐帧：get_images_from_video(episode, frame, chunk, image_keys, flip)
        │         backbone.get_hidden_states(images, instruction)
        │         stacked = output.to_stacked_tensor()
        │
        ▼
  LeRobot 数据集 (同目录下)
  └── vlm_hidden_states/
       ├── hidden_state_000000.npy        (per-frame: (num_layers, seq_len, hidden_dim))
       ├── hidden_state_000001.npy
       └── ...
       或
       ├── chunk-000.npz                  (per-episode: key=episode_XXXXXX,
       └── ...                              value=(num_frames, num_layers, S, D))
```

训练时 `lerobot_dataset_loader` 按 `vlm_hidden_state_index` 查表从中读取。

### 6.2 训练特征流

```
LeRobot 数据集   ─┐
 预提取的 VLM     │
 hidden states   │
                  ▼
  lerobot_dataset_loader.create_lerobot_dataloader
                  │     (可选) preload_vlm_cache_distributed 到 /dev/shm
                  │     (可选) webdataset tar-stream
                  ▼
 batch = {
   vlm_hidden_states: (B, L, S, D),
   observation_state: (B, state_dim),
   actions:           (B, N_chunks, action_dim),
   task_description:  List[str],
   ...
 }
                  │
                  ▼
 Action_Heads/<Head>/train_multigpu.py
  ├─ build collate：把 dict batch → (backbone_output, action_head_inputs)
  ├─ ddp wrap action_head；optimizer = AdamW(lr=1e-4, betas=(0.95, 0.999))
  ├─ scheduler：warmup (5%) + cosine 到 max_steps
  ├─ amp: float16 或 bfloat16
  ├─ loss = action_head(backbone_output, action_head_inputs)["loss"]
  ├─ W&B 日志（wandb_project="gr00t_flowmatching_training" 等）
  └─ 保存 epoch_X/action_head.pt + config.json + best/
```

### 6.3 部署推理流（见 3.2 节）

---

## 7. 训练全流程串联

### 7.1 以 Flow_Matching_1 + Qwen3-VL-2B + LIBERO 为例

```bash
# ────────── STAGE 0：LeRobot 格式化数据集 ──────────
# 假设你已经有了 LIBERO 原始 hdf5，先转为 LeRobot 格式：
python utils/convert_libero_plus_to_lerobot_v2.py \
  --input_dir  /path/to/libero_spatial_raw \
  --output_dir /path/to/libero_lerobot_spatial

# ────────── STAGE 1：离线提取 VLM hidden states ──────────
bash utils/extract_vlm_hidden_state/S0_1/scripts/run_qwen_multi_dataset_multi_gpu_save_per_episode.sh
#  → 在数据集同目录下生成 vlm_hidden_states/chunk-XXX.npz

# ────────── STAGE 2：训练 Action Head ──────────
# （可选）提取 GR00T-N1.5 预训练权重（仅 Flow_Matching_0 需要）
python Action_Heads/Flow_Matching_0/extract_pretrained_weights.py \
  --pretrained_dir ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/<hash> \
  --output Action_Heads/Flow_Matching_0/pretrained_action_head.pt

# DDP 训练
cd Action_Heads/Flow_Matching_1
torchrun --nproc_per_node=8 train_multigpu.py \
  --data_path /path/to/libero_lerobot_spatial \
  --batch_size 32 --steps 20000 --lr 1e-4 \
  --action_backbone_dim 1536 --vlm_output_dim 2048 \
  --out_dir ./experiments/libero_spatial/checkpoints

# 也可以用 Action_Heads/batch_train.sh 串行跑多个实验
bash Action_Heads/batch_train.sh -r 10 \
  Action_Heads/OFT1_0/scripts/train/qwen/train_qwen.sh \
  Action_Heads/OFT1_0/scripts/train/eagle/train_eagle.sh

# ────────── STAGE 3：评估 ──────────
bash eval/Sai0_1/libero/Flow_Matching_1/scripts/run_eval_Sai0_1.sh
# 内部会：
#   python eval_Sai0_1.py --checkpoint_path .../step_20000/action_head.pt ...

# ────────── STAGE 4：部署 ──────────
# 改 deployment/Sai0_1_server/config.yaml 的 action_head_ckpt + vlm_type
./start_server.sh                                   # 启 FastAPI 服务
python deployment/Sai0_1_server/example_client_test.py  # 测试
```

### 7.2 训练脚本的共同骨架

所有 `Action_Heads/*/train_multigpu.py` 都遵循同一模板：

```python
1. 解析 argparse → TrainingConfig
2. init_process_group + find_free_port + 设置 device / local_rank
3. 创建 action_head = <Head>(get_<head>_config(...))；加载 pretrained_weights
4. DistributedDataParallel 包裹
5. 创建 dataloader：
     create_lerobot_dataloader(data_path, batch_size, DistributedSampler, ...)
6. 可选：preload_vlm_cache_distributed → /dev/shm 共享内存
7. Optimizer: AdamW(lr, betas=(0.95, 0.999), weight_decay=1e-5)
   Scheduler: LambdaLR(warmup 5%) → CosineAnnealingLR(max_steps)
8. AMP: GradScaler + autocast(float16/bfloat16)
9. for epoch in epochs:
     for step, batch in dataloader:
         # 转成 BatchFeature
         backbone_output = BatchFeature({"backbone_features": vlm_layer_L, "backbone_attention_mask": ...})
         action_head_inputs = BatchFeature({"state": padded_state, "action": padded_action, "action_mask": ..., "embodiment_id": 31})
         with autocast():
             out = action_head(backbone_output, action_head_inputs)
             loss = out["loss"] / gradient_accumulation_steps
         scaler.scale(loss).backward()
         if step % acc == 0:
             scaler.step(optimizer); scaler.update(); scheduler.step()
         if save_every_steps and step % save_every_steps == 0 and rank==0:
             save(f"step_{step}/action_head.pt", config.json)
10. W&B log(loss, lr, grad_norm, batch_time) 每 wandb_log_freq 步
11. 清理共享内存 + destroy_process_group
```

训练超参（所有 head 默认值基本一致）：
- `batch_size=32, lr=1e-4, weight_decay=1e-5, steps=20000, warmup_ratio=0.05`
- `AdamW betas=(0.95, 0.999)`（来自 GR00T N1.5 原始配方）
- `use_amp=True, amp_dtype=float16`
- `gradient_accumulation_steps=1` 默认，显存不够时调大
- `embodiment_id=31` 默认

---

## 8. 推理与部署全流程串联

见 3.2 节的数据流图。关键代码入口：

| 文件 | 职责 |
|---|---|
| `start_server.sh` | 一键启动 Sai0_1_server（硬编码默认值） |
| `deployment/Sai0_1_server/server.py` | FastAPI 应用主体 |
| `deployment/Sai0_1_server/config.yaml` | **部署期唯一配置来源** |
| `deployment/Sai0_1_server/auth.py` | `SAI0_API_KEYS` 环境变量读取 + Bearer 校验 |
| `deployment/Sai0_1_server/queue_worker.py` | `InferenceQueue`：asyncio 排队 + 超时 |
| `VLAs/Sai0_1/inference.py:RealtimeInference` | 对 `Sai0Model.predict()` 的进一步包装（加载 stats.json、管归一化） |

### 客户端示例（无鉴权本地测试）

```python
from deployment.Sai0_1_server.client import Sai0Client
import numpy as np
from PIL import Image

client = Sai0Client("http://localhost:5000")
result = client.predict(
    images=[Image.open("agentview.jpg"), Image.open("wrist.jpg")],
    state=np.zeros(16, dtype=np.float32),
    prompt="pick up the apple",
    use_numpy_format=True,
)
actions = np.array(result["actions"])  # (16, 7) 典型
```

### 多 task_suite 路由

```yaml
pipeline:
  task_suites:
    libero_spatial:
      action_head_ckpt: .../spatial.../action_head.pt
      dataset_path: .../spatial
    libero_object:
      action_head_ckpt: .../object.../action_head.pt
      dataset_path: .../object
    libero_goal: null        # 显式禁用
```

请求时带 `?task_suite=libero_spatial` → 路由到对应 `inference_engines[task_suite]`。

---

## 9. 评估全流程串联

### 9.1 LIBERO 闭环（`eval/Sai0_1/libero/<Head>/eval_Sai0_1.py`）

```python
# 伪代码骨架
cfg = Sai0Config(
    vlm = VLMConfig(model_type=..., model_path=..., layers=...),
    action_head = ActionHeadConfig(head_type=..., pretrained_weights=ckpt_path, ...)
)
inference = Sai0Inference.from_checkpoint(ckpt_path, ...)

benchmark = get_libero_benchmark("libero_spatial")
for task in benchmark.get_task_list():
    env = make_env(task)
    for rollout in range(num_rollouts):
        obs = env.reset()
        done = False
        while not done:
            images = [obs["agentview"], obs["robot0_eye_in_hand"]]
            state = extract_state(obs)   # 通常 9 维：gripper×2, xyz, quat
            actions = inference.predict(images, task.language_instruction, state)
            # actions 形状 (16, action_dim)，按 action chunking 执行前 K 步
            for a in actions[:execute_steps]:
                obs, reward, done, info = env.step(a)
                if done: break
```

### 9.2 离线开环（`*_trainingset.py` / `test_oft_cumulative_error.py`）

用训练集或测试集的 `(images, state)` 作为输入，跟 GT action 比较：
- L1 / L2 误差
- 累积误差（`tools/evaluate_checkpoint/plot_cumulative_error_per_dim.py`）
- 轨迹可视化（`tools/plot_action/plot_3d_trajectory.py`）

---

## 10. 关键配置点与默认值

### 10.1 VLM 默认层号

| 模型 | 默认 `layers` | `hidden_dim` |
|---|---|---|
| Qwen3-VL-2B-Instruct | `[14]` | 1536（VLM 语言侧）/ 2048（视觉 token 侧）|
| Qwen3-VL-4B-Instruct | `[16, 17, 18]` 典型 | 2560 |
| Qwen3-VL-7B-Instruct | — | 3584 |
| Eagle-2.5-VL (GR00T-N1.5-3B) | `[-1]` | 2048 |
| Cosmos-Reason-2B (Eagle-Block2A-2B-v2) | `[-1]` | 2048 |

### 10.2 Action Head padding 维度

**一旦训练，就不能再改**（权重 shape 固化）：
- `max_state_dim = 64`
- `max_action_dim = 32`
- `num_action_chunks = 16`
- `embodiment_id = 31`
- OFT 的 `PROPRIO_DIM = 8`, `ACTION_DIM = 7`, `NUM_ACTIONS_CHUNK = 16`

### 10.3 图像配置

- 默认两个视角：`["agentview", "wrist"]`（LIBERO）/ `["top", "left_wrist"]`（真机）
- `flip_images=True` 通常是为了修正仿真环境上下翻转

### 10.4 训练/部署环境

- Conda 环境：`qwen_eagle_hwl`
- Python 3.10 / PyTorch 2.8.0+cu128 / CUDA 12.8
- 必须安装：`flash_attn` / `timm` / `ninja` / `decord` / `pynvml` / `transformers==5.0.0rc1` / `lmdb`
- `/dev/shm` 至少 2T（训练时 VLM cache 共享内存）

---

## 11. 模块依赖矩阵

`A → B` 表示 A 直接 import / 调用 B。

| 上游 \ 下游 | VLMs/S0_1 | Action_Heads | Adapter/Pons | utils | VLAs/Sai0_1 |
|---|:-:|:-:|:-:|:-:|:-:|
| `VLMs/S0_1` | — |  |  | ✓ (少量) |  |
| `Action_Heads/Flow_Matching_*` |  | — |  | ✓ (lerobot_loader) |  |
| `Action_Heads/OFT1_0` |  | — |  | ✓ |  |
| `Action_Heads/ParaCAT` |  | — | ✓ (必须 Pons) | ✓ (discrete) |  |
| `Adapter/Pons` |  |  | — | ✓ |  |
| `utils/extract_vlm_hidden_state` | ✓ |  |  | — |  |
| `utils/lerobot_dataset_loader` |  |  |  | — |  |
| `VLAs/Sai0_1` | ✓ | ✓ (全部 head) |  | ✓ | — |
| `deployment/Sai0_1_server` | (via VLAs) | (via VLAs) |  | ✓ | ✓ |
| `deployment/tele*` | ✓ (直连) | ✓ (直连) | ✓ | ✓ |  |
| `eval/libero/*` | ✓ | ✓ |  | ✓ | — |
| `eval/Sai0_1/*` |  |  |  | ✓ | ✓ |

**关键洞察**：
- 🟢 `VLMs/S0_1` 是叶子模块（不依赖任何其他）
- 🟢 `Adapter/Pons` 也是叶子模块
- 🟡 `Action_Heads` 除了 ParaCAT 外都是叶子（只依赖 utils）
- 🔴 `VLAs/Sai0_1` 是**中心整合点**（新代码请走这里）
- 🔴 `deployment/tele*` 绕过了 `Sai0_1`，**属于历史遗留，谨慎修改**

---

## 12. 常见运维/接手注意事项

### 12.1 🔴 必须知道的"坑"

1. **`_1` vs 无后缀版本**：代码里实际生效的永远是 `_1` 结尾的版本（`VLAs/Sai0_1`、`VLMs/S0_1`）。`README.md` 里提到的 `VLAs/Sai0`、`VLMs/S0` **已不存在**。

2. **VLM 层号 off-by-one**：见 5.1 节。用户写的 `--layers 14` 对应真正的 transformer layer 13。所有 extract 脚本都统一按 `hidden_states[layer_idx]` 直接索引。

3. **`max_action_dim=32` / `max_state_dim=64` 不可改**：这两个值固化在预训练权重里。如需改，必须从头训。

4. **`/dev/shm` 大小**：训练时 VLM states 会被 mmap 到共享内存，默认 64GB 不够。参考 README 顶部设成 2TB。

5. **Flow_Matching_0 的预训练权重依赖**：必须先跑 `extract_pretrained_weights.py` 从 GR00T-N1.5-3B 提取。Flow_Matching_1 / OFT 不需要。

6. **OFT 的 `constants.py`**：是模块级全局变量，运行时会被 `Sai0Model._create_oft_head` 动态覆盖。**不要在多个线程 / 多个 OFT 实例并发使用不同配置**，否则互相冲突。

7. **训练的 batch embodiment_id**：所有 collate_fn 都硬编码 `embodiment_id=31`，如果接入新机体要同步修改。

8. **Eagle 默认路径写死**：`/home/sythoid_01/.cache/huggingface/...`，迁移机器时要改 `VLMs/S0_1/backbone/model_selector.py` 第 606 行。

9. **deployment 有 4 个独立服务，不要混淆**：生产用 `Sai0_1_server`，其他三个是历史遗留/特殊场景。

10. **共享内存清理**：训练异常退出后记得 `ls /dev/shm | grep vlm_cache` 检查是否有残留。`cleanup_shared_memory_cache()` 在 `train_multigpu.py` 里用 `atexit` 注册，但 SIGKILL 可能不触发。

### 12.2 建议的代码阅读顺序

新接手者按这个顺序看最顺：

1. 本文档（整体认识） ✅
2. `VLAs/Sai0_1/config.py` — 所有配置类
3. `VLAs/Sai0_1/sai0_model.py` — 主模型类
4. `VLMs/S0_1/backbone/model_selector.py` — VLM 工厂
5. `utils/lerobot_dataset_loader.py` — 数据格式定义
6. `Action_Heads/Flow_Matching_1/train_multigpu.py` — 训练流程模板
7. `deployment/Sai0_1_server/server.py` — 部署入口
8. `eval/Sai0_1/libero/OFT1_0/eval_Sai0_1.py` — 闭环评估模板

### 12.3 调试检查清单

当推理结果异常时：

- [ ] VLM 层号对不对？（`--layers` 的值和 `hidden_states` 下标的语义）
- [ ] `flip_images` 设了吗？（LIBERO 仿真/真实相机可能颠倒）
- [ ] Image resize 尺寸和训练时一致吗？（`config.yaml:preprocess.image.resize`）
- [ ] state 维度和 proprio_dim 匹配吗？（OFT 只取前 `PROPRIO_DIM=8` 维）
- [ ] `stats.json` 存在吗？（action 反归一化依赖它，不存在会用原始 raw action）
- [ ] Action Head 的 `head_type` 和 ckpt 真的匹配吗？
- [ ] `embodiment_id` 是否对应？（默认 31）
- [ ] VLM dtype 和训练时一致？（bfloat16 vs float16）

---

## 13. 文件级速查表

### 🔑 绝对不能漏看的 10 个文件

| # | 文件 | 说明 |
|---|---|---|
| 1 | `VLAs/Sai0_1/config.py` | 所有配置类定义 |
| 2 | `VLAs/Sai0_1/sai0_model.py` | 统一模型类 |
| 3 | `VLAs/Sai0_1/inference.py` | 推理接口 |
| 4 | `VLMs/S0_1/backbone/model_selector.py` | VLM 工厂 + 离线提取 CLI |
| 5 | `VLMs/S0_1/backbone/qwen3_vl/backbone.py` | Qwen backbone 实现 |
| 6 | `utils/lerobot_dataset_loader.py` | 数据层（2013 行） |
| 7 | `Action_Heads/Flow_Matching_1/train_multigpu.py` | 训练流程模板（2344 行） |
| 8 | `Action_Heads/OFT1_0/vlm2oft_pipeline.py` | OFT 完整架构 |
| 9 | `deployment/Sai0_1_server/server.py` | 主服务器 |
| 10 | `deployment/Sai0_1_server/config.yaml` | 部署唯一配置 |

### 各 Action Head 的关键文件对应表

| Head | 模型文件 | 训练脚本 | 配置 | Shell 启动 |
|---|---|---|---|---|
| FM0 | `Action_Heads/Flow_Matching_0/models/action_head/flow_matching_action_head.py` | `train_with_pretrained_action_head_weight_multigpu.py` | `config.py` | `scripts/train/{qwen,eagle,cosmos}/*.sh` |
| FM1 | `Action_Heads/Flow_Matching_1/models/action_head/flow_matching_action_head.py` | `train_multigpu.py` | `config.py` | `scripts/train/{qwen,eagle,cosmos}/*.sh` |
| OFT | `Action_Heads/OFT1_0/vlm2oft_pipeline.py` + `models/action_head/{action_heads_oft_orig,projectors_oft_orig}.py` | `train_multigpu.py` | `constants.py` | `scripts/train/{qwen,eagle,cosmos}/*.sh` |
| ParaCAT | `Action_Heads/ParaCAT/model/action_head/paracat_action_head.py` | `train_multigpu_{only,pons}_paracat.py` | 内嵌 | `scripts/*.sh` |

### 各 VLM 的关键文件

| VLM | backbone | config | prompt YAML |
|---|---|---|---|
| Qwen3-VL | `VLMs/S0_1/backbone/qwen3_vl/backbone.py` | `config.py` | `prompt_config.yaml` |
| Eagle 2.5 | `VLMs/S0_1/backbone/eagle2_5_vl/backbone.py` | `config.py` | `prompt_config.yaml` |
| Cosmos Reason | `VLMs/S0_1/backbone/cosmos_reason_2b_vl/backbone.py` | `config.py` | `prompt_config.yaml` |

### 各部署服务的对应场景

| 服务 | 场景 | 入口 | 依赖的 Head |
|---|---|---|---|
| `Sai0_1_server` | 生产通用 VLA 推理 | `server.py` + `config.yaml` | FM0/FM1/OFT |
| `tele` | 真机 ParaCAT 遥操作 | `server.py` + `config.yaml` | ParaCAT + Pons |
| `tele_parad` | ParaCAT 带误差分析 | `server.py` | ParaCAT + Pons |
| `tele_parad_new` | 兼容旧客户端 API 的 VLM+OFT | `server.py` | OFT |

---

## 附录 A：版本迁移备注（`Sai0` → `Sai0_1`）

本仓库历史上曾有 `VLAs/Sai0` 和 `VLMs/S0`（README 仍提到），但当前代码已全部迁移到 `_1` 后缀版本：

- `VLAs/Sai0_1`：统一支持 FM0 / FM1 / OFT 三种 head（旧版只支持 FM1）
- `VLMs/S0_1`：新增 Eagle 2.5 和 Cosmos Reason 后端（旧版只有 Qwen）
- 新增 `deployment/Sai0_1_server`（旧 `deployment/sai0_server` 已删除）
- 新增 `eval/Sai0_1/`（统一评估入口）

**新代码全部走 `_1` 版本**。旧版本的 `eval/libero/Flow_Matching_0|1|N1_5|OFT1_0` 仍可用，但**不推荐扩展**，请用 `eval/Sai0_1/` 下的版本。

---

## 附录 B：一份典型的完整训练→部署 checklist

```bash
# ① 环境
conda activate qwen_eagle_hwl
sudo mount -o remount,size=2T /dev/shm
df -h /dev/shm

# ② 数据集
ls /path/to/dataset/{data,meta,videos}
#   data/chunk-000/episode_000000.parquet
#   meta/{info.json, tasks.jsonl, stats.json}
#   videos/chunk-000/observation.images.{agentview,wrist}/episode_000000.mp4

# ③ 预提取 VLM features
bash utils/extract_vlm_hidden_state/S0_1/scripts/run_qwen_multi_dataset_multi_gpu_save_per_episode.sh
ls /path/to/dataset/vlm_hidden_states/   # chunk-XXX.npz

# ④ 训练
torchrun --nproc_per_node=8 Action_Heads/OFT1_0/train_multigpu.py \
  --data_path /path/to/dataset --batch_size 80 --steps 20000 \
  --out_dir Action_Heads/OFT1_0/experiments/run_YYYYMMDD/checkpoints

# ⑤ 离线评估（开环 L1 误差）
python tools/evaluate_checkpoint/evaluate_oft_action_trajectory.py \
  --checkpoint Action_Heads/OFT1_0/experiments/run_YYYYMMDD/checkpoints/step_20000/action_head.pt \
  --dataset /path/to/dataset

# ⑥ LIBERO 闭环评估
bash eval/Sai0_1/libero/OFT1_0/scripts/run_eval_Sai0_1.sh

# ⑦ 部署
vim deployment/Sai0_1_server/config.yaml   # 改 action_head_ckpt / dataset_path
./start_server.sh
curl http://localhost:5000/health
python deployment/Sai0_1_server/example_client_test.py

# ⑧ 真实客户端接入
# 参考 deployment/Sai0_1_server/client.py 的 Sai0Client 类
```

---

> **文档版本：v1.0（2026-04）**
> 如果在阅读或使用本文档过程中发现**新增的目录、重命名、API 变更**，请同步更新本文档对应章节，并在顶部 "文档更新日期" 字段留下日期，保证交接文档长期有效。

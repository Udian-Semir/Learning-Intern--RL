# ============================================================================
# VLM 配置 (VLM Configuration)
# ============================================================================
# 注意：保留此 constants 文件是为了维持与 OpenVLA 原生结构的兼容性。
# 虽然命名为常量，但由于进行了常量设置，这种做法并不完全合理。
# 然而，为了既保留 OpenVLA 原生结构又符合我们的代码需求，这些值可以被视为变量。
#
# NOTE: This constants file is preserved to maintain compatibility with the 
# original OpenVLA structure. While named as constants, this approach is not 
# entirely reasonable due to constant configuration requirements. However, to 
# preserve OpenVLA's native structure while meeting our codebase needs, these 
# values can be treated as variables.
#
# 重要注意事项 (IMPORTANT CONSIDERATIONS):
# 1. 【可动态修改】LLM_OUTPUT_DIM_MLP_INPUT_DIM 和 NUM_VLM_HIDDEN_LAYERS 这两个值
#    可以在 train.py 或其他模块调用时被覆盖/修改
#    [Dynamically modifiable] LLM_OUTPUT_DIM_MLP_INPUT_DIM and NUM_VLM_HIDDEN_LAYERS 
#    can be overridden/modified in train.py or when called from other modules
#
# 2. 【需手动修改】除上述两个值外，其余所有配置项只能在此 constants.py 文件中手动修改才能生效
#    虽然有点麻烦，但这是 OpenVLA 原生结构的限制
#    [Manual modification required] All other configuration items can only take effect 
#    by manually modifying this constants.py file. While somewhat inconvenient, this 
#    is a limitation of OpenVLA's native structure
#
# 3. 如果遇到写入失败，请检查文件权限问题
#    If encountering write failures, check file permissions
# ============================================================================
import os as _os


def _env_int(key: str, default: int) -> int:
    """从环境变量取 int，缺省/解析失败时回退 default（保持向后兼容）。"""
    raw = _os.environ.get(key, None)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(
            f"[constants.py] WARNING: 环境变量 {key}={raw!r} 不是整数，"
            f"回退 default={default}"
        )
        return default


LLM_OUTPUT_DIM_MLP_INPUT_DIM = 2048  # VLM 输出维度默认值 (可在 train.py 中覆盖)
NUM_VLM_HIDDEN_LAYERS = 1  # VLM 隐藏层数量默认值 (可在 train.py 中覆盖)

# Action-related
# 可在训练脚本里 `export ACTION_DIM=7` 覆盖，不必再改源码。
# - libero / mydataset256 (8-dim, 含 quat 第 7 位): 8
# - libero_plus           (7-dim, XYZ+RPY+gripper): 7
ACTION_DIM = _env_int("ACTION_DIM", 8)
NUM_ACTIONS_CHUNK = _env_int("NUM_ACTIONS_CHUNK", 1000)  # 30 -> 1000 20260402

# Proprio
# 注意: 此值需要与预处理后的 state 维度一致
# LIBERO 数据集: 8 维 (四元数转轴角后)
# mydataset256: 14 维 (hand_binary 预处理后)
# libero_plus:  8 维 (eef_pos_xyz + eef_euler_rpy + gripper_qpos_left + gripper_qpos_right)
PROPRIO_DIM = _env_int("PROPRIO_DIM", 8)

# ----------------------------------------------------------------------------
# ⚠️ 以下三个 USE_* 开关在当前 OFT1_0 代码里 **完全没被读取**（dead config）。
#    grep 'if USE_PROPRIO_PROJECTOR'  → No matches
#    grep 'if USE_NOISY_ACTION_PROJECTOR' → No matches
#    grep 'if USE_DIFFUSION'         → No matches
#
#    实际行为是写死的：
#      - vlm2oft_pipeline.VLM2OFTPipeline 在 __init__ 里 **无条件**创建
#        ProprioProjector_Changed (line 343)，forward 里 **无条件** 调用 (line 447)。
#      - L1RegressionActionHead 永远启用，DiffusionActionHead 没人实例化。
#
#    保留它们仅为向后兼容（README.md 和 deployment/.../config.yaml 还引用了名字）。
#    要真正做"开关"功能，应改为 CLI 参数（例如 --zero_proprio）而不是改这里。
# ----------------------------------------------------------------------------
USE_PROPRIO_PROJECTOR = False        # [DEPRECATED] 永远启用 proprio_projector
USE_NOISY_ACTION_PROJECTOR = False   # [DEPRECATED] 没有代码引用
USE_DIFFUSION = False                # [DEPRECATED] 训练写死走 L1Regression 路径
NUM_DIFFUSION_STEPS = 50             # [DEPRECATED] 仅 DiffusionActionHead 内部默认值

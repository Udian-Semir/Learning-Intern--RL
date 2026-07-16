#!/bin/bash
# ============================================================================
# Sai0-VLA 环境安装脚本
# ============================================================================
#
# 功能: 在 conda qwen 环境中安装项目所需的所有依赖
#
# 使用方法:
#   1. 确保已安装 conda
#   2. 运行: bash install_env.sh
#   3. 可选参数:
#      --env-name <name>    : 指定 conda 环境名称 (默认: qwen_eagle_hwl)
#      --cuda <version>     : 指定 CUDA 版本 (默认: 12.8, 可选: 11.8, 12.1, 12.4, 12.8)
#      --skip-libero        : 跳过 LIBERO 仿真环境安装
#      --skip-vlm           : 跳过 VLM 相关依赖安装
#      --skip-server        : 跳过部署服务器依赖安装
#      --full               : 完整安装 (包括所有可选依赖)
#      --dry-run            : 仅显示将要执行的命令，不实际执行
#      --help               : 显示帮助信息
#
# 示例:
#   bash install_env.sh                        # 使用默认配置安装
#   bash install_env.sh --env-name myenv       # 指定环境名称
#   bash install_env.sh --cuda 11.8            # 指定 CUDA 版本
#   bash install_env.sh --skip-libero          # 跳过 LIBERO 安装
#   bash install_env.sh --full                 # 完整安装
#
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 默认配置
# ============================================================================
ENV_NAME="qwen_eagle_hwl"
CUDA_VERSION="12.8"
INSTALL_LIBERO=true
INSTALL_VLM=true
INSTALL_SERVER=true
INSTALL_FULL=false
DRY_RUN=false
PYTHON_VERSION="3.10.0"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# 辅助函数
# ============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║           🚀 Sai0-VLA 环境安装脚本 v1.0                           ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN] $1${NC}"
    else
        echo -e "${CYAN}$ $1${NC}"
        eval "$1"
    fi
}

show_help() {
    echo "Sai0-VLA 环境安装脚本"
    echo ""
    echo "用法: bash install_env.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --env-name <name>    指定 conda 环境名称 (默认: qwen_eagle_hwl)"
    echo "  --cuda <version>     指定 CUDA 版本 (默认: 12.8)"
    echo "                       支持: 11.8, 12.1, 12.4, 12.8"
    echo "  --python <version>   指定 Python 版本 (默认: 3.10.0)"
    echo "  --skip-libero        跳过 LIBERO 仿真环境安装"
    echo "  --skip-vlm           跳过 VLM 相关依赖安装"
    echo "  --skip-server        跳过部署服务器依赖安装"
    echo "  --full               完整安装 (包括所有可选依赖)"
    echo "  --dry-run            仅显示将要执行的命令，不实际执行"
    echo "  --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  bash install_env.sh                        # 使用默认配置安装"
    echo "  bash install_env.sh --env-name myenv       # 指定环境名称"
    echo "  bash install_env.sh --cuda 11.8            # 指定 CUDA 版本"
    echo "  bash install_env.sh --skip-libero          # 跳过 LIBERO 安装"
    echo "  bash install_env.sh --full                 # 完整安装"
    exit 0
}

# ============================================================================
# 解析命令行参数
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --skip-libero)
            INSTALL_LIBERO=false
            shift
            ;;
        --skip-vlm)
            INSTALL_VLM=false
            shift
            ;;
        --skip-server)
            INSTALL_SERVER=false
            shift
            ;;
        --full)
            INSTALL_FULL=true
            INSTALL_LIBERO=true
            INSTALL_VLM=true
            INSTALL_SERVER=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 确定 PyTorch CUDA 版本
# ============================================================================

get_pytorch_cuda_version() {
    case $CUDA_VERSION in
        11.8)
            echo "cu118"
            ;;
        12.1)
            echo "cu121"
            ;;
        12.4)
            echo "cu124"
            ;;
        12.8)
            echo "cu128"
            ;;
        *)
            print_warning "未知 CUDA 版本 $CUDA_VERSION，使用默认 cu128"
            echo "cu128"
            ;;
    esac
}

PYTORCH_CUDA=$(get_pytorch_cuda_version)

# ============================================================================
# 开始安装
# ============================================================================

print_banner

echo -e "${CYAN}安装配置:${NC}"
echo "  • 环境名称: $ENV_NAME"
echo "  • Python 版本: $PYTHON_VERSION"
echo "  • CUDA 版本: $CUDA_VERSION"
echo "  • PyTorch CUDA: $PYTORCH_CUDA"
echo "  • 安装 LIBERO: $INSTALL_LIBERO"
echo "  • 安装 VLM 依赖: $INSTALL_VLM"
echo "  • 安装服务器依赖: $INSTALL_SERVER"
echo "  • 完整安装: $INSTALL_FULL"
echo "  • Dry Run: $DRY_RUN"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY-RUN 模式: 仅显示命令，不实际执行"
fi

# ============================================================================
# 步骤 1: 检查 conda
# ============================================================================

print_section "步骤 1: 检查 Conda 环境"

if ! command -v conda &> /dev/null; then
    print_error "未找到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

print_success "Conda 已安装: $(conda --version)"

# 检查环境是否存在
if conda env list | grep -q "^$ENV_NAME "; then
    print_success "环境 '$ENV_NAME' 已存在"
    print_step "激活现有环境..."
    run_cmd "source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME"
else
    print_step "创建新环境 '$ENV_NAME' (Python $PYTHON_VERSION)..."
    run_cmd "conda create -n $ENV_NAME python=$PYTHON_VERSION -y"
    run_cmd "source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME"
fi

# ============================================================================
# 步骤 2: 安装 PyTorch
# ============================================================================

print_section "步骤 2: 安装 PyTorch (CUDA $CUDA_VERSION)"

print_step "安装 PyTorch, TorchVision, TorchAudio..."

# 根据 CUDA 版本选择安装命令
# PyTorch 2.8.0 + TorchVision 0.23.0
case $CUDA_VERSION in
    11.8)
        run_cmd "pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu118"
        ;;
    12.1)
        run_cmd "pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu121"
        ;;
    12.4)
        run_cmd "pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu124"
        ;;
    12.8)
        run_cmd "pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128"
        ;;
    *)
        run_cmd "pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128"
        ;;
esac

# ============================================================================
# 步骤 3: 安装核心依赖
# ============================================================================

print_section "步骤 3: 安装核心依赖"

print_step "安装数据处理库..."
run_cmd "pip install numpy==2.2.6 pandas==2.3.3 pyarrow==22.0.0 scipy==1.15.3"

print_step "安装图像处理库..."
run_cmd "pip install pillow==11.3.0 opencv-python==4.12.0.88 opencv-python-headless==4.12.0.88"

print_step "安装视频处理库..."
run_cmd "pip install imageio==2.37.2 imageio-ffmpeg==0.6.0 av==16.0.1"

print_step "安装工具库..."
run_cmd "pip install tqdm==4.67.1 pyyaml==6.0.3 einops==0.8.1 rich==14.2.0"

# ============================================================================
# 步骤 4: 安装 VLM 依赖 (Transformers + Qwen)
# ============================================================================

if [ "$INSTALL_VLM" = true ]; then
    print_section "步骤 4: 安装 VLM 依赖 (Transformers + Qwen)"
    
    print_step "安装 Transformers 和 Accelerate..."
    run_cmd "pip install transformers>=4.43.0 accelerate==1.11.0 safetensors==0.6.2 tokenizers==0.22.1"
    
    print_step "安装 Qwen-VL 相关依赖..."
    run_cmd "pip install qwen-vl-utils tiktoken==0.12.0"
    
    print_step "安装 Flash Attention (可选，提升性能)..."
    # Flash Attention 安装可能失败，所以用 || true
    run_cmd "pip install flash-attn==2.8.3 --no-build-isolation || echo '⚠ Flash Attention 安装失败，将使用标准 attention'"
    
    print_step "安装 BitsAndBytes (量化支持)..."
    run_cmd "pip install bitsandbytes"
    
    print_step "安装 Diffusers 和 PEFT..."
    run_cmd "pip install diffusers==0.35.2 peft==0.18.0"
    
    print_success "VLM 依赖安装完成"
else
    print_section "步骤 4: 跳过 VLM 依赖安装"
fi

# ============================================================================
# 步骤 5: 安装服务器依赖 (FastAPI)
# ============================================================================

if [ "$INSTALL_SERVER" = true ]; then
    print_section "步骤 5: 安装服务器依赖 (FastAPI)"
    
    print_step "安装 FastAPI 和 Uvicorn..."
    run_cmd "pip install fastapi==0.121.2 uvicorn==0.38.0 starlette==0.49.3"
    
    print_step "安装 Pydantic 和 Requests..."
    run_cmd "pip install pydantic==2.12.4 requests==2.32.5 httpx==0.28.1"
    
    print_success "服务器依赖安装完成"
else
    print_section "步骤 5: 跳过服务器依赖安装"
fi

# ============================================================================
# 步骤 6: 安装 LIBERO 仿真环境 (可选)
# ============================================================================

if [ "$INSTALL_LIBERO" = true ]; then
    print_section "步骤 6: 安装 LIBERO 仿真环境"
    
    print_warning "LIBERO 安装需要 MuJoCo，请确保系统已配置好 MuJoCo"
    
    print_step "安装 MuJoCo Python 绑定..."
    run_cmd "pip install mujoco==3.3.7"
    
    print_step "安装 Robosuite..."
    run_cmd "pip install robosuite==1.4.0"
    
    print_step "安装 Gymnasium..."
    run_cmd "pip install gymnasium==1.2.2"
    
    print_step "安装 LIBERO 和 Robomimic..."
    # LIBERO 可能需要从源码安装
    run_cmd "pip install libero==0.1.1 robomimic==0.2.0 || echo '⚠ LIBERO pip 安装失败，可能需要从源码安装'"
    
    print_success "LIBERO 相关依赖安装完成"
else
    print_section "步骤 6: 跳过 LIBERO 安装"
fi

# ============================================================================
# 步骤 7: 安装额外依赖 (完整安装)
# ============================================================================

if [ "$INSTALL_FULL" = true ]; then
    print_section "步骤 7: 安装额外依赖 (完整安装)"
    
    print_step "安装 Wandb (实验追踪)..."
    run_cmd "pip install wandb==0.23.0"
    
    print_step "安装 TensorBoard..."
    run_cmd "pip install tensorboard==2.20.0 tensorboardX==2.6.4"
    
    print_step "安装 Matplotlib (可视化)..."
    run_cmd "pip install matplotlib==3.10.7"
    
    print_step "安装配置管理工具..."
    run_cmd "pip install hydra-core==1.3.2 omegaconf==2.3.0"
    
    print_step "安装 HDF5 支持..."
    run_cmd "pip install h5py==3.15.1"
    
    print_step "安装 Jupyter..."
    run_cmd "pip install jupyter jupyterlab ipywidgets"
    
    print_step "安装代码质量工具..."
    run_cmd "pip install black isort flake8"
    
    print_success "额外依赖安装完成"
else
    print_section "步骤 7: 跳过额外依赖安装"
fi

# ============================================================================
# 步骤 8: 验证安装
# ============================================================================

print_section "步骤 8: 验证安装"

if [ "$DRY_RUN" = false ]; then
    print_step "验证 PyTorch CUDA..."
    python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 设备: {torch.cuda.get_device_name(0)}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
" || print_error "PyTorch 验证失败"
    
    if [ "$INSTALL_VLM" = true ]; then
        print_step "验证 Transformers..."
        python -c "
import transformers
print(f'Transformers 版本: {transformers.__version__}')
" || print_error "Transformers 验证失败"
    fi
    
    print_step "验证核心库..."
    python -c "
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tqdm
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'Pillow: {Image.__version__}')
print('核心库验证成功!')
" || print_error "核心库验证失败"
fi

# ============================================================================
# 完成
# ============================================================================

print_section "安装完成!"

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║           ✅ Sai0-VLA 环境安装完成!                               ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${CYAN}后续步骤:${NC}"
echo ""
echo "  1. 激活环境:"
echo -e "     ${YELLOW}conda activate $ENV_NAME${NC}"
echo ""
echo "  2. 验证安装:"
echo -e "     ${YELLOW}python -c \"import torch; print(torch.cuda.is_available())\"${NC}"
echo ""
echo "  3. 提取 VLM Hidden States (示例):"
echo -e "     ${YELLOW}python -m VLMs.S0_1.backbone.model_selector --list_models${NC}"
echo ""
echo "  4. 训练 Action Head (示例):"
echo -e "     ${YELLOW}cd Action_Heads/Flow_Matching_0 && bash scripts/train/eagle/train_eagle.sh${NC}"
echo ""
echo "  5. 启动推理服务器:"
echo -e "     ${YELLOW}./start_server.sh${NC}"
echo ""
echo -e "${CYAN}更多信息请参阅 README.md${NC}"
echo ""


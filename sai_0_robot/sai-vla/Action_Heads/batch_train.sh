#!/bin/bash
# =============================================================================
# Batch Training Script (Enhanced Version)
# 批量训练脚本 - 按顺序执行多个训练脚本，每个脚本完成后清理资源
# =============================================================================
# Usage / 用法:
#   bash batch_train.sh [OPTIONS] script1.sh script2.sh script3.sh ...
#
# Options / 选项:
#   -r, --rest-time MINUTES   设置休息时间（分钟），默认30分钟
#   -n, --no-rest             跳过休息时间
#   -l, --log-dir DIR         日志输出目录，默认当前目录
#   -d, --dry-run             预览模式，不实际执行
#   -c, --continue            从上次失败位置继续执行
#   -h, --help                显示帮助信息
#
# Examples / 示例:
#   bash batch_train.sh script1.sh script2.sh                    # 默认配置
#   bash batch_train.sh -r 10 script1.sh script2.sh              # 休息10分钟
#   bash batch_train.sh -n script1.sh script2.sh                 # 不休息
#   bash batch_train.sh -d script1.sh script2.sh                 # 预览模式
#   bash batch_train.sh -c script1.sh script2.sh                 # 断点续训
#   bash batch_train.sh -l /data/logs -r 5 script1.sh            # 日志+休息5分钟
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 默认配置
REST_TIME=1800          # 休息时间（秒）- 默认30分钟
LOG_DIR="."             # 日志目录
DRY_RUN=false           # 预览模式
CONTINUE_MODE=false     # 断点续训模式
NO_REST=false           # 跳过休息

# 状态文件（用于断点续训）
STATE_FILE=".batch_train_state"

# 日志文件
LOG_FILE=""

# =============================================================================
# 日志函数
# =============================================================================
log_info() {
    local msg="${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}[INFO]${NC} $1"
    echo -e "$msg"
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "$LOG_FILE"
}

log_warn() {
    local msg="${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${YELLOW}[WARN]${NC} $1"
    echo -e "$msg"
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> "$LOG_FILE"
}

log_error() {
    local msg="${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${RED}[ERROR]${NC} $1"
    echo -e "$msg"
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "$LOG_FILE"
}

log_debug() {
    local msg="${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${CYAN}[DEBUG]${NC} $1"
    echo -e "$msg"
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEBUG] $1" >> "$LOG_FILE"
}

# =============================================================================
# 显示帮助信息
# =============================================================================
show_help() {
    cat << EOF
批量训练脚本 (Enhanced Version)

用法: bash batch_train.sh [OPTIONS] script1.sh script2.sh ...

选项:
  -r, --rest-time MINUTES   设置休息时间（分钟），默认30分钟
  -n, --no-rest             跳过休息时间（脚本间不休息）
  -l, --log-dir DIR         日志输出目录，默认当前目录
  -d, --dry-run             预览模式，显示执行计划但不实际执行
  -c, --continue            从上次失败位置继续执行
  -h, --help                显示此帮助信息

示例:
  bash batch_train.sh script1.sh script2.sh              # 默认配置
  bash batch_train.sh -r 10 script1.sh script2.sh        # 休息10分钟
  bash batch_train.sh -n script1.sh script2.sh           # 不休息
  bash batch_train.sh -d script1.sh script2.sh           # 预览模式
  bash batch_train.sh -c script1.sh script2.sh           # 断点续训
  bash batch_train.sh -l /var/log -r 5 script1.sh        # 指定日志目录，休息5分钟

EOF
    exit 0
}

# =============================================================================
# 信号处理 - 优雅退出
# =============================================================================
cleanup_on_exit() {
    local exit_code=$?
    echo ""
    log_warn "收到中断信号，正在清理..."
    
    # 清理资源
    cleanup_resources
    
    # 保存当前状态（用于断点续训）
    if [ -n "$CURRENT_SCRIPT_PATH" ]; then
        save_state "$CURRENT_SCRIPT_PATH" "interrupted"
        log_info "状态已保存，可使用 -c 选项继续执行"
    fi
    
    log_info "清理完成，退出"
    exit $exit_code
}

# 捕获信号
trap cleanup_on_exit SIGINT SIGTERM SIGHUP

# =============================================================================
# 状态管理函数（断点续训）
# =============================================================================
save_state() {
    local script_path="$1"
    local status="$2"
    echo "LAST_SCRIPT=$script_path" > "$STATE_FILE"
    echo "STATUS=$status" >> "$STATE_FILE"
    echo "TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')" >> "$STATE_FILE"
}

load_state() {
    if [ -f "$STATE_FILE" ]; then
        source "$STATE_FILE"
        echo "$LAST_SCRIPT"
    fi
}

clear_state() {
    rm -f "$STATE_FILE" 2>/dev/null
}

# =============================================================================
# 清理函数 - 清理GPU内存、共享内存和僵尸进程
# =============================================================================
cleanup_resources() {
    log_info "开始清理系统资源..."
    
    # 1. 清理可能残留的Python/训练进程
    log_info "检查并清理残留的训练进程..."
    pkill -9 -f "torchrun" 2>/dev/null || true
    pkill -9 -f "torch.distributed" 2>/dev/null || true
    pkill -9 -f "train_multigpu.py" 2>/dev/null || true
    sleep 3
    
    # 2. 清理GPU内存
    log_info "清理GPU内存..."
    if command -v nvidia-smi &> /dev/null; then
        # 显示当前GPU使用情况
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
        
        # 获取所有GPU进程并kill
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
        if [ -n "$gpu_pids" ]; then
            log_warn "发现残留GPU进程，正在清理..."
            for pid in $gpu_pids; do
                if [ -n "$pid" ] && [ "$pid" != "pid" ]; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done
        fi
        
        sleep 3
        
        # 显示清理后的GPU状态
        log_info "GPU清理后状态:"
        nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv
    else
        log_warn "nvidia-smi 不可用，跳过GPU清理"
    fi
    
    # 3. 清理共享内存 /dev/shm
    log_info "清理共享内存 (/dev/shm)..."
    df -h /dev/shm 2>/dev/null || true
    
    # 清理PyTorch相关的共享内存文件
    rm -rf /dev/shm/torch_* 2>/dev/null || true
    rm -rf /dev/shm/*_torch_* 2>/dev/null || true
    
    # 清理NCCL相关的共享内存（增强版）
    rm -rf /dev/shm/nccl* 2>/dev/null || true
    rm -rf /dev/shm/nccl-* 2>/dev/null || true
    rm -rf /dev/shm/*nccl* 2>/dev/null || true
    
    # 清理 c10d 相关的共享内存
    rm -rf /dev/shm/c10d_* 2>/dev/null || true
    
    # 清理 Python multiprocessing 共享内存（解决 leaked shared_memory 问题）
    # 这些文件通常以 psm_ 或随机字符串命名
    rm -rf /dev/shm/psm_* 2>/dev/null || true
    rm -rf /dev/shm/shm_* 2>/dev/null || true
    # 清理当前用户创建的共享内存文件（谨慎模式）
    find /dev/shm -maxdepth 1 -user "$(whoami)" -type f -mmin +5 -delete 2>/dev/null || true
    
    # 清理用户共享内存段
    if command -v ipcs &> /dev/null; then
        user_shm=$(ipcs -m | grep "$(whoami)" | awk '{print $2}')
        if [ -n "$user_shm" ]; then
            log_info "清理用户共享内存段..."
            for shmid in $user_shm; do
                ipcrm -m "$shmid" 2>/dev/null || true
            done
        fi
    fi
    
    log_info "共享内存清理后状态:"
    df -h /dev/shm 2>/dev/null || true
    
    # 4. 清理Python缓存和multiprocessing共享内存
    log_info "清理Python缓存和共享内存..."
    python3 -c "
import gc
gc.collect()

# 清理 multiprocessing 共享内存（解决 leaked shared_memory 警告）
try:
    from multiprocessing import resource_tracker
    # 尝试清理资源追踪器中的残留资源
except:
    pass

# 尝试关闭所有残留的共享内存对象
try:
    from multiprocessing.shared_memory import SharedMemory
    import os
    for f in os.listdir('/dev/shm'):
        if f.startswith('psm_') or f.startswith('shm_'):
            try:
                shm = SharedMemory(name=f, create=False)
                shm.close()
                shm.unlink()
            except:
                pass
except:
    pass
" 2>/dev/null || true
    
    # 5. 清理 /tmp 下的训练相关临时文件
    log_info "清理临时文件..."
    rm -rf /tmp/torch_* 2>/dev/null || true
    rm -rf /tmp/nccl_* 2>/dev/null || true
    
    # 6. 同步文件系统缓存
    log_info "同步文件系统..."
    sync
    
    log_info "系统资源清理完成！"
}

# =============================================================================
# 解析命令行参数
# =============================================================================
SCRIPTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rest-time)
            REST_TIME=$(($2 * 60))  # 转换为秒
            shift 2
            ;;
        -n|--no-rest)
            NO_REST=true
            shift
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -c|--continue)
            CONTINUE_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        -*)
            log_error "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
        *)
            SCRIPTS+=("$1")
            shift
            ;;
    esac
done

# 检查是否提供了脚本
if [ ${#SCRIPTS[@]} -eq 0 ]; then
    log_error "请提供至少一个训练脚本路径"
    echo ""
    echo "用法: bash batch_train.sh [OPTIONS] script1.sh script2.sh ..."
    echo "使用 -h 或 --help 查看帮助"
    exit 1
fi

# 创建日志目录和日志文件
mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG_FILE="${LOG_DIR}/batch_train_$(date '+%Y%m%d_%H%M%S').log"
log_info "日志文件: $LOG_FILE"

# =============================================================================
# 断点续训处理
# =============================================================================
START_INDEX=0
if [ "$CONTINUE_MODE" = true ]; then
    LAST_SCRIPT=$(load_state)
    if [ -n "$LAST_SCRIPT" ]; then
        log_info "检测到上次中断的脚本: $LAST_SCRIPT"
        # 找到该脚本在列表中的位置
        for i in "${!SCRIPTS[@]}"; do
            if [ "${SCRIPTS[$i]}" = "$LAST_SCRIPT" ]; then
                START_INDEX=$i
                log_info "将从第 $((START_INDEX + 1)) 个脚本继续执行"
                break
            fi
        done
    else
        log_warn "未找到断点续训状态，将从头开始执行"
    fi
fi

# =============================================================================
# 预览模式
# =============================================================================
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}  预览模式 - 执行计划${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo ""
    echo -e "配置信息:"
    echo -e "  休息时间: $(( REST_TIME / 60 )) 分钟"
    echo -e "  跳过休息: $NO_REST"
    echo -e "  日志目录: $LOG_DIR"
    echo -e "  断点续训: $CONTINUE_MODE"
    echo -e "  起始位置: 第 $((START_INDEX + 1)) 个脚本"
    echo ""
    echo -e "待执行脚本 (共 ${#SCRIPTS[@]} 个):"
    for i in "${!SCRIPTS[@]}"; do
        script="${SCRIPTS[$i]}"
        if [ $i -lt $START_INDEX ]; then
            echo -e "  ${YELLOW}[跳过]${NC} $((i + 1)). $script"
        elif [ -f "$script" ]; then
            echo -e "  ${GREEN}[就绪]${NC} $((i + 1)). $script"
        else
            echo -e "  ${RED}[不存在]${NC} $((i + 1)). $script"
        fi
    done
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "预览完成，未实际执行任何操作"
    echo -e "移除 -d/--dry-run 选项以实际执行"
    echo -e "${MAGENTA}========================================${NC}"
    exit 0
fi

# =============================================================================
# 主执行逻辑
# =============================================================================

# =============================================================================
# 初始化：设置 NCCL 环境变量 & 清理残留资源（防止 NCCL 超时错误）
# =============================================================================
log_info "=============================================="
log_info "🔧 初始化分布式训练环境..."
log_info "=============================================="

# 设置 NCCL 环境变量（增加超时时间，防止集合操作超时）
export NCCL_TIMEOUT=7200                    # NCCL 超时时间 2 小时
export TORCH_DISTRIBUTED_TIMEOUT_SEC=7200   # PyTorch 分布式超时 2 小时
export NCCL_ASYNC_ERROR_HANDLING=1          # 启用异步错误处理
export NCCL_IB_DISABLE=0                    # 允许 InfiniBand（如果有）
export NCCL_P2P_DISABLE=0                   # 允许 P2P 通信

log_info "NCCL 环境变量已设置:"
log_info "  NCCL_TIMEOUT=$NCCL_TIMEOUT"
log_info "  TORCH_DISTRIBUTED_TIMEOUT_SEC=$TORCH_DISTRIBUTED_TIMEOUT_SEC"
log_info "  NCCL_ASYNC_ERROR_HANDLING=$NCCL_ASYNC_ERROR_HANDLING"

# 启动前清理残留资源（防止之前崩溃的训练留下的共享内存导致问题）
log_info "清理残留的共享内存和进程..."
cleanup_resources

log_info "✅ 分布式训练环境初始化完成"
log_info "=============================================="
echo ""

# 统计信息
TOTAL_SCRIPTS=${#SCRIPTS[@]}
CURRENT_SCRIPT=0
FAILED_SCRIPTS=()
SUCCESS_SCRIPTS=()
SKIPPED_SCRIPTS=()

echo ""
log_info "=============================================="
log_info "批量训练开始"
log_info "=============================================="
log_info "总共 ${TOTAL_SCRIPTS} 个训练脚本待执行"
if [ "$NO_REST" = true ]; then
    log_info "休息时间: 已禁用"
else
    log_info "休息时间: $((REST_TIME / 60)) 分钟"
fi
log_info "日志文件: $LOG_FILE"
log_info "=============================================="

# 列出所有待执行的脚本
echo ""
log_info "待执行脚本列表:"
for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    if [ $i -lt $START_INDEX ]; then
        echo -e "  ${YELLOW}[跳过]${NC} $((i + 1)). $script"
    else
        echo "  $((i + 1)). $script"
    fi
done
echo ""

# 开始执行
START_TIME=$(date +%s)

for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    CURRENT_SCRIPT=$((i + 1))
    CURRENT_SCRIPT_PATH="$script"
    
    # 跳过断点续训之前的脚本
    if [ $i -lt $START_INDEX ]; then
        log_info "跳过脚本 [$CURRENT_SCRIPT/$TOTAL_SCRIPTS]: $script (断点续训)"
        SKIPPED_SCRIPTS+=("$script")
        continue
    fi
    
    echo ""
    log_info "=============================================="
    log_info "执行脚本 [$CURRENT_SCRIPT/$TOTAL_SCRIPTS]: $script"
    log_info "=============================================="
    
    # 检查脚本是否存在
    if [ ! -f "$script" ]; then
        log_error "脚本不存在: $script"
        FAILED_SCRIPTS+=("$script (不存在)")
        save_state "$script" "not_found"
        continue
    fi
    
    # 检查脚本是否可执行
    if [ ! -x "$script" ]; then
        log_warn "脚本没有执行权限，尝试使用 bash 执行: $script"
    fi
    
    # ========== 运行前清理（确保每个脚本启动时环境干净）==========
    if [ $CURRENT_SCRIPT -gt 1 ]; then
        # 非第一个脚本，在运行前再次清理（初始化时已清理过第一个）
        log_info "运行前清理：确保 NCCL/GPU 环境干净..."
        cleanup_resources
        log_info "✅ 运行前清理完成"
    fi
    
    # 显示当前 GPU 内存状态（用于诊断 OOM 问题）
    if command -v nvidia-smi &> /dev/null; then
        log_info "当前 GPU 内存状态:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv
    fi
    
    # 显示系统内存状态
    log_info "当前系统内存状态:"
    free -h
    
    # 记录脚本开始时间
    SCRIPT_START=$(date +%s)
    
    # 保存当前状态
    save_state "$script" "running"
    
    # 执行脚本
    if bash "$script"; then
        SCRIPT_END=$(date +%s)
        SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))
        log_info "脚本执行成功: $script"
        log_info "执行耗时: $((SCRIPT_DURATION / 3600))小时 $((SCRIPT_DURATION % 3600 / 60))分钟 $((SCRIPT_DURATION % 60))秒"
        SUCCESS_SCRIPTS+=("$script")
        save_state "$script" "success"
    else
        log_error "脚本执行失败: $script"
        FAILED_SCRIPTS+=("$script (执行失败)")
        save_state "$script" "failed"
    fi
    
    # 如果不是最后一个脚本，执行清理和休息
    if [ $CURRENT_SCRIPT -lt $TOTAL_SCRIPTS ]; then
        # ========== 第一次清理：运行结束后立即清理 ==========
        echo ""
        log_info "=============================================="
        log_info "训练完成，执行资源清理（第1次：运行结束后）..."
        log_info "=============================================="
        cleanup_resources
        
        # 休息（如果没有禁用）
        if [ "$NO_REST" = false ] && [ $REST_TIME -gt 0 ]; then
            echo ""
            log_info "=============================================="
            log_info "休息 $((REST_TIME / 60)) 分钟后继续下一个脚本..."
            log_info "=============================================="
            
            # 获取下一个脚本名称
            NEXT_IDX=$((i + 1))
            if [ $NEXT_IDX -lt ${#SCRIPTS[@]} ]; then
                log_info "下一个脚本: ${SCRIPTS[$NEXT_IDX]}"
            fi
            
            # 显示倒计时（每5分钟显示一次）
            for ((j=REST_TIME; j>0; j-=300)); do
                remaining_min=$((j / 60))
                if [ $remaining_min -gt 0 ]; then
                    log_info "剩余休息时间: ${remaining_min} 分钟"
                fi
                sleep_time=$((j > 300 ? 300 : j))
                sleep $sleep_time
            done
            
            log_info "休息结束，即将执行第2次清理..."
        fi
    else
        # 最后一个脚本也执行清理
        echo ""
        log_info "=============================================="
        log_info "最后一个训练完成，执行最终资源清理..."
        log_info "=============================================="
        cleanup_resources
    fi
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

# 清除状态文件（成功完成）
clear_state

# 打印总结
echo ""
log_info "=============================================="
log_info "批量训练完成"
log_info "=============================================="
log_info "总耗时: $((TOTAL_DURATION / 3600))小时 $((TOTAL_DURATION % 3600 / 60))分钟 $((TOTAL_DURATION % 60))秒"
log_info "成功: ${#SUCCESS_SCRIPTS[@]}/$TOTAL_SCRIPTS"
log_info "失败: ${#FAILED_SCRIPTS[@]}/$TOTAL_SCRIPTS"
log_info "跳过: ${#SKIPPED_SCRIPTS[@]}/$TOTAL_SCRIPTS"
log_info "日志文件: $LOG_FILE"

if [ ${#SUCCESS_SCRIPTS[@]} -gt 0 ]; then
    echo ""
    log_info "成功的脚本:"
    for script in "${SUCCESS_SCRIPTS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $script"
    done
fi

if [ ${#SKIPPED_SCRIPTS[@]} -gt 0 ]; then
    echo ""
    log_info "跳过的脚本:"
    for script in "${SKIPPED_SCRIPTS[@]}"; do
        echo -e "  ${YELLOW}○${NC} $script"
    done
fi

if [ ${#FAILED_SCRIPTS[@]} -gt 0 ]; then
    echo ""
    log_error "失败的脚本:"
    for script in "${FAILED_SCRIPTS[@]}"; do
        echo -e "  ${RED}✗${NC} $script"
    done
    echo ""
    log_info "提示: 可使用 -c 选项从失败位置继续执行"
    exit 1
fi

echo ""
log_info "全部训练任务完成！"

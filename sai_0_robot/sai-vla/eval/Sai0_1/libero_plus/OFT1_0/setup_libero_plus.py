#!/usr/bin/env python3
"""
LIBERO-plus 环境一键配置助手 (qwen_eagle_hwl 兼容)
================================================

本脚本只做"无副作用"的本地准备工作，不会动 conda env 中已安装的 libero
（因此不会破坏原始 LIBERO 评测）。具体做的事:

1. 在 ``/data_disk1/hwl/LIBERO-plus/libero/`` 下创建空的 ``__init__.py``
   （LIBERO-plus 仓库本身缺这个文件，导致 Python import 时会回退到 conda
   env 中已安装的原版 LIBERO；touch 一下就能让 ``sys.path.insert`` 真正生效）

2. 在用户目录写一份独立的 ``~/.libero_plus_sai0/config.yaml``（路径可改）。
   这样 LIBERO-plus 用 ``LIBERO_CONFIG_PATH`` 环境变量指向它，不会污染
   原版 LIBERO 用的 ``~/.libero/config.yaml``

3. 把 ``/data_disk1/hwl/LIBERO-plus/libero/libero/assets/`` 下缺的资产
   软链到 qwen_eagle_hwl 中已经安装好的 LIBERO assets，避免重新下载
   （注意: ``new_objects`` 目录是 LIBERO-plus 独有的，仍需要手动从
    HuggingFace 下载，否则带 ``_add_`` 或 ``_level`` 的任务会报错）

4. （可选）尝试用 ``huggingface_hub`` 下载 LIBERO-plus 自有的 ``new_objects``
   等 LIBERO-plus 独有资产到 ``/data_disk1/hwl/LIBERO-plus/libero/libero/assets``

使用:
    # 在 qwen_eagle_hwl 环境下:
    conda activate qwen_eagle_hwl
    python -m eval.Sai0_1.libero_plus.OFT1_0.setup_libero_plus

    # 或带选项:
    python -m eval.Sai0_1.libero_plus.OFT1_0.setup_libero_plus \\
        --libero_plus_root /data_disk1/hwl/LIBERO-plus \\
        --conda_libero_root /home/dev/miniconda3/envs/qwen_eagle_hwl/lib/python3.10/site-packages/libero \\
        --config_dir ~/.libero_plus_sai0 \\
        --download_assets

环境变量约定（被 eval_libero_plus.py 复用）:
    LIBERO_PLUS_ROOT       : LIBERO-plus 仓库根 (默认 /data_disk1/hwl/LIBERO-plus)
    LIBERO_PLUS_CONFIG_DIR : 自定义 libero config 目录 (默认 ~/.libero_plus_sai0)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import yaml

DEFAULT_LIBERO_PLUS_ROOT = "/data_disk1/hwl/LIBERO-plus"
DEFAULT_CONDA_LIBERO_ROOT = (
    "/home/dev/miniconda3/envs/qwen_eagle_hwl/lib/python3.10/site-packages/libero"
)
DEFAULT_CONFIG_DIR = "~/.libero_plus_sai0"
DEFAULT_HF_REPO_ID = "Sylvest/LIBERO-plus"

LIBERO_PLUS_NEW_ASSETS = ("new_objects",)

# LIBERO-plus 自带的 assets.zip 经常被解压到一个深路径里 (HuggingFace tar 包
# 解压后保留了上传者的服务器目录结构)。我们扫描这些"已知深路径"，尝试把里面
# 的扩展资产 merge 回 <libero_plus_root>/libero/libero/assets/ 下。
LIBERO_PLUS_DEEP_ASSET_CANDIDATES: tuple = (
    # 用户机器上观察到的实际位置
    "libero/libero/inspire/hdd/project/embodied-multimodality/public/syfei"
    "/libero_new/release/dataset/LIBERO-plus-0/assets",
    # 其它常见解压形态
    "_hf_assets_cache/assets",
    "assets",
)

# 合并时按子目录处理；列出来主要为了优先级和日志可读性（其他子目录会自动扫描）。
LIBERO_PLUS_ASSET_SUBDIRS: tuple = (
    "scenes",
    "new_objects",
    "articulated_objects",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)


def _info(msg: str) -> None:
    print(f"[setup_libero_plus] {msg}")


def _warn(msg: str) -> None:
    print(f"[setup_libero_plus][WARN] {msg}")


def _err(msg: str) -> None:
    print(f"[setup_libero_plus][ERROR] {msg}", file=sys.stderr)


def ensure_libero_package_init(libero_plus_root: Path) -> Path:
    """让 ``<libero_plus_root>/libero/`` 成为合法的 Python 包。

    LIBERO-plus 仓库的 ``libero/`` 子目录默认没有 ``__init__.py``，导致
    ``sys.path.insert(0, libero_plus_root)`` 之后 ``import libero`` 仍然
    会找到 conda env 中已经安装的旧版 LIBERO。我们 touch 一下就够了。
    """
    pkg_dir = libero_plus_root / "libero"
    if not pkg_dir.is_dir():
        raise FileNotFoundError(f"找不到 LIBERO-plus 包目录: {pkg_dir}")

    init_file = pkg_dir / "__init__.py"
    if init_file.exists():
        _info(f"✓ {init_file} 已存在")
    else:
        try:
            init_file.touch()
            _info(f"✓ 已创建 {init_file} (空文件，仅用于让 Python 识别 libero 为包)")
        except PermissionError as e:
            _err(f"无法创建 {init_file}: {e}")
            _err(f"  请运行: touch {init_file}")
            raise
    return init_file


def _link_or_skip(target: Path, source: Path) -> bool:
    """如果 source 不存在则跳过；如果 target 已存在但内容不一致只警告不替换。"""
    if not source.exists():
        _warn(f"  - 源 {source} 不存在，跳过")
        return False

    if target.exists() or target.is_symlink():
        _info(f"  - 目标 {target.name} 已存在，跳过")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
    _info(f"  ✓ symlink: {target.name} -> {source}")
    return True


def link_assets_from_conda(
    libero_plus_root: Path,
    conda_libero_root: Path,
) -> None:
    """把 conda env 已经安装好的 LIBERO 资产软链到 LIBERO-plus 资产目录。

    LIBERO-plus 自有但 conda env 缺失的资产 (例如 ``new_objects``) 不在这里处理，
    需要单独下载。
    """
    src_assets = conda_libero_root / "libero" / "assets"
    dst_assets = libero_plus_root / "libero" / "libero" / "assets"

    if not src_assets.exists():
        _warn(f"  - conda env 中找不到 LIBERO assets: {src_assets}")
        _warn(f"    请确认 qwen_eagle_hwl 环境已经安装了原版 libero")
        return

    dst_assets.mkdir(parents=True, exist_ok=True)
    _info(f"准备从 {src_assets} 软链 assets 到 {dst_assets}")

    for item in sorted(src_assets.iterdir()):
        target = dst_assets / item.name
        _link_or_skip(target, item)

    # 提示 LIBERO-plus 独有的资产
    missing_lp_only: List[str] = []
    for sub in LIBERO_PLUS_NEW_ASSETS:
        if not (dst_assets / sub).exists():
            missing_lp_only.append(sub)
    if missing_lp_only:
        _warn(
            f"  ⚠️ 以下 LIBERO-plus 独有的资产仍然缺失: {missing_lp_only}\n"
            f"     带 '_add_X' 或 '_levelX' 的任务（共 ~290 个 / 约 3% 任务）\n"
            f"     将无法运行。建议从 HuggingFace 下载 LIBERO-plus assets:\n"
            f"     https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main\n"
            f"     解压到: {dst_assets}\n"
            f"     或使用 --download_assets 参数尝试自动下载（见下方 try_download_lp_assets）"
        )


def _find_deep_asset_dirs(libero_plus_root: Path) -> List[Path]:
    """在 LIBERO-plus 仓库内寻找 LIBERO-plus 自有资产真正解压到的位置。

    判定标准: 候选目录里同时存在 ``new_objects`` 子目录或大量 ``tabletop_table_*.xml``
    文件，说明它就是 ``assets.zip`` 解压后的根。
    """
    found: List[Path] = []
    for rel in LIBERO_PLUS_DEEP_ASSET_CANDIDATES:
        cand = libero_plus_root / rel
        if not cand.is_dir():
            continue
        has_new_objects = (cand / "new_objects").is_dir()
        scenes_dir = cand / "scenes"
        scenes_count = 0
        if scenes_dir.is_dir():
            scenes_count = sum(
                1 for p in scenes_dir.iterdir()
                if p.is_file() and p.name.startswith("tabletop_table_") and p.suffix == ".xml"
            )
        # 只要满足"new_objects 存在"或"tabletop_table_*.xml >= 30"任一条件就接受
        if has_new_objects or scenes_count >= 30:
            found.append(cand.resolve())
    # 去重保序
    seen = set()
    unique: List[Path] = []
    for p in found:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _materialize_symlinked_dir(target: Path) -> None:
    """如果 ``target`` 是一个软链 → 其它真目录, 把它替换成"包含原内容的真目录"。

    用于把上一步 link_assets_from_conda() 创建的 symlink 转为可写实目录，
    然后才好把 LIBERO-plus 自己的扩展资产合并进来（不污染 conda env 源目录）。
    """
    if not target.is_symlink():
        return
    src = target.resolve()
    if not src.is_dir():
        target.unlink()
        return
    target.unlink()
    target.mkdir(parents=True, exist_ok=False)
    # 同名子项再用 symlink 指回 conda env，避免一次性 cp 上 GB 数据
    for child in src.iterdir():
        link = target / child.name
        try:
            link.symlink_to(child)
        except FileExistsError:
            continue
    _info(f"  ✓ 物化软链目录 {target} (内部仍逐项 symlink 回 {src})")


def merge_libero_plus_assets(
    libero_plus_root: Path,
    conda_libero_root: Path,
) -> None:
    """把 LIBERO-plus 自带的 assets (深路径) 合并到 ``assets/`` 正确位置。

    步骤:
      1. 扫描候选深路径，自动定位 LIBERO-plus 的 assets 根
      2. 把 ``assets/<sub>`` 里的"指向 conda env"软链物化为实目录（仍逐项 symlink 回 conda env）
      3. 把深路径中 LIBERO-plus 独有的文件 / 子目录 symlink 进来 (不会覆盖已有项)
    """
    deep_dirs = _find_deep_asset_dirs(libero_plus_root)
    if not deep_dirs:
        _warn(
            "  - 没在 LIBERO-plus 仓库内找到 LIBERO-plus 自带 assets 的解压位置。"
            "请确认 assets.zip 是否解压成功；常见路径见 LIBERO_PLUS_DEEP_ASSET_CANDIDATES。"
        )
        return

    dst_assets = libero_plus_root / "libero" / "libero" / "assets"
    dst_assets.mkdir(parents=True, exist_ok=True)

    for deep in deep_dirs:
        _info(f"合并 LIBERO-plus 资产 {deep} -> {dst_assets}")

        merged_files = 0
        merged_dirs = 0

        # 先处理已知子目录（顺序决定日志输出，没列入的也会兜底处理）
        all_subs: List[str] = list(LIBERO_PLUS_ASSET_SUBDIRS)
        for extra in sorted(p.name for p in deep.iterdir() if p.is_dir()):
            if extra not in all_subs:
                all_subs.append(extra)

        for sub in all_subs:
            src_sub = deep / sub
            if not src_sub.exists():
                continue
            dst_sub = dst_assets / sub

            if not dst_sub.exists() and not dst_sub.is_symlink():
                # 完全没有 -> 整个子目录直接 symlink 过来
                dst_sub.parent.mkdir(parents=True, exist_ok=True)
                dst_sub.symlink_to(src_sub)
                count = sum(1 for _ in src_sub.rglob("*"))
                merged_dirs += 1
                _info(f"  ✓ symlink {sub}/ -> {src_sub} ({count} entries)")
                continue

            # 已有 -> 物化软链, 然后逐项合并
            _materialize_symlinked_dir(dst_sub)
            if not dst_sub.is_dir():
                _warn(f"  - {dst_sub} 不是目录, 跳过")
                continue

            local_added = 0
            for child in sorted(src_sub.iterdir()):
                target = dst_sub / child.name
                if target.exists() or target.is_symlink():
                    continue
                target.symlink_to(child)
                local_added += 1
            if local_added:
                merged_files += local_added
                _info(f"  ✓ {sub}/: 合并 {local_added} 个新条目")
            else:
                _info(f"  - {sub}/: 已是最新, 无新条目")

        # 顶层散文件 (例: assets/wall.xml, serving_region.xml)
        for child in sorted(deep.iterdir()):
            if child.is_dir():
                continue
            target = dst_assets / child.name
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(child)
            merged_files += 1
            _info(f"  ✓ 顶层文件: {child.name}")

        _info(f"  合并完成: 新增 {merged_dirs} 个子目录, {merged_files} 个文件 (相对 {deep.name})")

    # 完成后再校验: new_objects 是否到位、scenes 至少应该有 200+ 项
    final_scenes = dst_assets / "scenes"
    final_new = dst_assets / "new_objects"
    n_scenes = sum(1 for _ in final_scenes.iterdir()) if final_scenes.exists() else 0
    n_new = sum(1 for _ in final_new.iterdir()) if final_new.exists() else 0
    _info(f"最终校验: scenes/={n_scenes} 项, new_objects/={n_new} 项")
    if n_scenes < 100:
        _warn("  ⚠️ scenes 数量看起来仍偏少, 后续评估可能仍会缺资产")
    if n_new == 0:
        _warn("  ⚠️ new_objects 仍然为空, 带 _add_/_level 的任务会失败")


def try_download_lp_assets(
    libero_plus_root: Path,
    repo_id: str = DEFAULT_HF_REPO_ID,
) -> bool:
    """尝试用 huggingface_hub 下载 LIBERO-plus 独有资产。

    如果下载失败（鉴权、网络等问题）只发警告，不阻断脚本。
    """
    dst_assets = libero_plus_root / "libero" / "libero" / "assets"
    dst_assets.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download  # noqa: WPS433
    except ImportError:
        _warn("  - huggingface_hub 未安装，无法自动下载，请手动下载并解压")
        return False

    _info(f"尝试从 HuggingFace 下载 LIBERO-plus assets: {repo_id}")
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["assets/**", "assets.zip"],
            local_dir=str(dst_assets.parent.parent.parent / "_hf_assets_cache"),
            local_dir_use_symlinks=False,
        )
    except Exception as e:  # noqa: BLE001
        _warn(f"  - HuggingFace 下载失败: {e}")
        _warn(
            "    请手动:\n"
            "      1) 访问 https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main\n"
            "      2) 下载 assets.zip 并解压到 "
            f"{dst_assets}"
        )
        return False

    _info(f"✓ HuggingFace 下载完成，缓存路径: {local_dir}")
    _warn(
        "  ⚠️ HuggingFace 上的 assets 还需要手动确认是否解压到 "
        f"{dst_assets}（README 给的路径），按需自行调整"
    )
    return True


def write_libero_config(
    libero_plus_root: Path,
    config_dir: Path,
    extra_dataset_path: Optional[str] = None,
) -> Path:
    """生成独立的 LIBERO-plus config.yaml。"""
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / "config.yaml"
    benchmark_root = libero_plus_root / "libero" / "libero"
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": extra_dataset_path or str(benchmark_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with cfg_path.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    _info(f"✓ 写入 LIBERO-plus 配置: {cfg_path}")
    for k, v in config.items():
        _info(f"    {k}: {v}")
    return cfg_path


def export_env_hint(libero_plus_root: Path, config_dir: Path) -> None:
    _info("")
    _info("===== ⬇️  评估前请 export 以下环境变量 (eval 脚本内部也会自动 export) =====")
    _info(f"  export LIBERO_PLUS_ROOT={libero_plus_root}")
    _info(f"  export LIBERO_PLUS_CONFIG_DIR={config_dir}")
    _info(f"  export LIBERO_CONFIG_PATH={config_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--libero_plus_root", type=str, default=DEFAULT_LIBERO_PLUS_ROOT,
        help="LIBERO-plus 仓库根目录"
    )
    parser.add_argument(
        "--conda_libero_root", type=str, default=DEFAULT_CONDA_LIBERO_ROOT,
        help="conda env 中已安装的 libero 包根目录 (libero/__init__.py 所在目录的父级)"
    )
    parser.add_argument(
        "--config_dir", type=str, default=DEFAULT_CONFIG_DIR,
        help="LIBERO-plus 自定义 config.yaml 所在目录"
    )
    parser.add_argument(
        "--extra_dataset_path", type=str, default=None,
        help="config.yaml 里的 datasets 字段 (评估流程不会用到，可空)"
    )
    parser.add_argument(
        "--download_assets", action="store_true",
        help="尝试用 huggingface_hub 下载 LIBERO-plus 独有资产 (new_objects 等)"
    )
    parser.add_argument(
        "--merge_libero_plus_assets", action="store_true",
        help=(
            "把 LIBERO-plus 自带 assets 从深路径 (例: "
            "libero/libero/inspire/hdd/.../LIBERO-plus-0/assets) 合并到正确位置 "
            "libero/libero/assets/。如果 assets.zip 已经手动解压但放错地方就用这个修复。"
        )
    )
    parser.add_argument(
        "--no_link_assets", action="store_true",
        help="跳过从 conda env 软链 assets 这一步 (默认执行, 仅在你已经手动准备好资产时使用)"
    )
    parser.add_argument(
        "--hf_repo_id", type=str, default=DEFAULT_HF_REPO_ID,
        help="HuggingFace repo id (datasets/Sylvest/LIBERO-plus)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    libero_plus_root = Path(args.libero_plus_root).expanduser().resolve()
    conda_libero_root = Path(args.conda_libero_root).expanduser().resolve()
    config_dir = Path(args.config_dir).expanduser().resolve()

    _info(f"LIBERO-plus root  : {libero_plus_root}")
    _info(f"Conda libero root : {conda_libero_root}")
    _info(f"Config dir        : {config_dir}")

    if not libero_plus_root.exists():
        _err(f"LIBERO-plus 仓库不存在: {libero_plus_root}")
        return 1

    ensure_libero_package_init(libero_plus_root)
    if not args.no_link_assets:
        link_assets_from_conda(libero_plus_root, conda_libero_root)
    if args.merge_libero_plus_assets:
        merge_libero_plus_assets(libero_plus_root, conda_libero_root)
    if args.download_assets:
        try_download_lp_assets(libero_plus_root, repo_id=args.hf_repo_id)
    write_libero_config(libero_plus_root, config_dir, args.extra_dataset_path)

    export_env_hint(libero_plus_root, config_dir)
    _info("✓ Setup 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import os
import sys
import subprocess
import glob
import platform


def find_files(category, base_dir="./data/origin00/Amazon"):
    """查找所需的数据文件"""
    train_files = glob.glob(f"{base_dir}/train/{category}*11.csv")
    eval_files = glob.glob(f"{base_dir}/valid/{category}*11.csv")
    test_files = glob.glob(f"{base_dir}/test/{category}*11.csv")
    info_files = glob.glob(f"{base_dir}/info/{category}*.txt")

    return {
        "train_file": train_files[0] if train_files else None,
        "eval_file": eval_files[0] if eval_files else None,
        "test_file": test_files[0] if test_files else None,
        "info_file": info_files[0] if info_files else None,
    }


def run_sft_training(category, config=None):
    """运行 SFT 训练"""
    # 默认配置
    default_config = {
        "base_model": "./Qwen2.5-0.5B",
        "batch_size": 128,
        "micro_batch_size": 4,
        "output_dir": f"./output/Qwen2.5-0.5B/{category}",
        "wandb_project": "MiniOneRec",
        "wandb_run_name": f"{category}_sft",
        "category": category,
        "train_from_scratch": False,
        "seed": 42,
        "sid_index_path": "./data/Amazon/index/Industrial_and_Scientific.index.json",
        "item_meta_path": "./data/Amazon/index/Industrial_and_Scientific.item.json",
        "freeze_LLM": False,
    }

    # 更新配置
    if config:
        default_config.update(config)
    
    config = default_config

    # 查找数据文件
    files = find_files(category)
    if not files["train_file"] or not files["eval_file"] or not files["info_file"]:
        print(f"错误：无法找到 category {category} 的数据文件")
        print(f"查找结果: {files}")
        return False

    # 打印文件路径
    print("-" * 80)
    print(f"训练文件: {files['train_file']}")
    print(f"评估文件: {files['eval_file']}")
    print(f"Info 文件: {files['info_file']}")
    if files['test_file']:
        print(f"测试文件: {files['test_file']}")
    print("-" * 80)

    # 设置环境变量
    env = os.environ.copy()
    env["NCCL_IB_DISABLE"] = "1"  # 禁用 IB/RoCE

    # 构建命令
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", "1",
        "sft.py",
        "--base_model", config["base_model"],
        "--batch_size", str(config["batch_size"]),
        "--micro_batch_size", str(config["micro_batch_size"]),
        "--train_file", files["train_file"],
        "--eval_file", files["eval_file"],
        "--output_dir", config["output_dir"],
        "--wandb_project", config["wandb_project"],
        "--wandb_run_name", config["wandb_run_name"],
        "--category", config["category"],
        "--train_from_scratch", str(config["train_from_scratch"]),
        "--seed", str(config["seed"]),
        "--sid_index_path", config["sid_index_path"],
        "--item_meta_path", config["item_meta_path"],
        "--freeze_LLM", str(config["freeze_LLM"]),
    ]

    # 执行命令
    print(f"\n运行命令: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"训练失败，返回码: {e.returncode}")
        return False


def main():
    # 要处理的类别列表
    categories = ["Industrial_and_Scientific"]
    # 如果需要处理多个类别，可以改为:
    # categories = ["Industrial_and_Scientific", "Office_Products"]

    print("=" * 80)
    print("MiniOneRec SFT 训练脚本 (Python 版本)")
    print(f"平台: {platform.platform()}")
    print(f"Python: {sys.version}")
    print("=" * 80)

    for category in categories:
        print(f"\n正在处理类别: {category}")
        print("=" * 80)
        
        success = run_sft_training(category)
        
        if success:
            print(f"\n✓ 类别 {category} 训练成功")
        else:
            print(f"\n✗ 类别 {category} 训练失败")
            return 1

    print("\n" + "=" * 80)
    print("所有类别处理完成")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
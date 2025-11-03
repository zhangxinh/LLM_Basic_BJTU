# PowerShell 运行训练脚本（使用 GPU）
# 用法：在项目根目录的 PowerShell 中运行此脚本
# 如果使用虚拟环境，请先激活，例如: .\.venv\Scripts\Activate

# 选择 GPU（可选）
$env:CUDA_VISIBLE_DEVICES = "0"

# 可选：显示 CUDA / PyTorch 环境信息
python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda runtime version:', torch.version.cuda)
    try:
        print('device name:', torch.cuda.get_device_name(0))
    except Exception as e:
        print('get_device_name error:', e)
PY

# 运行训练（exact reproducible 命令，含随机种子）
python src/train.py --config src/config.yaml --data_dir data --seed 42

<h1 align="center">BJTU—LLM-ZH2EN</h1>




## 📖 总览

这是一个基于 **PyTorch** 的纯手写代码，包括模型的架构，数据的提前处理，一整个训练的流程。它提供了一个完整的、可复现的训练和评估流程，旨在作为 Transformer 模型在机器翻译任务中的最小可用示例。

## 🗂️ 目录结构

```text
LLM_Basic_BJTU/
|
+-- configs/
|   +-- config.yaml
|
+-- data/	#此处为我从hugging_face上取得数据集处理后的结果
|   +-- iwslt2017_test.jsonl
|   +-- iwslt2017_train.jsonl
|   +-- iwslt2017_validation.jsonl
|
+-- src/
|   +-- __init__.py
|   +-- __pycache__/
|   +-- ablation.py
|   +-- config.yaml
|   +-- evaluate.py
|   +-- model.py
|   +-- modules.py
|   +-- test.ipynb
|   +-- train.py
|   +-- utils.py
|   +-- visualization.py
|
+-- scripts/
|   +-- train.sh
|   +-- evaluate.sh
|
+-- results/	#我已经在此存了我8次消融实验的全部结果
|   +-- model.pt
|   +-- loss.png
|   +-- ....
|
+-- requirements.txt
+-- README.md
```

---

## 📦 环境与依赖

```bash
$ conda create -n LLM_basic python=3.10
$ conda activate LLM_basic
$ pip install -r requirements.txt
```


---

## 🚀 快速开始
### 1. 准备数据 (JSONL)

默认配置使用 `data/` 目录下的 `iwslt2017_train.jsonl` 和 `iwslt2017_validation.jsonl`。

文件格式为 **JSON Lines**（每行一个 JSON 对象），且必须包含 `chinese` 和 `english` 两个键：

```
print(len(zh_train), len(en_train))
231266 231266
print(len(zh_test), len(en_test))
8549 8549
```

### 2. 训练

使用 `src/train.py` 脚本启动训练。通过 `base.yaml` 参数可以方便地切换不同的实验配置。

```bash
# 启动训练
$ python train.py --config configs/base.yaml

# 或直接通过命令行参数指定
$ python train.py \
    --dataset iwslt2017 \
    --batch_size 32 \
    --epochs 20 \
    --lr 3e-4 \
    --num_layers 4 \
    --num_heads 4 \
    --hidden_dim 256 \
    --dropout 0.1 \
    --max_len 128 \
    --save_path ../results/model.pt
```

最后模型权重以及结果储存在results文件夹中

### 3.环境要求

**GPU**：`NVIDIA RTX 4090 (24GB)`
**CUDA**: `12.5`
**OS**: `Ubuntu 22.04 LTS`
**平均训练时长**: `3 小时 / 20 epoch (batch size=32)`

## 总结

本研究实现并分析了一个完整的Transformer 神经机器翻译模型，基于IWSLT2017 Zh->En 数据集进行了系统实验。从模型结构、训练配置到结果评估，复现了标准 Transformer 的主要模块（多头注意力、前馈层、残差连接、位置编码等），并通过可视化与消融实验验证了关键组件的重要性。

- 在小规模平行语料上，参数配置为hidden\_dim=256, num\_heads=4, num\_layers=4的模型能在训练稳定性与性能之间取得良好平衡；
- 正弦位置编码显著提升模型的顺序建模能力，其缺失会导致性能大幅下降；
-  合理的学习率调度与 warmup 策略能有效防止早期不稳定，促进快速收敛；
- item 模型输出的译文流畅且语义一致，说明实现的自注意力机制正确且有效。


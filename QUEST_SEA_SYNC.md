# QUEST + 温和 SEA：独立 decoder、公共配置复用的同步说明

## 需要同步的改动

以下路径相对于 `deimv2` 目录。原有 MoE、HSFPN、QUEST、SEA、LWGA 的实现和实验配置没有修改。

| 类型 | 路径 | 内容 |
| --- | --- | --- |
| 新增 | `engine/deim/deim_decoder_quest_sea.py` | 独立注意力、decoder layer、FDR decoder 和模型类 |
| 新增 | `configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_SEA_decoder.yml` | 与 QUEST 对照相同的简洁配置，复用 HSFPN_dw 公共设置 |
| 新增 | `outputs/run_deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_SEA_decoder.ps1` | Windows PowerShell 启动脚本 |
| 修改一行 | `engine/deim/__init__.py` | 添加下方注册导入，其他内容保持不变 |
| 附带说明 | `QUEST_SEA_SYNC.md` | 本文件 |

在目标机器 `engine/deim/__init__.py` 中添加以下一行，已有则不重复添加：

```python
from .deim_decoder_quest_sea import DEIMTransformer_QUEST_SEA
```

同步包有意不包含整个 `__init__.py`，避免覆盖目标机器已有的其他模块注册。
压缩包保留 `deimv2/...` 路径：解压到 `huya-deimv2` 根目录，或按上述相对路径复制文件。
它不包含数据、权重、训练日志或训练环境。

## 独立实现的范围

- 新文件中的四个类都只继承 PyTorch 的 `nn.Module`。
- 不导入、不继承、不委托调用旧 `DEIMTransformer`、QUEST 或 LWTSEA decoder。
- Q/K/V 投影、QUEST、SEA gate、query 初始化、DN、逐层解码、FDR 和输出组装均有独立实现。
- decoder 代码独立，但 YAML 与 QUEST 对照一样，通过 `__include__` 引用 `deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw.yml`。
- 新 YAML 只声明新实验输出目录、与 QUEST 一致的 MoE loss 开关、模型选择、decoder 参数及 `sea_gate_scale`；数据、MoE、HSFPN、优化器和训练日程复用公共配置。
- 仍复用本工程的通用算子，例如 `MSDeformableAttention`、`LQE`、`Integral`、`Gate`、`RMSNorm`、框变换和 DN 工具；不复制整套基础工程。

目标机器需要已经有同一版本的 DEIMv2 + HGNetv2_MoE_v2 + HybridEncoder_HSFPN 基础工程及其注册、依赖。
还需要已有 `configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw.yml` 及其引用的公共配置。同步包不覆盖这些原有配置；若缺失，需要一并同步基础工程的配置目录。
本同步包只添加第三个改进点，不会为原始上游 DEIMv2 自动安装前两个改进。
新 decoder 本身不需要旧的 `deim_decoder_quest.py` 或 `deim_decoder_lwtsea.py`。
若目标机器与本机的基础算子接口不同，需先同步基础工程版本。

## 新注意力及保持不变的部分

- 只对每个 head 的 K 做 L2 归一化，SDPA 使用 `scale=1.0`。
- 延续原 DN mask；门控只作用于当前 query，不混合不同 query。
- 门控为 `G = 1 + 0.1 * tanh(Linear(V))`，作用于 attention head 输出，再拼接并执行输出投影。
- 门控线性层零初始化，初始 `G=1`；0.1 是固定幅度，不是从零开始的可学习系数。
- 门控在 heads 间共享，每个 decoder layer 独立；128 通道、8 heads、3 层时增加 816 个参数。
- 不添加 LWGA、伪二维 query 网格或二维 memory 增强。
- 保持 cross-attention、Gateway、FFN、FDR、预测头、候选选择和 DN 主流程。

新 YAML 与 QUEST 对照复用同一份公共配置。目前有效设置包括：

- MoE stage 4、3 个专家、Top-2 和共享专家；`moe_loss_weight: 0.0`。
- HSFPN stride-16、DCT、残差和 `dw` 输出。
- decoder 128 通道、3 层、300 queries、SiLU、采样点 `[6, 6]`。
- PaQ 关闭；优化器、160 epoch、数据增强、EMA 和 matcher 日程不随新模块修改。
- 原先的 `flat_epoch: 7800`、`lr_gamma: 1.0` 原样保留，不在本次改动中调整学习率方案。

以后修改公共配置时，这两个实验都会继承公共设置的变化，不再保存独立的完整配置快照。

## 目标机器需要调整的路径

数据路径默认复用 `configs/dataset/uavdt_detection.yml`。在目标机器确认该公共文件的三个节点中的 `dataset.img_folder` 和 `dataset.ann_file` 正确：

- `train_dataloader`：相同的 UAVDT 图像目录及 `uavdt_realtrain.json`。
- `val_dataloader`：相同的 UAVDT 图像目录及 `uavdt_test.json`。
- `test_dataloader`：相同的 UAVDT 图像目录及 `uavdt_test.json`。

当前值是本机 Windows 的 E 盘路径。跨机器时只调整位置，不要无意更换数据划分。
如果其他实验需要保留原路径，也可以只在新 YAML 中增加对应的路径覆盖项，而不修改公共数据配置。
启动脚本按自身位置定位工程和输出目录，不需要修改盘符。

## Windows 启动

激活已有训练环境后，在 `huya-deimv2` 根目录执行：

```powershell
conda activate deimv2
powershell -ExecutionPolicy Bypass -File .\deimv2\outputs\run_deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_SEA_decoder.ps1
```

脚本只启动一次训练：新实验目录没有 `last.pth` 就从头训练，有则仅续训这个实验自己的检查点。
不会加载旧 QUEST/LWGA 目录的权重。不要把不同结构实验的 `last.pth` 复制进新目录。
可以通过脚本的 `-PythonBin` 参数指定目标环境中的 `python.exe`。

也可以在 `deimv2` 目录直接启动新实验（此命令不自动续训）：

```powershell
python train.py -c configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_SEA_decoder.yml --use-amp --seed=0
```

若需要禁用门控做 QUEST 对照，可把新配置的 `sea_gate_scale` 设为 0，但必须使用不同的输出目录；不要混用两个实验的日志或检查点。

## 本次验证范围

已进行 Python/YAML/PowerShell 语法与静态一致性检查，包括构造参数、DN mask、自注意力公式、原 decoder 前向方法和配置有效值对照。
未导入工程构造模型，未执行模型前向/反向、训练或冒烟测试；未产生训练检查点。

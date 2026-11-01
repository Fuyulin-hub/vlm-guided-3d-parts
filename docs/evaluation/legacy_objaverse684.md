# 论文三张表的评估脚本

> 历史文档：本页保留早期 684 清单适配记录，不代表当前服务器、论文或发布状态。现行说明见 [README.md](README.md)。原文中“独立人工参考”不能用于声称 D 臂的评分独立于其输入；D 使用同一参考作引导时，只衡量参考一致性。

这是本项目的评估实现，不是 SegViGen 官方评估代码。只读取已有标签图、相机和日志，不启动模型推理，也不需要 GPU。`config.example.json` 是通用模板；`config.objaverse684.json` 已录入用户提供的服务器目录与事件格式，但尚未连接服务器执行真实批量评价。

## 当前 Objaverse 目录适配

共同根目录：`/data2/codefile/fuyulin/experiments`。服务器上将脚本包解压到任意工作目录，从解压根目录运行：

```bash
python scripts/eval_table2.py --config docs/evaluation/config.objaverse684.json --out outputs/evaluation
python scripts/eval_table1.py --config docs/evaluation/config.objaverse684.json --out outputs/evaluation
python scripts/eval_table3.py --config docs/evaluation/config.objaverse684.json --out outputs/evaluation
```

目前表 2 的二维部分已配置；三维投影、外部方法、未提供路径的 v5 单因素消融、人工评分没有虚构目录，相关单元格为 `TBD`。表 1 仍输出所有已约定的方法行，而不是把 Ours 单独误报为完整对比。

- 清单固定为 `objaverse_684/manifest_frozen.csv`，检查 220 / 232 / 232 的分组数量及 UID 唯一性；不会改用 pilot、v5 运行副本或 D 工程全集。数量匹配不等于身份匹配：应另行核对 v5 运行副本与冻结清单的 UID 集合或 SHA-256，不能仅凭文件字节数。
- D 读取 `human_fixed_710/masks/C_qwen_sam2/{group}/{uid}/labels_uint16.png`，只查冻结清单中的对象；不把工程名中的 `C_qwen_sam2` 当作其标注来源证明。676/710、约 642/684 是用户提供的覆盖情况，尚未在本地重计。
- B 读取 `objaverse_684/guidance/B_sam2/{group}/{uid}.png`，使用 `colors_pure.json` 的数组索引。配置中的调色板绝对路径来自现有 v5 代码的默认参数，仍需核对实际运行是否改过该参数。
- B 的 `label_semantics=palette_index` 会进入逐样本 CSV 和报告 JSON。它表示匿名调色板分区，不保证还原真实独立部件。若生成 B 时循环复用颜色，多个部件已不可逆合并，不能把其 Part F1 解读为真实部件实例 F1；正式比较前需核查颜色是否一对一，或提供原始独立掩码/整数 ID。
- v4 基线选择 `C_v4_exhaustive_box_point`，配置使用其同名 annotations 中的 `parts[].label_id + color` 精确解码。提供的真实样例来自 `C_v4_exhaustive`，故仍需确认 box-point 目录也采用同构 JSON；若不同，不能用另一配置的标注冒充。
- v5 使用现有本地 v5 runner 的 `final_map.labels_path` 读取整数图，目录设为 `annotations/C_v5_full`。用户提供了全量运行臂名，尚未逐文件核验 full 批次是否采用该 schema；缺少记录不自动回退到 pilot 或彩色图。
- 不用 `*_rot90`。如果实际标签方向不同，先核对渲染/导出历史；不自动旋转寻找最高分。
- 缺 D 记为 `missing_reference`，仍保留在固定清单内，不当作方法失败、也不按空预测填零。当前均值仍要求固定组内样本齐全，因此缺 D 时主表为 `TBD`。若之后批准固定 D 有效子集，应单独冻结清单并让所有方法共用；不能按方法成功情况动态挑选。

### 真实事件日志如何筛选

事件日志可使用配置字段：

```json
{
  "path": "objaverse_684/logs/abc_events.jsonl",
  "phase": "segmentation",
  "filters": {"stage": "segvigen", "arm": "B_sam2"},
  "time_key": "seconds",
  "memory_key": "peak_allocated_bytes",
  "memory_unit": "bytes"
}
```

`filters` 对顶层字段或点分嵌套字段作精确匹配。D 可改为 `d_events.jsonl`、`arm=D_human_fixed`；D 不是自动方法对比行，此示例仅供审计。不把 `stage=trellis, arm=shared` 的 611 秒样例算作 B 方法开销。`waiting`、`cached` 以及跳过类记录均不提供本次实测成功/耗时/显存；D 的 `waiting + guidance missing` 即使携带大显存值也不计入。

同一 UID 筛选后仍有多个真正测量事件时会报错，要求选择唯一 run/config/日志，既不统计每次重试为一个新物体，也不任意选最快一次。可在 `filters` 中加 `config_hash` 约束；若同一配置仍重复，需明确选择一次运行记录。

仅有 SegViGen 阶段时设方法字段 `measurement_scope=segmentation_only`，组件测量保留在逐样本 `measured_component_seconds` 与 `measured_component_peak_gpu_memory_gb` 中，但不会写入端到端总耗时/全流程显存主列。B 在当前配置中只属于表 2；其事件适配已登记，若需要单独审计效率，可在复制的配置中将 B 的 `tables` 加上 3，不改变正式主表行设计。

v5 当前只连接二维 guidance 日志，没有捏造尚未提供的三维阶段日志，所以其端到端耗时、显存、失败率与塌缩率仍为 `TBD`。正式表 3 还需完整的各方法计时与独立塌缩判定。

## 1. 三个入口

在项目根目录运行，服务器上的 `python` 应指向实验环境：

```bash
python -m pip install -r scripts/requirements-evaluation.txt
python scripts/eval_table1.py --config docs/evaluation/config.example.json --root /your/experiment/root --out outputs/evaluation
python scripts/eval_table2.py --config docs/evaluation/config.example.json --root /your/experiment/root --out outputs/evaluation
python scripts/eval_table3.py --config docs/evaluation/config.example.json --root /your/experiment/root --out outputs/evaluation
```

请先复制配置再修改，不要把示例目录当成真实实验路径。`--root` 覆盖 JSON 的 `root`。其余数据路径相对该根目录，支持 `{group}`、`{uid}` 和 manifest 中的其他列；也支持绝对路径。

每个入口导出 `tableN.csv`、`tableN.json`、`tableN_per_sample.csv`。CSV 可直接用 Excel 打开，JSON 保留未四舍五入的逐组统计、覆盖率和输入错误。表 1 同时生成空白人工评分模板。

| 入口 | 主表内容 |
| --- | --- |
| `eval_table1.py` | 方法对比：投影 Matched mIoU、投影 Boundary F1、两项人工评分及 Part Condition |
| `eval_table2.py` | 消融：2D Matched mIoU、2D Boundary F1、2D Part F1@0.5、投影 Matched mIoU |
| `eval_table3.py` | 效率与可靠性：总耗时、峰值显存、运行失败率、塌缩率 |

表 1 的 `Proj. mIoU` 也是下述一对一匹配口径，建议论文表头完整写成 `Proj. Matched mIoU`，不要将其与另一种 IoU 协议混用。

## 2. 固定测试清单与输入

manifest 是 UTF-8 CSV，至少两列：

```csv
group,uid
```

填入你真实固定测试集的全部对象，不从预测文件反推测试集，也不删除失败对象。`groups` 必须使用 manifest 的实际组名。示例中的三个组名仅为命名约定，不声称已经加载任何样本。默认先在每个组内按对象取平均，再对各组等权平均；同时输出 pooled object mean 供检查。

### 标签图

- 首选二维整数 NPY 或整数 PNG（如 uint16；P 模式 PNG 读取调色板索引）。
- 默认背景 ID 为 0，参考图中 -1、65535 为 ignore，可在配置中修改。
- 同一 ID 的不连通区域仍是同一部件。脚本不擅自拆连通域、补掩码、缩放或改标签。
- RGB/RGBA 可视化图必须提供精确 `palette`：`{"0":[0,0,0],"1":[255,0,0]}`，或包含该对象的 JSON 文件路径。RGBA 当前只按 RGB 解码，因此透明背景也必须具有可匹配的 RGB 值；优先提供整数图。
- 不能从有光照、烘焙插值的 GLB 截图中直接量化颜色当作部件真值。

对已有 v5 输出，可使用实际目录中的整数图：

```json
{"labels": "masks/YOUR_ACTUAL_V5_SUFFIX/{group}/{uid}/labels_uint16.png"}
```

或者读取 annotation JSON 中已经记录的路径：

```json
{"annotation": "annotations/YOUR_ACTUAL_V5_SUFFIX/{group}/{uid}.json", "labels_key": "final_map.labels_path"}
```

annotation 内的标签相对路径以实验根目录解释；若日志搬家后仍记录旧服务器绝对路径，改为明确的 `labels` 模板。

### 投影与相机

`reference.guidance` 为独立人工 D 的二维部件标签；`reference.projection` 为该视角的参考部件标签。两者在确实使用同一张 D、同一像素坐标时可以指向同一文件。各方法的 `projection` 必须来自最终三维结果的离散部件投影，不是输入二维引导图。

投影比较要求相同视角、分辨率、内外参、坐标系及几何规范化。脚本检查图像尺寸和已提供的相机数值，但不能仅凭相机文件证明网格已正确对齐。若 TRELLIS 结果重新居中、缩放，需在投影时使用正确变换；不能通过调整每个方法的相机来追逐 D 标签。

相机当前支持 JSON，字段如：

```json
{"transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "camera_angle_x": 0.7}
```

这是结构示例，不是你的真实相机。也支持 `camera_to_world`、`world_to_camera`、`extrinsic`、现有投影脚本的 `c2w + K + image_size`，以及 `intrinsics` 或 FOV 等字段；支持带 `frames` 的 transforms JSON，通过 `camera_index` 选视角。预测和参考必须使用同一种字段约定；脚本不自动转换内外参。其他相机格式待拿到真实文件后适配。

原始 Objaverse GLB 没有部件标注，所以不能由此直接计算 3D part mIoU。当前投影指标衡量已标注单视角上的一致性，不证明背面或完整表面的三维分割正确。

## 3. 指标定义

所有质量指标内部为 [0,1]，主表乘 100。颜色和部件 ID 不要求跨方法对应。

- **Matched mIoU**：以部件像素 IoU 作权重，匈牙利一对一最大总 IoU 匹配，总和除以参考部件数；未匹配参考部件计 0。额外预测部件主要由 Part F1 反映。
- **Boundary F1**：沿用上述 IoU 匹配，计算匹配部件边界的精确率和召回率，再按参考部件平均；漏掉的参考部件计 0。8 邻域形态学边界，欧氏距离容差默认 2 像素，ignore 邻域排除。必须固定评估分辨率及该容差。
- **Part F1@0.5**：以 IoU≥0.5 为 TP 条件的一对一最大匹配数量，`2TP/(N_pred+N_ref)`；同时惩罚多分与漏分。为避免表头和计算不一致，表 2 固定阈值 0.5。
- **补充 `gt_best_iou`**：每个参考部件独立选择最佳预测掩码，可复用预测。单独写入 JSON，不混入 Matched mIoU 列。

公开 [PartField `compute_metric.py`](https://github.com/nv-tlabs/PartField/blob/main/compute_metric.py) 的 GT-best overlap 允许复用预测，且其脚本可以搜索聚类层级。本实现的补充值不搜索层级，因此不声称复现完整官方协议。SegViGen 的公开 [README](https://github.com/Nelipot-Lee/SegviGen) 未提供独立评估命令；其 [论文](https://arxiv.org/html/2603.16869v3) 中交互式 IoU@N 的 N 是三维点击数，也不是这里的单视角指标。

缺失输出默认使受影响的组均值和 macro 为 `TBD`，不会偷偷只平均成功样本。只有确认整个待评方法已跑完，缺失文件均为真实失败时，才可选择 `--missing-policy zero` 将缺失预测按空预测计 0。未开展的实验不能用该开关填零。参考损坏、相机不一致等错误不能按失败输出填零。

退出码 0 表示执行完报告生成，不保证数据齐全；退出码 2 表示输入错误。使用主表前必须检查 JSON 中的每组数量、逐样本状态及 `input_errors`。

## 4. 人工评分

真实 CSV 字段为：

```csv
method,group,uid,rater_id,plausibility,boundary_quality
```

评分 1–5；method 使用配置的 key 或完整 name。先对同一对象的评分者平均，再组内平均、组间等权平均。重复评分记录或超范围分值报错。没有评分时保持 `TBD`，脚本不生成“主观分”。请自行固定盲评顺序、视角及评分准则。

## 5. 耗时、显存与失败日志

每个 `run_components` 项读取一段不重叠的实测流程。使用一个覆盖全流程的 `pipeline` 日志，或者相互不重叠的 `guidance`、`segmentation` 等分段；不能把总流程日志和已包含的子阶段一起相加。

规范化后的单条 JSONL 结构：

```json
{"group":"YOUR_GROUP","uid":"YOUR_UID","run_id":"YOUR_RUN","status":"success","timing_seconds":{"total":123.4},"peak_gpu_memory_gb":12.3,"stage_status":{"segvigen":"success"},"collapsed":false}
```

数值只是格式示例，不是实验结果。配置可用点分字段覆盖 `time_key`、`memory_key`、`status_key`、`uid_key`、`group_key`、`collapse_key`。`memory_unit` 支持 GB、GiB、MB、MiB、bytes；输出 GB 为十进制。选择唯一测量日志或 `run_id`，重跑记录重复时明确报错，不任意取最快一次。

- 总耗时：样本各分段墙钟秒数之和，再按固定 manifest 的所有对象平均。共享 TRELLIS 重建不计入方法对比，但各自 Qwen/SAM2、反馈、传播与后处理要计入。固定硬件、精度、缓存和计时边界，GPU 计时要同步。
- 峰值显存：分段峰值取最大，再在固定测试集取最大；不是各阶段显存相加。要求相同计量方式（例如统一 allocated 或 reserved）、相同 GPU / 精度，分段开始前重置峰值计数器。
- `skipped_existing` 的零耗时不计入；应提供最初实际运行的记录。缺日志、未跑三维阶段、未记录显存均为 `TBD`。
- 默认 `partial` 仅可视为 guidance 阶段完成；segmentation / pipeline 的 `partial` 不当作有效 GLB。只有确认具体日志语义后才可覆写 `success_statuses`。
- 失败率：明确失败对象数 / manifest 总对象数；未知状态不偷偷计成功。
- 塌缩率：已确认有效三维输出中的明确塌缩数量 / 有效输出数。当前读取记录中的布尔判定，不凭“有几种 RGB 颜色”或单视角投影自动断言整个三维模型塌缩。必须预先固定全表一致的塌缩协议，再填写该字段；缺判定为 `TBD`。JSON 同时记录分母及判定覆盖率。

现有 v5 guidance JSONL 的 `status=partial`、`stage_status.segvigen=not_run` 只证明生成了二维引导，不能单独用于表 3 的完整三维耗时及成功率。

## 6. 目录适配还需要哪些真实路径

提供实验根目录、固定 manifest、人工 D 整数标签、各方法 GLB 投影标签及相机、B/v4/v5/单因素消融的整数标签，以及实测日志。v4 的运行配置须明确；论文使用配置名称，不将内部版本号当成机制解释。

示例方法顺序沿用已商定的表格，不在此验证外部方法是否已运行或是否公平适配。所有方法需使用相同生成 GLB，必要的文本/提示来源应如实填写 `part_condition`。D 是独立人工参考，不计为待比较的自动方法。

## 7. 测试

```bash
python -m pip install pytest
python -m pytest tests -q
```

测试中的小数组及计时数字仅为工程夹具，禁止作为论文结果。旧 `evaluate_guidance_masks.py` 未改动，不要将其可能不同的匹配归一化口径与新主表分数混合。

# 评估适配验证记录

日期：2026-10-19（Asia/Shanghai）。未连接服务器批量评价，没有生成真实论文分数。

## 本次改动

- `paper_table_runner.py`：逐对象 annotation 调色板、显式 palette-index 模式及身份披露、按 arm/stage 等字段过滤日志、忽略 waiting/cached、固定清单数量检查、缺 D 单独记录、分段开销不冒充端到端开销、c2w 相机兼容。
- `config.objaverse684.json`：用户提供的共同根目录、684 冻结清单、D 整数标签，以及 B/v4/v5 输入约定。
- 新增对应适配测试；三份 CLI 与指标核心继续复用，未修改模型推理或投影代码。

## 评估相关测试

```text
python -m pytest tests/test_eval_objaverse_adapters.py tests/test_eval_table_commands.py tests/test_paper_table_metrics.py -q --tb=short --basetemp=work/eval_objaverse_green
32 passed in 40.57s
```

此前观察到缺失适配能力对应的失败，再实现并验证通过。测试使用生成的小数组、日志和相机，不是实验结果。

## 全项目测试

```text
python -m pytest tests -q --tb=short --basetemp=work/eval_objaverse_full_verified
3 failed, 62 passed, 1 warning in 43.57s
```

三个失败均位于本次未修改的 `tests/test_project_glb_labels.py` / `scripts/project_glb_labels.py` 路径：

1. `test_atomic_bundle_commits_v00_as_final_cache_sentinel`：写临时 PNG 时 `os.fsync(handle.fileno())` 报 `OSError: [Errno 9] Bad file descriptor`；尚未诊断其根因。
2. `test_process_one_reports_missing_and_cached_without_renderer`：所加载的投影模块当时没有 `process_one` 属性。
3. `test_egl_flat_render_preserves_two_texture_palette_colors`：本地测试环境缺少 `trimesh`，未进入真实渲染验证。

另有投影 PNG 写出代码的 Pillow `mode` 参数弃用警告。以上不以 mock、跳过测试或改动其他模块掩盖；不能声称完整项目回归全绿。投影模块可能仍在其他工作流开发中，本记录只描述此次运行时的状态。

验证环境使用项目内 `work/eval_table_deps` 的依赖和已配置的 Python 3.12；评估脚本的正式依赖列在 `scripts/requirements-evaluation.txt`，不要求安装 EGL 或 GPU 推理栈。

## 配置与交付检查

- JSON 可解析；表 1/2/3 各保留 7 个约定行。
- 分组数量约束为 assemblies=220、solid_volume=232、thin_fine_grained=232。
- 投影未提供，因此服务器配置不编造 projection 路径；v5 消融和外部方法未提供的输入也保留未配置。
- D 的约 642/684 覆盖率未自行复核；当前未采用有效 D 子集来替换固定清单。
- B 的 palette-index 分区不能证明原始部件 ID 一对一。正式使用实例级指标前仍需确认颜色未复用。
- 调色板绝对路径来自现有 v5 runner 默认值；v4 box-point 的 annotation 路径与 v5 full 的 schema 是当前代码约定，仍待服务器逐文件验证。
- `peak_allocated_bytes` 按十进制 GB 转换，但还应核对原日志是否在每个被测阶段开始前重置峰值；等待记录携带的残留峰值不进入指标。

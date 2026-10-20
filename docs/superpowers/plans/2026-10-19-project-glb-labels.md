# Project GLB Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个可批量把 SegViGen 纹理分割 GLB 投影为确定性 `uint16` 部件标签图与严格复现相机文件的 Python 脚本。

**Architecture:** 单文件 CLI 负责 manifest 遍历、GLB/EGL 平涂渲染、调色板最近邻量化、annotation 映射、相机/元数据写出和 JSONL 事件日志。渲染相关依赖延迟导入，使相机、映射、量化和原子写等纯逻辑可以独立测试；每个对象独立提交，`label_v00.png` 最后 rename，作为完整缓存哨兵。

**Tech Stack:** Python 3.10、NumPy、Pillow、trimesh 4.11.2、pyrender 0.1.45、OpenCV（可用但核心量化不依赖聚类）、pytest。

**Spec:** 当前会话中用户于 2026-10-19 提供的 `project_glb_labels.py` 完整需求。

## Global Constraints

- 服务器解释器固定为 `/data0/data/3dnew/envs/trellis2_py310/bin/python`，无显示器，导入 OpenGL 前设置 `PYOPENGL_PLATFORM=egl`。
- 相机严格使用 azimuth 45°、elevation 20°、yfov 42°、方形输出、指定的距离/姿态/znear/zfar 公式；多视角从 45° 起均匀覆盖 360°。
- 禁止 K-means；只允许无光照纹理平涂与确定性调色板最近邻，等距时选择更小的 palette index。
- 除新增 `projection/`、追加 `logs/projection_events.jsonl` 和新增脚本外，不修改任何已有实验文件。
- PNG/JSON 先写同目录临时文件后 rename；多视角完成时最后提交 `label_v00.png`。
- 单个对象失败不得中断批次；缺 GLB 记 `skipped_missing`；已有 v00 且未覆盖记 `skipped_cached`。
- `--allow-overwrite false` 必须解析为 False，不能使用 Python 的 `bool("false")`。

## Review Focus

- annotation 缺失或 palette 取模冲突：缺失采用匿名 palette-ID 回退并披露，冲突判失败。
- TextureVisuals 受材质因子、线性采样或抗锯齿影响：内存中固定白色 baseColorFactor、最近邻 sampler，并使用阈值量化。
- EGL 设备与 `--device cuda:N` 不一致：延迟导入 pyrender，并在导入前设置 `EGL_DEVICE_ID=N`。
- 多视角中途失败：不得留下 v00 缓存哨兵；下次运行必须可恢复。
- D 参考标签编号不同：按 Hungarian IoU 做匿名部件匹配，禁止直接比较编号。

---

### Task 1: 纯函数与失败测试

**Files:**
- Create: `tests/test_project_glb_labels.py`
- Create: `scripts/project_glb_labels.py`

**Interfaces:**
- Produces: `parse_bool`, `view_azimuths`, `camera_pose`, `camera_parameters`, `load_palette`, `load_label_mapping`, `quantize_palette`, `palette_indices_to_labels`。

- [ ] **Step 1: 写失败测试**：覆盖布尔解析、N=1/N=4 方位角、c2w 正交与目标位于相机负 z 轴、K、阈值内外、等距 tie、背景、annotation 映射与碰撞。
- [ ] **Step 2: 运行 `pytest tests/test_project_glb_labels.py -q`**，确认因实现文件或函数缺失而失败。
- [ ] **Step 3: 写最小纯函数实现**，量化使用分块整数平方距离，排除 palette index 0。
- [ ] **Step 4: 重跑定向测试**，确认全绿。

### Task 2: 相机、EGL 平涂渲染与投影元数据

**Files:**
- Modify: `scripts/project_glb_labels.py`
- Modify: `tests/test_project_glb_labels.py`

**Interfaces:**
- Consumes: Task 1 的相机与量化接口。
- Produces: `load_textured_scene`, `prepare_pyrender_scene`, `render_views`, `build_projection_meta`。

- [ ] **Step 1: 增加失败测试**：空/退化 bounds、全空前景、RGBA/depth 前景规则、元数据字段和映射命中统计。
- [ ] **Step 2: 运行测试并确认预期失败**。
- [ ] **Step 3: 实现延迟 pyrender 导入、TextureVisuals 检查、`FLAT|RGBA` 渲染、白色材质因子和最近邻纹理 sampler；按 depth 与 alpha 取前景。
- [ ] **Step 4: 运行定向测试并确认全绿**。

### Task 3: 批处理、原子提交、缓存与参考校验

**Files:**
- Modify: `scripts/project_glb_labels.py`
- Modify: `tests/test_project_glb_labels.py`

**Interfaces:**
- Produces: `load_manifest`, `reference_consistency`, `process_one`, `append_event`, `main`。

- [ ] **Step 1: 增加失败测试**：manifest 唯一性与路径安全、缺 GLB、缓存、字符串 false、多视角提交顺序、reference 换号匹配、单对象失败后继续。
- [ ] **Step 2: 运行测试并确认预期失败**。
- [ ] **Step 3: 实现 CLI 与逐对象 try/except；输出 cameras/meta/labels，最后 rename v00；日志加 `flock` 并逐行 flush/fsync。
- [ ] **Step 4: 运行 `pytest tests/test_project_glb_labels.py -q` 及完整 `pytest -q`，记录任何与本改动无关的既有失败。

### Task 4: 服务器兼容性与 EGL 冒烟验证

**Files:**
- Create on server: `/data2/codefile/fuyulin/experiments/objaverse_684/scripts/project_glb_labels.py`

**Interfaces:**
- Consumes: 完整 CLI。
- Produces: 可在指定 trellis2 环境运行的服务器脚本。

- [ ] **Step 1: 用 `py_compile` 和 `--help` 验证服务器解释器兼容性**，不渲染数据集。
- [ ] **Step 2: 在 `/tmp` 创建纯色纹理合成 GLB，运行单对象 EGL 渲染函数，确认非空 alpha/depth、标签为 uint16 且内部像素命中预期颜色。
- [ ] **Step 3: 只读抽取一个真实 GLB 做内存渲染检查；正式实验输出需用户另行启动批处理。
- [ ] **Step 4: 核对远端脚本 SHA-256 与本地一致，并报告命令、验证结果及尚未覆盖的真实数据风险。

## Self-Review

- 规格覆盖：输入路径、相机、确定性量化、映射、输出、日志、缓存、可选 reference、健壮性和禁止事项均有对应任务。
- 占位符检查：无 TBD/TODO；未把未定义行为留给实施者。
- 类型一致性：所有标签图均为二维 `np.uint16`；相机矩阵为 `float64 4×4`；调色板为 `uint8 P×3`。
- 审查重点均已绑定到 Task 1–4 的具体测试。

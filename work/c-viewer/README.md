# PartScope · C 组三维语义部件查看器

代码发布说明：真实消防车样本未包含在仓库中，需自行加载同源资产。`npm test` 运行 55 项合成数据单元测试；`npm run test:browser` 是独立浏览器集成测试，须先配置测试文件中的 Playwright 与浏览器路径。

已接入真实 GLB 的表面颜色，而不是演示椅子的 mesh 名称。

## 启动与使用

在本目录运行：

```powershell
node serve.cjs
```

打开 http://127.0.0.1:4173/ 。Three.js 0.162.0 已保存于 vendor/three（附 MIT LICENSE），运行无需 CDN。请勿直接通过 file:// 打开。

- 点击“消防车样本”加载本地保存的 C_v5 r04 对应模型、二维标注和下载来源记录。
- 或点击“加载文件”，在**同一次选择**中选一个 segmented.glb、一份二维 annotations.json，以及可选的 provenance.json。
- 缺少完整来源记录时，须自行核对同源并勾选确认；有明确哈希冲突时禁止绑定。
- 点击三维表面或右侧部件列表，整组对应面片会高亮，显示实际 label_id、英文名称、面数和记录。
- “导出三维绑定”输出每个 mesh 的 face_label_ids；0 表示未匹配。不会修改源 GLB、二维标注或服务器实验文件。

## 实际绑定链

`GLB mesh + faceIndex → 三角面原始纹理/顶点颜色 → 调色板 → label_id → parts.label 英文名称`

使用 GLB 的原始 base-color 数据，不从经过光照、色调映射的屏幕截图取色。支持索引/非索引几何、多 mesh、材质组、UV 变换和纹理通道；正确转换 glTF 的线性顶点颜色与 sRGB 纹理，并乘材质因子。

每面取 7 个内部样本。RGB 欧氏距离上限为 24，第一与第二候选距离差至少为 8；至少 80% 样本通过且不能出现两个已知标签，才将该面绑定。阈值是当前工程配置，**不是经数据集优化的科研指标**。近邻颜色、混合边界、远离调色板的表面保留 0，不强制归类。重复调色板颜色无法区分语义。

局限：有限采样可能遗漏三角形内的细小边界；不可读/压缩纹理、骨骼或形变、实例化、alphaMap、alphaTest 等目前保守标记未匹配。未对颜色已相近或语义错误的 SegViGen 输出进行纠错。默认限制 200 万面；大模型需要额外性能评估。

**颜色匹配率不等于分割 IoU、语义准确率或部件召回率。** 本功能是在已有输出上恢复可追踪的颜色关联，不是重新进行三维语义预测。

## 输入格式

二维 annotations.json 的每个 part 必须有唯一正整数 label_id 和 0–255 整数 RGB：

```json
{
  "uid": "sample-uid",
  "run_id": "actual-run",
  "final_map": {"sha256": "<引导图完整 SHA256>"},
  "parts": [
    {
      "part_id": "part_05",
      "label_id": 4,
      "label": "Front wheel",
      "color": [199, 229, 22]
    }
  ]
}
```

part_id 是规划记录标识，不应按数字后缀猜测 label_id。不得把 results 目录中 parts:[] 的占位 JSON 当作二维标签来源。

名称字段统一为 `label`，值只保留英文。导入旧标注时会将 `label_en` 转为 `label`，移除 `label_zh` 等中文名称字段；显示、复制 JSON、部件导出和三维绑定导出均使用该格式。原始导入文件的哈希仍按原始字节验证，语言字段清理在解析后执行。

已有实验标注可用项目中的 `scripts/english_only_annotations.py --input 原标注目录 --output 新导出目录` 批量转换。该脚本只处理 JSON 名称字段，源文件、GLB、标签图和引导图保持原样。转换后的 JSON 哈希会改变；如另有声明标注哈希的 provenance，需记录原始哈希和转换过程，并更新新标注文件的哈希。

可选 provenance.json 使用明确的输出字段：

```json
{
  "uid": "sample-uid",
  "output_glb_sha256": "<segmented.glb 完整 SHA256>",
  "guidance_sha256": "<该模型使用的引导图完整 SHA256>",
  "annotation_sha256": "<可选，标注文件完整 SHA256>"
}
```

旧流水线 segmented.glb.meta.json 的 glb_sha256 可能指**输入几何 GLB**；查看器不会把它当输出分割 GLB 的校验值。来源校验表示与所选来源记录一致，并非外部签名认证，更不证明语义正确。

## 随附的真实样本

- UID：20856d9731014b8f9c1010f2867978dc（消防车）。
- 2D：C_v5_feedback_pilot5_r04 的 15 部件标注。
- 3D：服务器 results/C_v5_new/assemblies/<uid>/segmented.glb。
- 原始 .meta.json 保留不变；其中引导图 SHA256 与二维标注 final_map.sha256 相同。
- 独立下载记录 assets/firetruck/provenance.json 保存经过服务器及本地双端核对的输出 GLB 完整 SHA256，并记录当前英文标注文件 SHA256。annotation_transform 记录原始标注哈希与名称字段转换；GLB、引导图、label_id 和颜色未变。
- 本样本用于软件接入验证，不用于宣称 C 组实验准确率。

### 当前真实样本的限制（2026-10-19）

实测原始纹理与二维调色板仅匹配 **11 / 94,224 面（0.0117%）**，仅命中 label 5。不能据此声称消防车已完成可靠语义绑定，偶然颜色命中也不能证明名称正确。界面会提示“匹配不足”，未匹配部件保持灰色。

独立解码 PNG 并在真实 UV 采样得到相同结果；常见 gamma 逆变换也不能解释全部颜色。诊断文件：qa/firetruck-color-diagnosis.json，脚本 qa/diagnose-firetruck-colors.mjs。这说明当前生成纹理不能直接充当离散语义 ID。**不应放宽颜色阈值来追求高覆盖率。**

要为这一例建立可信标签，需要人工确认三维区域对应关系，或加入有相机/可见性约束的二维标签投影，再保存显式 face_label_id。前端的加载、点选、高亮和导出接口已可承接这一步，但不能替代缺失的语义对应证据。低于10%的界面警告仅为工程提醒，不是论文评价标准。

## 导出与代码

导出 schema 为 partscope.face-labels.v1，含源 GLB/标注哈希、uid/run_id、来源检查状态、阈值、匹配统计、原始部件记录、mesh 对应 glTF node/mesh/primitive 信息及逐面 label_id。面序号遵循加载后 geometry 的索引顺序，**重网格化、简化、合并或改变拓扑后必须重新绑定**。当前不把标签嵌入 GLB，也不自动重新导入这个导出文件。

- semantic-core.mjs：标注解析、颜色匹配、保守面标签决策。
- semantic-three.mjs：真实纹理/顶点采样、mesh–face 标签数组、高亮几何。
- provenance.mjs：文件来源检查与内嵌 GLB 预检。
- viewer.mjs：文件加载、拾取、UI 联动、导出。
- serve.cjs：仅监听 127.0.0.1 的本地静态服务。

## 测试

```powershell
node --test *.test.mjs
```

纯逻辑测试无需 GPU。浏览器集成测试使用本机 Playwright + Edge 软件 WebGL；其依赖路径在 viewer.integration.test.mjs 中针对当前机器设置，其他机器需调整。集成测试使用人工构造的双色面板验证真实面片点击、列表、导出和未知拒绝，再加载真实消防车。测试产物在 qa/；合成面板不能作为论文实验结果。

# MergeNet 结果回传数据合同（E1/E2/E3/E6）

本文档供持有 ImageNet 与 checkpoint 的学长/CC 及其 AI 助手执行。目标是把**可分享的实验结果**交回本地，使作者能自行核对数字、计算配对置信区间、重画论文表格和路由图，而不是只能引用截图。它补充 `EXPERIMENTS.md`；实验定义、smoke gates 和正式运行协议仍以后者为准。

## 0. 安全边界与交付优先级

1. 不回传 checkpoint、模型参数、optimizer state、训练数据、ImageNet 原图、逐图完整 logits、公司内部路径、用户名、主机名、密钥或原始 1,000 图 trace NPZ。原图 crop 只有 CC 明确批准才作为**可选**图片结果交付；否则论文图画在 28×28 网格上。
2. 可回传匿名逐图**数值结果**：top-1 预测与 loss、少量预选案例的路由边、最终 carrier slot/mass、聚合直方图。这些是分析数据，不包含权重。若公司政策连这些也不允许，请在 `BLOCKED.md` 写明可提供的最高粒度，不能用假数据替代。
3. 分三个等级交付。**L0**：论文成图/表、聚合数值和运行收据；**L1（强烈建议）**：匿名逐图评测及 E6 图表底层数据，可在作者本地重算统计；**L2（经批准才给）**：预选案例的 224×224 eval crop。优先争取 L1，L2 不构成实验完成条件。
4. 所有回传文件必须是 UTF-8 CSV/CSV.gz、JSON 或仅含普通数值数组的 NPZ；不要用 pickle、`.pth`、`.pt` 或 Python 对象数组。CSV 浮点至少 8 位有效数字，JSON 不写 NaN/Infinity。缺项用空值并在状态文件解释，不能填 0。
5. 一个独立的 `share_packet_<UTC日期时间>.tar.gz`；解压后顶层只有下面的 `share_packet/`。先在 CC 机器上完成脱敏扫描、解压演练和 SHA-256 核对。不要把本地完整 `outputs/` 直接打包。

## 1. 跨实验身份约定

所有 E2/E3/E6 逐图文件使用相同 `image_id`。CC 机器本地创建并永久保留一把随机私钥，用 `HMAC-SHA256(key, ImageNet-val 相对 POSIX 路径)` 的前 32 个十六进制字符作为 `image_id`；私钥和 `image_id → relative_path` 映射**不得回传或提交**。不要使用无密钥 SHA 充当隐私保护。内部 E6 选图仍按 `EXPERIMENTS.md` 的既定 public-digest 规则，不改变选择；回传时仅用 `image_id` 连接文件。

每个实验结果必须共享以下身份字段，放在 `run_receipt.json` 和每个 CSV 的对应列：

| 字段 | 要求 |
|---|---|
| `schema_version` | 固定为 `1.0`；不兼容修改须升版本 |
| `experiment_id`, `run_id` | 如 `e2`, `e2_224_bestema_20260923`；稳定且唯一 |
| `git_commit`, `dirty` | MergeNet 实际执行 commit，正式数据要求 `dirty=false` |
| `upstream_commits` | DTEM/ToMe/OpenToMe 的实际 HEAD；不用的方法写 `null` |
| `checkpoint_sha256`, `state_key`, `epoch`, `selection` | 只记录权重摘要/加载身份；`selection` 为 `best_ema`、`final_ema` 等 |
| `data_split`, `data_count`, `class_list_sha256` | `imagenet1k_val`、`50000`、两 split class-list 收据 |
| `img_size`, `patch_size`, `crop_pct`, `interpolation`, `precision` | 评测实际值，不能只抄 YAML 默认 |
| `software`, `hardware` | Python、PyTorch、timm、CUDA、flash-attn、driver、GPU UUID/型号；UUID 可哈希脱敏 |

`run_receipt.json` 是按 `run_id` 索引的记录集合，另需有完整的已解析非敏感 config、启动命令的脱敏版、world size、batch、TF32/attention backend、开始/结束 UTC、环境包版本、代码/config/adapter 文件 SHA-256、失败与重跑记录。给私有文件只报摘要和语义，不报路径。每个指标文件的单位、聚合口径与来源摘要统一写入 `README.md` 或 `run_receipt.json`，不要求把这些说明列重复塞进每一行 CSV；绝不只有 Excel 手填总表。

## 2. 建议的回传目录

```text
share_packet/
  README.md
  SCHEMA_VERSION.txt                   # 1.0
  run_receipt.json
  dataset_receipt_redacted.json
  file_manifest.json                   # 每个相对路径、字节数、SHA-256
  e1/
    resolved_config.json
    epoch_metrics.csv.gz               # 300 epoch 的完整记录
    checkpoint_receipts.json           # best/final EMA，只含摘要/状态
    endpoint_eval.json
    token_trajectory.csv
    train_eval_trace_gate.json
  e2/
    checkpoint_manifest.json
    eval_summary.csv                    # 五方法完整 50k 汇总
    per_image.csv.gz                    # L1；5×50k 行
    token_trajectory.csv
    latency_raw.csv.gz                  # 两 batch × 五方法 × 1000 样本
    memory_raw.csv
    benchmark_protocol.json
  e3/
    ablation_summary.csv
    per_image.csv.gz                    # L1；4×50k 行
    paired_bootstrap.json
    integrity_gate.json
  e6/
    trace_config.json
    equivalence_gate_redacted.json
    sample_selection_receipt.json
    per_image.csv.gz                    # L1；1000 行
    per_image_layer.csv.gz              # L1；6000 行
    distance_hist_per_image_layer.csv.gz # L1；固定 bins
    final_carriers.csv.gz               # L1；1000×392 行
    case_metadata.csv                   # 固定案例；不含路径
    case_edges.csv.gz                   # L1；固定案例六步实际边
    selection_frequency_28x28.csv       # 784 行，计数/1000
    aggregate_summary.json
    figures/paper_routing_main.pdf
    figures/paper_routing_main.png
    figures/paper_routing_population.pdf
    figures/paper_routing_population.png
    figures/caption_draft.md
    images/<image_id>.png               # L2，批准后才有；实际 eval crop
  STATUS.json                           # done/partial/blocked，每项原因
```

`file_manifest.json` 格式为 `{"files": [{"path": "e2/eval_summary.csv", "bytes": 1234, "sha256": "...", "share_level": "L0", "redaction_status": "checked"}, ...]}`，不包含自身；其它所有文件逐项列出，路径相对 `share_packet/`。目录名不得包含内部用户/项目代号。PDF/PNG 是方便作者直接看，CSV/JSON 是核查与重画的权威来源；两者不一致时先查明原因，不以图片读数覆盖原始数据。

## 3. E1：matched DTEM 训练数据

`epoch_metrics.csv.gz` 每个实际 epoch 一行，至少 `epoch`, `train_loss`, `train_top1`（若记录）, `eval_loss_raw`, `eval_top1_raw`, `eval_top5_raw`, `eval_loss_ema`, `eval_top1_ema`, `eval_top5_ema`, `lr_start`, `lr_end`, `optimizer_updates`, `grad_scaler_skips`, `elapsed_seconds`, `checkpoint_saved`。如果某列原训练代码没有记录，保留空列并在 `README.md` 解释；不可从别的列估造。必须能检查 epoch 0–299 连续、数值有限、完整完成。

`checkpoint_receipts.json` 分别列 best/final EMA：SHA-256、实际 state key、保存 epoch、loaded/missing/unexpected key 数与非空明细、参数量、源代码 commit。`endpoint_eval.json` 给 best/final 的 raw/EMA 完整验证集 top-1/top-5/loss、样本数、预处理、checkpoint 身份及选择规则。`token_trajectory.csv` 逐层给训练 soft 路径与评估 hard 路径各自的 physical patch slots、logical active count、CLS 个数、attention/MLP 输入；不能把逻辑数当物理数。

最小 L0 可暂不发 300 行训练曲线，但那样本地无法审计训练终点，只能把 E1 标成 `summary-only`。建议直接给完整 CSV 和脱敏的关键启动/恢复日志片段，不给原始含路径的大日志。

## 4. E2：统一精度、延迟、显存

`eval_summary.csv` 五行，方法 ID 必须是 `dense,dtem,tome,pitome,mergenet`，每行包含 `checkpoint_sha256`, `state_key`, `epoch`, `selection`, `num_images`, `top1_pct`, `top5_pct`, `cross_entropy`, `final_patch_count`, `cls_count`, `preprocess_id`, `attention_policy`, `eval_precision`。Dense 的 final patch 为 784，其余目标 392，均以实际 hook 为准。

L1 `per_image.csv.gz` 为**长表**，五方法每个 `image_id` 各一行，至少：`method,image_id,label_index,pred_index,top1_correct,top5_correct,nll_loss`；correctness 列固定用整数 `0/1`，`nll_loss` 是该图自然对数单位的交叉熵。建议再给 `true_class_probability`、`top1_probability`、`top1_minus_top2_margin`（由 logits 在 CC 侧算出，不需要回传 logits）；这些能在本地分析失败样本和校准趋势。所有方法的 `image_id` 集合必须完全相同，计数各 50,000，不准因 DDP padding 重复图。

`token_trajectory.csv` 逐 `method,block,operation` 给 physical patch slots、CLS、logical active count（若适用）、reduce-before/after、实际 schedule。`latency_raw.csv.gz` 每个 CUDA-event 样本一行：`method,batch_size,round,iteration,latency_ms,valid_round,invalid_reason,gpu_id_hash,precision,attention_policy`；保留 outlier 和无效轮次，不删。`memory_raw.csv` 每种 method×batch 给 baseline/peak allocated/reserved MiB、input bytes、是否 OOM。`benchmark_protocol.json` 保存 warmup、轮次、计时/同步实现、排除项、功耗/时钟状态以及方法运行顺序。若 384px 扩展 E2，放独立 `e2_384/` 并用独立 `run_id`，不可与 224px 混表。

## 5. E3：同 checkpoint 的四行消融

`ablation_summary.csv` 四行：`native,mass_off,recovery_off,both_off`，含 `num_images`, `top1_pct`, `top5_pct`, `cross_entropy`, `delta_top1_pp`, `delta_top5_pp`, `ci_low_pp`, `ci_high_pp`, `checkpoint_sha256`, `preprocess_id`。L1 `per_image.csv.gz` 与 E2 共用 `image_id`，每个 variant×image 一行：`variant,image_id,label_index,pred_index,top1_correct,top5_correct,nll_loss`；可选概率/置信度字段同 E2。这样本地可以重做 paired bootstrap、逐类/难例分析，而不需 logits 或 checkpoint。

`integrity_gate.json` 至少记录四次 strict reload 结果、native 与 E2 logits 的最大误差、native 重复一致性、mass-off/recovery-off 的 top-K indices 完全相同与否、路由输入/输出的等价摘要、计数与 NaN/Inf。`paired_bootstrap.json` 包含 seed、10,000 resamples、采样单位 image、每个对照的原始差值与 CI；本地可用 L1 重算。若只给 L0 总表，不能独立验证配对区间。

## 6. E6：允许本地重画图的数值底稿

所有 E6 数据使用 **224px、patch 8、28×28 网格、6 个 local routing steps、K=392（CLS 不计）**，并与 E2 native MergeNet 的 checkpoint SHA、EMA key、preprocess ID 对齐。图只能基于 trace 等价 gate 通过后的数据生成。

### 6.1 全 1000 图 L1 底稿

- `sample_selection_receipt.json`：总数 1000、每类一张的验证结果、内部 manifest SHA-256、选图算法及 seed、正确/错误数、匿名案例选择规则；不含 relative path 或未加密原始 digest。
- `per_image.csv.gz`：每个 `image_id` 一行，列 `image_id,label_index,pred_index,top1_correct,final_k,final_unique_k,coverage_mean_patch,coverage_p95_patch,occupied_bins_7x7,boundary_retention,center_retention,final_mass_sum,final_mass_gini,top10pct_mass_share,nan_inf_count`。`label_index`/`pred_index` 若政策不允许可删，但 `top1_correct` 必须有。
- `per_image_layer.csv.gz`：每图×step 一行，列 `image_id,step,physical_patch_slots,logical_active_count,edge_count_positive,donor_count,receiver_count,distance_mean_patch,distance_p50_patch,distance_p90_patch,distance_p95_patch,mass_sum_pre,mass_sum_post,mass_conservation_abs_error,mass_p50,mass_p90,mass_p99`。`step` 用 1–6；所有 mass sum 与 mass 分位数只算 patch，不含 CLS；距离对实际 post-mask 正权边算，不能用 eligible edges 代替。
- `distance_hist_per_image_layer.csv.gz`：每图×step×bin 一行，列 `image_id,step,bin_left_patch,bin_right_patch,edge_count,transport_weight_sum`。bin edges 在 `trace_config.json` 预先固定，左右边界规则明确；各 bin edge_count 之和必须等于对应 `edge_count_positive`。这样本地能选择微平均/每图平均而不重新跑模型。
- `final_carriers.csv.gz`：每图必须恰好 392 行，列 `image_id,slot_index_0based,grid_row_0based,grid_col_0based,carrier_mass,selection_rank`。`selection_rank` 为 0–391，在每图内唯一；slot 是被 top-K 选中的**原始网格 slot**，非 center-of-mass。每图 slot 唯一，row=`slot//28`，col=`slot%28`，mass 有限且非负、每图总质量必须大于 0。作者本地可从该表重算 28×28 selection frequency、mass 分布、coverage 与 boundary 指标。
- `selection_frequency_28x28.csv`：784 行，`slot_index_0based,grid_row_0based,grid_col_0based,count_selected,frequency`；frequency=`count_selected/1000`，所有 count 之和应为 392000。该表是便捷缓存，权威来源仍是 `final_carriers.csv.gz`。

### 6.2 固定案例的可重画底稿

`case_metadata.csv` 至少包含预定正文一对正确/错误案例，建议再包含 12 张审查联系表案例；列为 `image_id,role,top1_correct,label_index,pred_index,selection_rule_rank,eval_crop_available,figure_use`，其中正文两张的 `figure_use=main`、其余为 `contact_sheet`。同一图片可能同时符合多个联系表角色，此时可有多个 role 行但只传一份边数组与 crop。角色由预冻结 manifest 与 correctness 自动决定；不允许本地重新挑“更好看”的案例。

`case_edges.csv.gz` 至少对正文两个案例保存**六个 step 的所有 `assign_postmask>0` 边**；联系表其它案例可按批准范围追加。每行：`image_id,step,donor_slot_0based,receiver_slot_0based,weight_postmask,donor_mass_pre,receiver_mass_pre`。只含数值与匿名 ID；避免直接传整块 hidden feature、attention logits 或原始图。CC 必须在实际 sparse transport 调用前、最终 physical mask 后捕获权重；边权范围 `[0,1]`，donor/receiver slot 不同，欧氏网格距离 ≤3（允许浮点容差），并核对每 step 的 edge 数与 `per_image_layer.csv.gz` 一致。`case_edges` 的完整边集合让我们在本地重新决定线型/透明度，但论文正文仍只画预定的每象限 8 条规则。

`trace_config.json` 记录权重 dtype、step 定义、非零阈值（默认严格 `>0`）、距离 bins、carrier mass 标度、preprocess/crop、选图和破同分规则、画图颜色/线宽映射、preview/real 标志。`equivalence_gate_redacted.json` 只给 8 张图的匿名 ID、max-abs/max-rel logit 差、pred/top-K 是否一致、每步质量守恒误差及阈值；不回传 logits。

### 6.3 图片与图稿

在 CC 侧用真实数据生成正文和附录 PDF/PNG；如图像 crop 获批可分享，`images/<image_id>.png` 仅放最多 12 个预选唯一案例的**实际 eval crop**，并在 `case_metadata.csv` 对应。若不获批，完全不发送图片，保留 `case_edges`、`final_carriers` 和 grid-only 图；这仍可在本地重画路由/载体图。图注草稿必须说明 1000 图选择、正确/错误固定规则、step 3、最多 32 条边、carrier mass 标度、图片授权状态及“不证明语义边界/因果收益”。`docs/visualization_preview.png` 只展示排版，不可当作实测图。

## 7. 到货后的本地核查清单

收到包后作者应只读取 `share_packet/`：

1. 核对 manifest 文件大小与 SHA-256；扫描绝对路径、密钥、pickle、checkpoint 后缀与未知大文件。疑似敏感内容先隔离，不公开转发。
2. 确认所有 run receipt 的 checkpoint SHA、preprocess ID、val split 与 E2/E3/E6 对齐；所有 upstream commit 与正式协议一致。
3. 用逐图 L1 重算 E2 五方法 top-1/top-5/loss 和 E3 四行 top-1/top-5/loss、配对差值与 bootstrap CI；与总表核对至原始精度。重算与交付不一致时停用该单元格。
4. 检查 E6 1000×6 层行数、1000×392 carrier 行数、所有唯一性/坐标/质量/距离 gate；从 `final_carriers` 重建 selection-frequency 热图，并从每图 histogram 重画总体分布。
5. 本地可做新的**描述性**分析：正确/错误分层、边界/中心留存、carrier mass 集中度与 coverage 的相关性、按类汇总、图片级 bootstrap。不能把这些后验分析写成预注册因果实验，也不能通过本地数字重建 checkpoint。
6. 若只收到 L0，仍可填经过协议核对的总表和放成图，但逐图复算、换图布局与新统计应标为 `Unable to verify from available artifacts`。不要把 L0 误写成已独立重放模型。

结构自检入口：`python scripts/validate_handoff.py /path/to/share_packet --level L1`（只读，不加载模型或权重）；脚本自身可先用 `python tests/test_handoff_validator.py` 验证。它检查文件哈希、行数、匿名配对 ID、top-1 与标签、E6 分层行数/直方图和 carrier 网格；**不能**代替对权重来源、实验公平性、图像授权或统计解释的人工审计。只有 L0 时可改 `--level L0`，并如实记录 L1 缺失。

校验通过后，本地可用 `python scripts/render_handoff_figures.py /path/to/share_packet --output /path/to/new_figure_dir` 重画正文网格版和附录总体图（需 `matplotlib`/`numpy`；离线自测为 `python tests/test_handoff_renderer.py`）。若包内有获批的 `images/<image_id>.png`，脚本会用其作浅背景；否则只画网格。输出采用 `*_local` 文件名，不覆盖 CC 原图；后续是否进入论文仍须人工核对图注与数据身份。

## 8. CC 发送前的完成门槛

- `STATUS.json` 为每项写 `done|partial|blocked` 和实际缺失原因；缺失 L1 不能静默用 L0 冒充。
- 所有 CSV 的 schema、单位、行数、排序、唯一键和有限值自动验证通过；保留验证器 stdout/stderr 为脱敏文本。
- 图 PDF/PNG 均能从同包数值与批准的图片重建；如图片不外传，numeric-only 网格版必须能在本地重建。
- 回传包内没有权重、原图（除获批的 12 个 crop）、内部路径或私钥；`file_manifest.json` 哈希全部通过。
- `README.md` 写清哪些是模型实际观测、哪些是派生统计、哪些只在 CC 本地可复核，以及所有协议偏差。

给 CC 的一句话：**请优先给我们匿名逐图数值和固定案例的边/载体数组，而不只是四张漂亮图片；checkpoint 和 ImageNet 数据始终留在你们机器上。**

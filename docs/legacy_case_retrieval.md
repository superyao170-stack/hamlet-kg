> 历史文档：以下是 v0.2 的案例检索说明，保留用于旧命令参考。当前检测主线、研究定位和状态以根目录 README.md 与 kais_research_protocol.md 为准。历史实验数字不代表新检测模型的效果。

# HAMLET-KG

**H**ierarchical **A**tomic-to-Risk Event **M**odeling with optional **L**LM-augmented
**E**vent **T**emporal **K**nowledge **G**raphs.

这是一个与银行生产系统完全解耦的研究原型，用于验证以下论文假设：

> 案例相似性不应由原子动作的表面重合或风险事件名称的简单一致决定，
> 而应由原子事件组成结构所增强的风险事件级语义与关系对齐决定。

项目不设置独立证据层。原始记录编号只作为原子事件的来源元数据，用于实验复现。

## 当前实现

- 类型化实体与原子事件模型；
- 稀疏时序关系和资金延续关系构建；
- 可执行的风险模式 JSON DSL；
- 基于 FATF、FinCEN、AUSTRAC 与图挖掘文献整理的七类 AML 风险模式 V1；
- CP-REM 原型：索引化滑动时间窗、候选主体剪枝和重叠窗口抑制；
- 可保留多个时间分离实例的确定性风险事件物化；
- 风险事件之间的时序、共享主体关系构建；
- 原子组成感知的风险事件相似度；
- 基于 Sinkhorn 的语义基线和熵正则 FGW 风格结构感知对齐；
- 风险类型倒排索引、候选案例粗排和 Top-K 精排；
- Recall@K、MRR、NDCG@K 评价；
- IBM 合成 AML 交易 CSV 适配器；
- 可选的 OpenAI 兼容 LLM 原子事件抽取和自然语言规则编译；
- 纯标准库实现、离线单元测试和最小可复现实例。

## 三层知识表示

```text
实体 + 原子事件
      ↓ 形式化模式匹配
风险事件（内部保留原子事件子图）
      ↓ 风险事件关系构建
案例风险事件图
      ↓ 层次化软对齐
相似案例排序
```

原子事件负责资金流、时间和动作计算；风险事件负责稳定的业务语义；案例检索主要发生在风险事件层。

## 快速开始

核心项目要求 Python 3.10 或更高版本，不需要联网安装第三方依赖；参考算法和实验笔记使用 `requirements-experiments.txt` 中的依赖。

```bash
cd hamlet-kg
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m hamletkg.cli demo \
  --examples examples \
  --output outputs/demo
```

也可以安装为本地开发包：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
hamletkg demo --examples examples --output outputs/demo
```

演示会：

1. 读取三个匿名化案例；
2. 在原子事件图上识别三种风险模式；
3. 物化风险事件并构建风险事件图；
4. 以 `CASE-A` 为查询案例完成相似案例排序；
5. 输出检索结果和 Recall@2、MRR、NDCG@2。

## 常用命令

### 物化风险事件

```bash
hamletkg materialize \
  --case examples/raw_cases/case_a.json \
  --patterns examples/patterns.json \
  --matcher windowed \
  --output outputs/case_a.json
```

`windowed` 是改进算法，`legacy` 保留为整段历史聚合基线。输出同时包含窗口、剪枝、去重和重叠抑制统计。

文献驱动的首版模式库可直接替换 `--patterns` 参数：

```bash
hamletkg materialize \
  --case examples/raw_cases/case_a.json \
  --patterns patterns/aml_risk_patterns_v1.json \
  --output outputs/case_a_risk_patterns_v1.json
```

这套规则将命中结果定义为候选风险事件，不定义为洗钱结论。数值参数是待验证集校准的研究默认值。来源映射、误报场景和 V2 待扩展模式见 [AML 风险模式库 V1](docs/risk_pattern_catalog_v1.md)。

### 比较两个已物化案例

```bash
hamletkg similarity \
  --query outputs/demo/case_a.json \
  --candidate outputs/demo/case_b.json \
  --algorithm fused
```

`fused` 同时考虑风险事件语义和风险图拓扑；`hierarchical` 是原 Sinkhorn 基线。

### 索引化 Top-K 案例检索

```bash
hamletkg rank \
  --query outputs/demo/case_a.json \
  --candidates outputs/demo \
  --top-k 5 \
  --algorithm fused \
  --candidate-pool-size 50
```

检索先用风险类型和风险关系摘要粗排，再只对候选池执行结构感知精确对齐。

### 评价案例检索

```bash
hamletkg evaluate \
  --cases outputs/demo \
  --relevance examples/relevance.json \
  --top-k 2
```

### 导入 IBM AMLSim 交易 CSV

```bash
hamletkg import-amlsim \
  --transactions data/raw/amlsim/20K_fanin200/transactions.csv \
  --nodes data/raw/amlsim/20K_fanin200/nodes.csv \
  --case-id AMLSIM-FANIN-001 \
  --limit 10000 \
  --output outputs/amlsim-fanin.json
```

适配器不会把 `isFraud` 或 `fraudStep` 放入模型特征，只会将其保存在评价元数据中，防止标签泄漏。

### 运行公开数据基线

```bash
PYTHONPATH=src python3 -m hamletkg.cli benchmark-amlsim \
  --data-root data/raw/amlsim \
  --output outputs/amlsim-baselines

.venv/bin/python scripts/run_reference_algorithms.py
.venv/bin/python scripts/run_improved_algorithms.py
.venv/bin/python scripts/build_experiment_notebook.py
```

本轮已经实现并运行度数异常、重复边时间特征、自适应低金额、fan-in、cycle 和模式联合六类节点级基线，以及 Hungarian、Sinkhorn、Fused Gromov–Wasserstein 和 VF2 参考算法。节点级高分受 AMLSim 合成金额分布影响，必须结合数据质量限制解读。

## LLM 增强功能

复制 `.env.example` 中的配置到自己的环境变量。密钥不能写入代码或提交到版本库。

```bash
export HAMLET_LLM_BASE_URL=https://api.example.com/v1
export HAMLET_LLM_API_KEY=your-key
export HAMLET_LLM_MODEL=your-model
```

从非结构化案例文本抽取原子事件：

```bash
hamletkg extract-case \
  --case-id TEXT-001 \
  --text /path/to/case.txt \
  --output outputs/text-001.json
```

将自然语言规则编译为风险模式 DSL：

```bash
hamletkg compile-pattern \
  --rule examples/rule.txt \
  --samples 3 \
  --output outputs/generated-pattern.json
```

LLM 不是主算法，只能生成候选原子事件或候选模式，不能直接创建正式风险事件。候选模式依次经过字段白名单、动作—角色、必需参数、数值范围和先后关系无环校验；首次失败时允许携带验证错误修复一次，再失败即终止。正式风险事件只能通过确定性匹配产生。当前已验证 DeepSeek 的 OpenAI 兼容接口可以编译出合法 DSL；密钥只放在被忽略的本地 `.env` 中。

`quality.valid_sample_rate` 表示多次编译的合法率，`quality.semantic_agreement` 表示忽略子句编号后的语义一致率。这两项不是洗钱概率；在独立事件标注集上完成校准前，`quality.confidence_status` 为 `uncalibrated`，物化风险事件的 `confidence` 为 `null`。

### 事件级评价

```bash
hamletkg evaluate-events \
  --predicted outputs/predicted-case.json \
  --reference data/annotations/reference-case.json \
  --minimum-atomic-iou 0.5
```

预测事件与标注事件需满足风险类型、中心实体以及原子事件集合 IoU 阈值，并通过最大二分匹配计算事件级 Precision、Recall 和 F1。得到独立验证集统计后，可用 `calibrate_pattern_reliability` 写入拉普拉斯平滑后的模式可靠度。

## 风险模式 DSL

模式由动作子句和约束组成：

```json
{
  "pattern_id": "P-FAN-IN-OUT",
  "risk_type": "FAN_IN_THEN_OUT",
  "description": "来自多个账户的资金汇集后集中转出",
  "clauses": [
    {
      "clause_id": "incoming",
      "action_type": "TRANSFER",
      "anchor_role": "receiver",
      "min_count": 3,
      "distinct_role": "sender"
    },
    {
      "clause_id": "outgoing",
      "action_type": "TRANSFER",
      "anchor_role": "sender",
      "min_count": 1
    }
  ],
  "constraints": [
    {
      "constraint_type": "ordered_within",
      "parameters": {
        "earlier": "incoming",
        "later": "outgoing",
        "max_seconds": 86400
      }
    }
  ]
}
```

除时间、金额总和和比例约束外，DSL 还支持 `attribute_range`（先按单笔属性范围筛选动作）和 `role_overlap`（比较两个动作子句的参与者集合）。当前 DSL 有意限制在围绕中心实体的有界聚合模式，避免将通用子图同构的高复杂度直接带入在线检索。后续论文工作可以扩展循环、多跳传递、否定条件和有界树宽模式。

## 相似度模型

风险事件对相似度为：

```text
S_r = 0.35 × 风险类型
    + 0.30 × 原子组成
    + 0.20 × 角色结构
    + 0.15 × 时序与数值轮廓
```

改进案例相似度为：

```text
S_c = 0.55 × 风险事件语义传输对齐
    + 0.20 × Gromov 风格关系结构一致性
    + 0.15 × 风险事件时序序列
    + 0.10 × 风险事件规模一致性
```

结构感知实现采用依赖无关的熵正则交替更新，POT 的 FGW 作为外部参考实现。当前权重是可复现起点，不应被直接作为论文最终参数；正式实验中应在验证集学习权重，并报告参数敏感性和置信区间。

## 项目结构

```text
src/hamletkg/
  models.py          三层数据模型
  graph.py           原子事件关系与风险事件关系
  patterns.py        模式 DSL 校验
  materializer.py    整段历史基线与索引化滑窗物化
  similarity.py      Sinkhorn、结构感知对齐与案例索引
  evaluation.py      检索评价指标
  llm.py             可选 LLM 增强接口
  adapters/          公开数据集适配器
  cli.py             实验命令入口
examples/            模式、原始案例和相关性标注
patterns/            文献驱动、带来源与阈值说明的风险模式库
tests/               离线单元测试
docs/                理论定义与 KBS 实验路线
```

## 研究边界

当前版本是论文方法的可运行骨架和公开数据基线，不是声称已经完成的 KBS 投稿成果。进入正式实验前仍需：

- 扩展模式语言和形式语义，并把当前 CP-REM 原型完善为流式实现；
- 实现循环、Relay、否定和多中心模式；
- 建立规则—模式人工标注数据；
- 为 AMLSim 建立事件级标注，并接入第二个外部验证数据；
- 增加学习式权重、图编码器与强基线；
- 完成显著性、鲁棒性、复杂度和消融实验；
- 验证模式编译的语义保持性质。

详细内容见 [AML 风险模式库 V1](docs/risk_pattern_catalog_v1.md)、[改进算法说明](docs/algorithms.md)、[理论设计](docs/theory.md)、[KBS实验路线](docs/kbs_experiment_protocol.md) 和 [专硕算法—应用路线](docs/applied_research_protocol.md)。

# HAMLET-KG 改进算法说明

## 1. Indexed Window Risk-Event Materialization（CP-REM 原型）

### 输入与输出

输入为原子事件图 (G_A) 和风险模式 (P)，输出为风险事件集合及完整匹配轨迹。模式仍限定为共享中心实体的有界聚合模式。

### 过程

```text
1. 建立 I[action_type, role, entity] -> sorted events 倒排索引
2. 对每个子句检查全局最小计数和 distinct_role 下界
3. 求所有子句可行中心实体集合的交集
4. 对每个候选实体：
   4.1 合并相关事件时间，作为候选窗口终点
   4.2 用二分边界取得 [end-W, end] 内的各子句事件
   4.3 先检查计数下界，再验证顺序、金额比例和持续时间
5. 对相同原子事件集合去重
6. 用 overlap coefficient 抑制高度重叠窗口
7. 将每个保留窗口物化为一个风险事件
```

一个账户在相距超过窗口宽度的两个时期满足同一模式时，会产生两个风险事件实例。旧的整段历史算法只产生一个实例，因此被保留为 `legacy` 基线。

设原子事件角色记录总量为 (N)，候选中心实体 (a) 的相关事件数为 (m_a)。索引构造为 (O(N\log N))（桶内排序），窗口边界查询为 (O(\log m_a))，约束验证取决于窗口内事件数。`ordered_within` 的当前精确验证仍可能产生窗口内两组事件的笛卡尔积，是下一步共享前缀/增量状态优化点。

### 可实验统计

输出 `total_entities`、`candidate_anchors`、`pruned_anchors`、`windows_evaluated`、`accepted_windows`、`duplicate_windows`、`overlap_suppressed` 和 `event_lookups`，可直接用于剪枝消融与扩展性图表。

## 2. 规则编译后的确定性防线

LLM 输出先经过严格载荷检查，再进入类型和语义检查：

```text
自然语言规则
  -> LLM JSON 候选
  -> 顶层/子句/约束字段白名单
  -> 动作—角色合法性
  -> 必需参数与有限数值检查
  -> min/max 可满足性
  -> ordered_within 有向无环检查
  -> 可执行 RiskPattern
```

若第一次候选失败，系统最多进行一次带验证错误反馈的修复调用；再次失败则终止，不会进入图匹配。可通过 `--samples N` 独立编译多次，并报告合法率与忽略子句编号后的语义一致率。该机制只提升规则编译可靠性，不允许 LLM 直接创建风险事件。

### 置信度分层

```text
编译质量：valid_sample_rate + semantic_agreement
    ↓ 只能说明 LLM 编译是否稳定
模式可靠度：独立标注集 Precision / Recall
    ↓ 经拉普拉斯平滑得到 calibrated_confidence
风险事件置信度：继承已校准模式可靠度
```

编译一致率不是风险概率。模式未经过独立事件标注集校准时，`confidence_status=uncalibrated`，所有物化风险事件的 `confidence=null`。完成校准后，当前实现使用 ((TP+1)/(TP+FP+2)) 作为平滑可靠度，并保留原始 Precision 与 Recall。事件级评价通过风险类型、中心实体和原子事件集合 IoU 构造候选边，再用最大二分匹配计算 TP、FP 和 FN。

## 3. Fused Graph Sinkhorn 风险事件图对齐

### 风险事件语义代价

风险事件对先由风险类型、原子动作/关系组成、角色结构、时间与金额轮廓得到相似度 (S_{ij})，语义代价为：

\[
M_{ij}=1-S_{ij}
\]

### 风险图结构代价

对 `SAME_ANCHOR`、`SHARES_ENTITY` 和 `PRECEDES` 赋予不同边长，并计算归一化最短路矩阵 (C_q,C_c)。给定当前传输矩阵 (T)，结构失真为：

\[
G_{ij}=\sum_{k,l}(C_q(i,k)-C_c(j,l))^2T_{kl}
\]

融合代价为：

\[
D_{ij}=\alpha M_{ij}+(1-\alpha)\widehat G_{ij}
\]

算法在融合代价上的 Sinkhorn 更新和结构代价重估之间交替，直到传输矩阵变化低于阈值或达到迭代上限。当前默认 \(\alpha=0.65\)。这是无第三方依赖的 FGW 风格近似；正式论文实验使用 POT FGW 作为标准参考实现，不能将二者写成完全相同的算法。

若两个案例分别含 (n,m) 个风险事件，一次结构代价计算为 (O(n^2m^2))。因此它只用于候选精排，不用于全库暴力比较。

## 4. 两阶段案例检索

```text
风险类型倒排索引
      ↓
类型计数 + 关系计数余弦粗排
      ↓ 取 candidate_pool_size
Fused Graph Sinkhorn 精排
      ↓
Top-K 相似案例 + 风险事件对齐解释
```

索引若无法提供至少 (K) 个案例，会安全回退到全部案例，避免召回结果不足。输出总候选数、索引命中数、实际精排数和剪枝数。

## 5. 当前实跑结果边界

- AMLSim fan-in 前 5,000 条交易：20,000 个实体被剪到 19 个候选中心实体，剪枝率 99.905%；窗口算法识别 2 个事件。
- 这两个事件的中心实体均不是 AMLSim 欺诈标签节点，说明当前由自然语言生成的规则与该数据切片没有校准好。算法不会为了提高指标而放宽规则或使用标签。
- 三个手工示例案例上，融合匹配仍把 CASE-B 排为 CASE-A 的第一相似案例；该结果只证明链路正确，不证明统计显著性。

完整比较结果位于 `outputs/improved-algorithms/results.json`。

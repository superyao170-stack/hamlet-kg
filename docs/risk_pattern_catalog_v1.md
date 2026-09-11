# AML 风险模式库 V1：文献依据与可执行定义

## 1. 定位

这套模式用于把原子事件图聚合为候选风险事件图。它不是洗钱定罪规则，也不是直接生成可疑交易报告的规则。单个红旗指标只能触发进一步分析；最终判断还需要客户身份、账户用途、历史基线、地域、业务场景等上下文。

V1 采用两类依据：

1. **业务语义依据**：FATF、FinCEN、AUSTRAC 发布的风险指标和监管提示；
2. **图结构依据**：Graph Feature Preprocessor 与 IBM AML 数据论文中使用的 fan-in、fan-out、gather-scatter、cycle 等结构。

规则文件位于 `patterns/aml_risk_patterns_v1.json`。其中每条模式都保存来源、所需字段、误报场景和阈值策略，能够从风险事件回溯到原子事件，但不额外建设“证据层”。

## 2. 文献到模式的转换原则

监管材料通常使用“快速”“多笔”“多个第三方”“接近阈值”等定性语言，不能直接当作数值算法。因此转换分三步：

1. 把自然语言指标拆成动作、角色、拓扑、时间和金额关系；
2. 用 DSL 表达确定性可执行模式；
3. 把定性词替换为**研究默认值**，再在验证集上校准。

本版默认使用 24 小时时间窗、至少 3 个不同交易对手、转出/转入金额比 0.8–1.2。它们是首轮实验参数，不是监管结论。现金拆分模式中的 10,000 美元仅是演示配置，换成中国或其他司法辖区数据时必须替换相应口径、币种和业务日定义。

## 3. 首版七类可执行模式

| ID | 风险模式 | 原子结构 | 关键约束 | 主要依据 | 易误报场景 |
|---|---|---|---|---|---|
| AML-V1-01 | 资金快进快出 | 入账 → 出账 | 24h；转出/转入 0.8–1.2 | AUSTRAC；FATF/Egmont | 清算、内部归集、证券结算 |
| AML-V1-02 | 多来源集中入账 | 多个发送方 → 锚点 | 24h；发送方至少 3 个 | FinCEN；AUSTRAC | 平台收款、众筹、小微商户 |
| AML-V1-03 | 多来源归集后转出 | 多个发送方 → 锚点 → 外部 | 24h；金额比 0.8–1.2 | FinCEN；AUSTRAC；GFP | 资金池、总分账户、批量清算 |
| AML-V1-04 | 多来源归集后取现 | 多个发送方 → 锚点 → 现金 | 24h；取现/入账至少 0.5 | FinCEN；AUSTRAC | 现金密集型商户、劳务结算 |
| AML-V1-05 | 入账后分散转出 | 外部 → 锚点 → 多个收款方 | 24h；收款方至少 3 个 | FATF/Egmont；GFP | 工资、理赔、供应链付款 |
| AML-V1-06 | 拆分现金存入 | 多笔低额现金 → 锚点 | 单笔低于阈值、合计达到阈值 | FinCEN；AUSTRAC；FATF/Egmont | 现金商户正常日结 |
| AML-V1-07 | U 形资金回流 | 对手方 → 锚点 → 同一对手方 | 3d；对手方重合；金额比 0.8–1.2 | AUSTRAC；FATF/Egmont | 退款、撤销、借款归还 |

## 4. DSL 本次新增能力

### `attribute_range`

先按属性范围筛选原子事件，再检查 `min_count`。它使“多笔单笔低于阈值的现金存入”可执行，而不会因同一窗口内混有一笔正常大额存款就整体失败。

```json
{
  "constraint_type": "attribute_range",
  "parameters": {
    "clause": "cash_deposits",
    "attribute": "amount",
    "maximum": 9999.99
  }
}
```

### `role_overlap`

比较两个动作子句中的角色集合是否有交集。它用于表达“原发送方又成为返回资金的接收方”，从而检测两跳 U 形回流。

```json
{
  "constraint_type": "role_overlap",
  "parameters": {
    "left_clause": "incoming",
    "left_role": "sender",
    "right_clause": "returning",
    "right_role": "receiver",
    "minimum": 1
  }
}
```

## 5. 暂不进入 V1 可执行库的模式

以下模式有文献依据，但当前数据或“共享单锚点”匹配器不足以可靠执行：

- **三跳以上循环与跨账户链式分层**：需要有界路径或时间有序环算法；
- **跨境、高风险地域中转**：需要账户和交易的国家/地区字段以及按日期版本化的地域名单；
- **沉睡账户突然活跃**：需要账户历史基线、开户日期和正常行为画像；
- **多网点、多 ATM 现金拆分**：需要渠道、设备和地理位置字段；
- **个人账户与企业账户异常混用**：需要客户类型、账户用途和业务关系字段。

这些模式应放入 V2，并分别由“有界时间路径/环检测算法”和“历史基线异常算法”支持，避免为了增加模式数量而让 LLM 猜测缺失字段。

## 6. 模式置信度

自然语言经 LLM 编译后的“一致性”只表示生成过程稳定，不表示风险模式在业务上准确。因此这里保留三种不同概念：

- **来源可靠性**：规则是否有可追溯文献；
- **编译一致性**：LLM 多次生成是否得到等价 DSL；
- **检测置信度**：在独立标注验证集上得到的经验精确率，经平滑后写入 `calibrated_confidence`。

当前七条模式均为 `uncalibrated`。在没有标注集前，生成的风险事件 `confidence` 必须为空，不能用人工设置的 0.8 或 LLM 自报概率代替。

## 7. 实验建议

第一轮先做三组实验：

1. **规则正确性**：为每条模式构造正例、相似负例和边界样例；
2. **阈值敏感性**：时间窗、交易对手数量和金额比例分别做网格搜索，只在验证集选择参数；
3. **消融实验**：去掉时间约束、金额约束、角色重合约束，比较 Precision、Recall、F1 与误报类型。

公开数据上优先验证图结构可检测性；银行数据上再验证业务有效性。IBM AML 数据适合作为结构算法基准，但其中合成标签不等于真实银行分布，论文中应单独列为外部有效性限制。

## 8. 参考来源

- FinCEN, *FIN-2014-A005: Funnel Accounts and Trade-Based Money Laundering*: https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2014-a005
- FinCEN, *FIN-2014-A008*: https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2014-a008
- AUSTRAC, *Indicators of suspicious activity for the banking sector*: https://www.austrac.gov.au/industry-and-business/education-and-resources/publications-and-resources/indicators-suspicious-activity-banking-sector
- FATF and Egmont Group, *Trade-Based Money Laundering: Risk Indicators*: https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Trade-Based-Money-Laundering-Risk-Indicators.pdf
- Blanuša et al., *Graph Feature Preprocessor*: https://arxiv.org/abs/2402.08593
- Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*: https://research.ibm.com/publications/realistic-synthetic-financial-transactions-for-anti-money-laundering-models

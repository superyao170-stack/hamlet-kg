"""Build the bounded Data Analytics report artifact from reviewed outputs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "amlsim_experiment_report_artifact.json"


def source(source_id: str, label: str, path: str, description: str, code: str,
           filters: list[str], definitions: list[str]) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": description,
            "sql": code,
            "tables_used": [path],
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def load_metrics() -> list[dict]:
    path = ROOT / "outputs" / "amlsim-baselines" / "metrics.csv"
    rows = []
    dataset_names = {
        "20K_cycle200": "Cycle",
        "20K_fanin200": "Fan-in",
        "20K_fanin200cycle200": "Combined",
    }
    algorithm_names = {
        "degree_iqr": "Degree IQR",
        "repeated_edge_time": "Repeated edge + time",
        "adaptive_low_value": "Adaptive low value",
        "low_value_fan_in": "Low-value fan-in",
        "low_value_cycle": "Low-value cycle",
        "pattern_aware_union": "Pattern-aware union",
    }
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "dataset": dataset_names[row["dataset"]],
                "algorithm": algorithm_names[row["algorithm"]],
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "predicted_nodes": int(row["predicted_nodes"]),
                "true_positive": int(row["true_positive"]),
                "false_positive": int(row["false_positive"]),
                "false_negative": int(row["false_negative"]),
            })
    return rows


def build() -> dict:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    sources = [
        source(
            "src_metrics", "AMLSim baseline metrics",
            "outputs/amlsim-baselines/metrics.csv",
            "读取确定性基准命令生成并经核对的节点级指标。",
            "SELECT * FROM read_csv_auto('outputs/amlsim-baselines/metrics.csv')",
            ["三个 AMLSim 20K 配置", "所有标签只在预测完成后加载"],
            ["F1 = 2PR/(P+R)", "Precision 与 Recall 的总体是 AMLSim 节点"],
        ),
        source(
            "src_quality", "AMLSim data-quality profile",
            "outputs/amlsim-baselines/data_quality.csv",
            "读取数据解释前生成并经核对的数据质量结果。",
            "SELECT * FROM read_csv_auto('outputs/amlsim-baselines/data_quality.csv')",
            ["每个解压样本的全部节点与交易"],
            ["欺诈节点率 = isFraud 节点数 / 全部节点数", "缺失风险步统计 isFraud=1 且 fraudStep=-1 的节点"],
        ),
        source(
            "src_reference", "Reference algorithm benchmark",
            "outputs/reference-algorithms/results.json",
            "读取图匹配与最优传输参考算法的确定性运行结果。",
            "SELECT * FROM read_json_auto('outputs/reference-algorithms/results.json')",
            ["VF2 效率测试使用 fan-in 样本前 5,000 条交易"],
            ["运行时间为本机单次墙钟时间"],
        ),
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "AMLSim 原子事件—风险事件图：首轮基线实验报告",
        "description": "面向专业硕士论文的公开数据质量、算法基线与下一步改造建议。",
        "generatedAt": generated_at,
        "sources": sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# AMLSim 原子事件—风险事件图：首轮基线实验报告"},
            {"id": "summary", "type": "markdown", "body": "## 执行摘要\n\n今天的思路需要对应修改代码，但无需推倒重写。原子事件图、风险事件物化和风险事件级案例检索应保留；主算法转为有界时间窗模式匹配、模式索引剪枝和风险事件图对齐。LLM 只作为候选规则编译器，正式风险事件仍由可复现算法生成。\n\n首轮公开数据实验已经跑通。简单金额与重复边基线获得较高节点级 F1，但这是 AMLSim 合成痕迹带来的结果，不能作为论文创新。语义模式在 fan-in 上有效、在 cycle 上召回较弱，恰好给后续算法改进留下清晰空间。"},
            {"id": "data", "type": "markdown", "sourceId": "src_quality", "body": "## 数据与评价边界\n\n使用 IBM AMLSim 官方 20K cycle、fan-in 和 combined 三个样本。三组数据都通过孤立账户引用、非正金额与主键完整性检查；但存在自环、完全重复交易和不完整的 fraudStep。预测过程不读取 isFraud 或 fraudStep，标签只在预测完成后计算节点级指标。\n\n当前指标是节点级代理任务，不等同于银行案件级告警效果。数据中低金额异常与普通交易之间存在明显断层，因此 1.0 的精确率反映了生成器可分性，而非真实部署能力。"},
            {"id": "result_intro", "type": "markdown", "sourceId": "src_metrics", "body": "## 基线结果\n\n下图比较六类算法在三组数据上的节点级 F1。自适应低金额规则在三组数据上均最高；模式联合基线在 fan-in 上达到 0.755，在 cycle 上仅为 0.205。论文主结果应重点比较模式算法的召回、效率与鲁棒性，而不是追逐金额规则的虚高分数。"},
            {"id": "f1_chart_block", "type": "chart", "chartId": "f1_chart"},
            {"id": "metric_table_block", "type": "table", "tableId": "metric_table"},
            {"id": "efficiency", "type": "markdown", "sourceId": "src_reference", "body": "## 效率参考\n\n在前 5,000 条 fan-in 交易上，通用 VF2 枚举两源汇聚模式耗时 23.750878 秒，面向模式的索引计数耗时 0.001031 秒。VF2 会计算源节点排列，两者计数口径并不相同，因此本轮只把它作为效率方向的证据。"},
            {"id": "recommendations", "type": "markdown", "body": "## 论文算法路线\n\n### A1 有界时间窗风险事件匹配\n实现滑动窗口、共享前缀、金额守恒和重叠事件抑制，与朴素扫描和 VF2 比较事件级 F1、时间与内存。\n\n### A2 模式索引与候选剪枝\n用动作、中心角色、度数下界、时间跨度和金额范围先生成候选，再精确验证；报告候选缩减率与加速比。\n\n### A3 原子组成增强的风险事件图对齐\n在风险事件层计算语义、角色、时间和金额组成代价，比较 Hungarian、Sinkhorn、FGW 与分层模型的 Recall@K、MRR 和 NDCG@K。\n\nLLM 仅编译候选 DSL，并单独评价编译合法率和语义保持率。"},
            {"id": "limits", "type": "markdown", "body": "## 限制与下一步\n\n下一轮需要构造事件级 ground truth，加入金额扰动、重复交易去除、时间窗敏感性和跨生成配置测试；随后再接入第二个公开数据集或小规模人工标注集。当前版本是可运行基线和论文脚手架，不应表述为已经达到 Knowledge-Based Systems 的完整成果。"},
        ],
        "charts": [{
            "id": "f1_chart",
            "title": "AMLSim 节点级 F1 基线比较",
            "subtitle": "金额与重复边线索占优；cycle 模式召回仍弱。",
            "type": "bar",
            "dataset": "baseline_metrics",
            "encodings": {
                "x": {"field": "dataset", "type": "nominal", "title": "数据集"},
                "y": {"field": "f1", "type": "quantitative", "title": "F1"},
                "color": {"field": "algorithm", "type": "nominal", "title": "算法"},
            },
            "options": {"grouping": "grouped", "orientation": "vertical", "legend": True},
            "source": sources[0],
        }],
        "tables": [{
            "id": "metric_table",
            "title": "完整基线指标",
            "dataset": "baseline_metrics",
            "columns": [
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "algorithm", "label": "算法", "type": "text"},
                {"field": "precision", "label": "Precision", "type": "number"},
                {"field": "recall", "label": "Recall", "type": "number"},
                {"field": "f1", "label": "F1", "type": "number"},
                {"field": "balanced_accuracy", "label": "Balanced accuracy", "type": "number"},
            ],
            "defaultSort": {"field": "f1", "direction": "desc"},
            "source": sources[0],
        }],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {"baseline_metrics": load_metrics()},
        },
        "sources": sources,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

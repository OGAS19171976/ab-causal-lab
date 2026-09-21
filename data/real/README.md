# 真实数据入口：契约与「门」

这个目录**现在是空的**，而它空着本身是一件被检查的事：
`src/ablab/validation/unimplemented.py` 里有一条机检项

```
id="real_traffic"  kind="file_absent"  target="data/real/provenance.json"
```

意思是：**只要 `data/real/provenance.json` 不存在，README 里那句
「数据里没有真实流量」就仍然成立**；一旦有人把真实数据接进来，
`scripts/check_unimplemented.py` 会在检查集里直接红，逼着 README 改口径。
这是把一句"无法核对的话"变成**带证据的记录**——与"没做"清单里其他条同一个机制。

## 怎么把真实数据接进来

1. 把三张表放到这个目录（Parquet，列名见下），以及一份 `provenance.json`；
2. 跑 `python scripts/check_real_traffic.py` —— 它会校验契约（含**反冒充**），
   然后真的调用 `warehouse.ingest.load_real_traffic` +
   `warehouse.build.build_warehouse(generate=False)`，把**同一套 SQL** 在外部数据上跑一遍；
3. 把 `reports/` 里的读数与这份 provenance 一起提交，并把 README 那句改掉。

## 三张表的列（与合成器完全相同的落地布局）

| 表 | 必需列 | 说明 |
|---|---|---|
| `exposure_log` | `ds`, `ts`, `user_id`, `experiment`, `variant` | 曝光（分流）日志；`variant` 必须落在声明里 |
| `event_log` | `ds`, `user_id`, `event_name`, `metric_value` | 指标事件；`metric_value` 必须能转成数 |
| `user_profile` | `user_id`, **`reg_ds`**（可选 `city`） | 用户画像；`city` 缺失时簇级路径不可用 |

列名与 `src/ablab/warehouse/generate.py` 的产物**完全一致** —— 所以下游 SQL 一行都不用改，
这正是"换数据源不改链路"的可执行版本。

> **这份契约被实测修正过一次**：第一版把 `reg_ds` 写成"可选"，
> 依据是 `_normalize_profile` 里 `if "reg_ds" in out.columns` 的写法。
> 端到端测试（跑真实链路）当场给出 `KeyError: "['reg_ds'] not in index"` ——
> **归一化器认为可选、链路认为必需**。契约要从**链路**写，不是从某一段代码写。
> 现在 `scripts/check_real_traffic.py` 会先查必需列，缺列时报清楚哪张表缺哪列，
> 而不是把错误留到 SQL 深处。

## `provenance.json` 契约

```json
{
  "source": "某业务线的曝光/事件导出（2026-09-01 ~ 2026-09-07）",
  "exported_at": "2026-09-08",
  "external_generator": true,
  "notes": "由外部系统导出；不含本仓库合成器的任何产物",
  "experiments": [
    {
      "name": "exp_checkout_v2",
      "variants": {"control": 0.5, "treatment": 0.5},
      "control": "control",
      "treatment": "treatment",
      "guardrails": [["latency_p99", "lower_is_better", 0.05]]
    }
  ],
  "tables": [
    {"name": "exposure_log", "file": "exposure_log.parquet", "rows": 12345,
     "sha256": "…64 位十六进制…"},
    {"name": "event_log", "file": "event_log.parquet", "rows": 98765, "sha256": "…"},
    {"name": "user_profile", "file": "user_profile.parquet", "rows": 4321, "sha256": "…"}
  ]
}
```

**为什么声明要跟着数据走**：设计权重、护栏方向与容忍度、哪个变体是对照 ——
这些**不能从观测数据反推**（`ExternalExperiment` 在构造时就会检查权重和为 1）。
把它们写进 provenance，等于把"实验口径"变成接入的一部分。

## 门查什么（`scripts/check_real_traffic.py`）

1. **schema**：上面这些键都在、类型对、权重和为 1；
2. **字节与声明一致**：每个表的 `sha256` 与磁盘上的文件一致、`rows` 一致
   （声明与数据对不上，说明中间被人改过）；
3. **反冒充**（这一条是重点）：`external_generator` 必须为 `true`，
   且源目录里**不许**出现本仓库合成器的痕迹（`.generated` 标记、
   `config_fingerprint` 列）。否则报错：
   **"合成数据不能冒充外部数据"** —— 这是"接入过外部数据"这句话唯一可核对的形式；
4. **真接入**：契约通过后跑 `load_real_traffic` + `build_warehouse(generate=False)`，
   把行数、日期范围、实验清单打出来；这一步失败同样报红。

## 它买不到什么（写在最前面，免得被读成更多）

即使接入了真实数据，也**不等于**"在生产流量上验证过"：公开数据集或某个业务线的
导出都没有埋点丢失、跨端身份合并、实验互斥层配错这些**只有生产流量才有**的问题。
所以 README 的口径只能是"**接入过**外部数据"，而且差异审计（哪些数字变了、
哪些没变）才是这一步真正买到的东西。

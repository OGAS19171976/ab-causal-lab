# `data/real_nsw/` —— LaLonde NSW 真实**处置**数据

与隔壁 `data/real/`（MovieLens）的**分工**：

| | `data/real/`（MovieLens） | `data/real_nsw/`（本目录） |
|---|---|---|
| 处置来自哪 | **我们自己**做的奇偶切分（A/A） | 1970 年代 NSW 项目的**真实随机化** |
| 能验证什么 | 口径、方差、SRM 校准 | **效应**（有公开基准可比） |
| 已知真值 | 没有（真效应就是 0） | Dehejia–Wahba 子样本 **ATT = +1794.34**（SE 671.0） |

## 三张表（契约与 `data/real/` 完全一致）

| 表 | 行数 | 说明 |
|---|---|---|
| `exposure_log` | 445 | 每个单元一行；`ds = 1976-01-01`（项目开始），`variant` 是**真实分组**（185 处置 / 260 对照） |
| `event_log` | 1,335 | 每人三行：re74（1974-01-01）、re75（1975-01-01）落前置；re78（1978-01-01）落后置；事件名 `earnings` |
| `user_profile` | 445 | 契约列 + **协变量**（age / education / black / hispanic / married / nodegree） |

**注意**：`user_profile` 的 `reg_ds` 是 1970-01-01 的**占位值** —— 这份数据里没有
"注册日"这个概念，契约要求这一列。协变量目前**不会进数仓**：归一化只保留契约列
（接入报告里会印出"丢掉的多余列"）。下一轮做观察性对照时要么扩契约、要么直接读源表。

## 出处与许可

* 源：[Rajeev Dehejia 在 NBER 的公开页面](https://users.nber.org/~rdehejia/nswdata2.html)，
  文件 `nswre74_treated.txt`（185 行）与 `nswre74_control.txt`（260 行）；
* 许可：**CC BY-NC 2.0**（署名、非商业）；
* 引用（源站要求的三篇）：Dehejia & Wahba (1999) JASA；Dehejia & Wahba (2002) REStat；
  LaLonde (1986) AER；
* 上游文件的 sha256 记在 `provenance.json` 的 `upstream_files` 里 —— 源站是**可变 URL**，
  而结论挂在具体某一份字节上。

## 复现

    python scripts/build_nsw_dataset.py     # 下载（带重试）+ 校验基准 + 落盘
    python scripts/run_nsw_validation.py    # 走完整条链路 + 写 reports/nsw_lalonde.md

`build_nsw_dataset.py` 会在落盘前**先复现发表基准**（两臂 re78 均值差 = +1794.34 ± 0.05）；
对不上就直接停 —— 那通常意味着列序或子样本选错了，不是"差一点"。

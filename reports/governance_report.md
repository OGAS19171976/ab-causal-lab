# 治理验证报告（操作审计 + 护栏指标）

```text
==========================================================================
ab-causal-lab · 治理验证：操作审计（append-only）+ 护栏指标显式化
==========================================================================

### 1. 操作审计：谁改了什么，留痕
  创建实验 gov_demo，并依次改状态 / 改判定口径 / 绑数仓
  （实验 id 是 uuid4，每次入库都变 —— 刻意不打印它，否则这份报告每次都不一样；审计本身按 seq 排序，不依赖 id。）

   seq  action           field                  before -> after
     1  create
     2  set_status       status                 draft -> running
     3  set_estimator    estimator              cuped -> post_only
     4  bind_warehouse   warehouse_experiment   （空） -> exp_rank_v2

  改判定口径那条的备注（它改变的是**判定规则**，不只是元数据）：
    判定口径变更：历史结论的判定规则会随之改变

### 2. 删掉实验之后，审计必须还在
  实验本身已删除：True
  该实验的审计仍有 5 条，最后一条是 'delete'：实验已删除；这条审计保留（append-only，无级联）
  —— 也就是说「谁删的、删之前是什么状态」仍然查得到；
     审计表**没有外键级联**，这是有意的。

### 3. append-only：不是「我们不写 UPDATE」，是**写不动**
  UPDATE  被 SQLite 触发器拒绝：experiment_events 是 append-only：不允许 UPDATE
  DELETE  被 SQLite 触发器拒绝：experiment_events 是 append-only：不允许 DELETE
  试完之后审计条数仍然是 5（没有被改掉）

### 4. 落盘重开：审计跟着数据库走
  重新打开 registry.db，同一个实验的审计 5 条，动作序列 ['create', 'set_status', 'set_estimator', 'bind_warehouse', 'delete']

### 5. 失败的写入**不能**留下审计
  非法口径被拒：estimator 必须是 ('cuped', 'post_only') 之一，收到 'not_an_estimator'
  不存在的实验被拒：找不到实验 '不存在的实验'
  审计条数 5 -> 5（没有变化，说明拒绝了就没记）
  —— 否则审计会记下「没发生的事」，那比不记更糟。

### 6. 接口层：只读查询，且删除后仍可查
  GET /api/experiments/{id}/events -> 2 条：['create', 'set_status']
  DELETE 之后再查同一个接口 -> HTTP 200，3 条：['create', 'set_status', 'delete']
  GET /api/events?limit=3 -> 最近三次操作：['delete', 'set_status', 'create']（同一个实验的 id 未打印，见上）

### 7. 护栏指标：把「没分析」这条静默变成显式
  该实验声明了 3 个护栏：latency_p99、crash_rate、revenue_per_user
  分析报告的检查项（7 项）：['SRM', '协变量平衡', 'CUPED 收益', '效应分解', '序贯监控', '功效 / MDE', '护栏指标']
  护栏那条：status=info，health 不受影响（当前 health=pass）
  正文：已声明 3 个护栏：latency_p99、crash_rate、revenue_per_user；**本平台尚不分析护栏指标** —— 数据模型只有主指标一条时间序列，护栏需要在数仓里另建指标表。也就是说：这批护栏**目前没有任何东西在看着**，主结论显著不代表可以上线。

  为什么是 info 而不是 warn：护栏未接入是**平台级**缺口，
  声明了护栏的每个实验都会一直 warn —— 而「一条永远亮的告警等于没有告警」，
  health 会因此失去意义（这条教训来自第 31 条）。
  信息要显式（这条检查永远在报告里），但不占用「这次运行有问题」这个信号。

  对照（没声明护栏）：检查项里有没有护栏那条 = False（应为 False）

### 8. 结论
  * 审计是 append-only 的**机械**保证：SQLite 触发器拒绝 UPDATE/DELETE，
    而且它是被当场试出来的，不是一句声称。
  * 审计与业务变更在**同一个事务**里；失败的写入不留痕。
  * 删除实验不会删除审计 —— 那正是最需要它的时刻。
  * 护栏的「未分析」状态出现在每一份相关报告里，并说清了原因。
  * 仍未做的（写在这里而不是留着让人误会）：
    - 审计没有「操作者」字段。平台上还没有鉴权，写上去也只是个空字段；
      真上线要先有身份，再谈「谁做的」。
    - 护栏**仍然没有被分析**：数据模型只有主指标一条时间序列。
      要做需要数仓里另建指标表 + 停实验的判据，那是另一件事。
```

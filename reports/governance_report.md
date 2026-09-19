# 治理验证报告（操作审计 + 护栏指标）

```text
==========================================================================
ab-causal-lab · 治理验证：操作审计（append-only）+ 护栏指标显式化
==========================================================================

### 1. 操作审计：谁改了什么，留痕
  创建实验 gov_demo，并依次改状态 / 改判定口径 / 绑数仓
  （实验 id 是 uuid4，每次入库都变 —— 刻意不打印它，否则这份报告每次都不一样；审计本身按 seq 排序，不依赖 id。）

   seq  操作者            action           field                  before -> after
     1  gov-validator  create
     2  gov-validator  set_status       status                 draft -> running
     3  gov-validator  set_estimator    estimator              cuped -> post_only
     4  gov-validator  bind_warehouse   warehouse_experiment   （空） -> exp_rank_v2

  改判定口径那条的备注（它改变的是**判定规则**，不只是元数据）：
    判定口径变更：历史结论的判定规则会随之改变；v2 -> v3

### 1.5 身份：操作者**只能**来自凭据
  审计里的 `actor` 一列现在是必填的（忘了记谁会在调用点报错），
  而它的值**只**由服务端从凭据推导。三种伪造尝试实测：

  用户          角色       库里存的是         明文 token 可读?
  gov_admin   admin    sha256 前 12 位 否（只存哈希）
  gov_editor  editor   sha256 前 12 位 否（只存哈希）
  gov_viewer  viewer   sha256 前 12 位 否（只存哈希）

  认证结果（真值来自凭据，伪造一律无效）：
    正确 token            -> 'gov_admin'
    错误 token            -> None
    空凭据                -> None
    被停用的用户          -> None（停用立即生效）

  **请求里写的名字不算数** —— 这一条走真实 HTTP 接口实测：
    无凭据 POST                      -> 401
    错 token POST                    -> 401
    viewer POST                      -> 403（角色不够）
    伪造（query+header 都写 http_admin）-> 审计记为 ['http_alice']（凭据是 http_alice）
    editor DELETE                    -> 403（删是 admin 的权限）
    admin DELETE                     -> 204
    被拒的三次尝试留下审计条数        -> 0（只记成功的那两条）
    另外 `actor` 也不是请求体字段（`_Strict` 直接 422），见 `tests/test_governance.py::TestAuthAndActor`：
    · 有一条测试**自动枚举所有写路由**逐个断言 401，
      所以「新加了端点忘了鉴权」会在 CI 上直接红。

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

### 7.5 并发：丢失更新（后写覆盖），以及乐观锁怎么挡住它
  场景：两个客户端（**两个独立连接**，不是同一个对象）都读到同一版本，
  然后都要改状态 —— 这就是「两个人同时改」的最小复现。

  实验建好，version = 1；两个客户端各自读到 v1 与 v1

  A) 不带版本号（默认语义 = 后写覆盖）：
     alice 写 running -> 成功；bob 写 stopped -> 成功
     最终状态 = stopped（bob 覆盖了 alice），version = 3
     审计里两条都在：['alice', 'bob'] —— 但**alice 的意图已经不在结果里了**，而谁都没收到错误。
     这就是丢失更新：比崩溃难查，因为一切「看起来都成功了」。

  B) 带上各自读到的版本号（乐观锁）：
     alice 写 -> 成功；bob 写 -> **被拒**（RegistryConflict）
     拒绝理由：版本冲突：你读到的是 v4，当前已经是 v5 —— 说明这中间有人改过。请重新读取后再提交（这次写没有生效）。
     最终状态 = running（alice 的改动还在），version = 5
     也就是说：**冲突被变成了一次可见的失败**，而不是一次静默的覆盖。

  边界（写在明处）：
    · 乐观锁是**可选**的 —— 不带 If-Match 就退回后写覆盖，
      这是刻意的默认（单机平台上大多数调用就是这么用的）；
    · 没有自动重试与自动合并：拿到 412 之后要**重新读、重新决定**，
      因为「该不该改」取决于中间那次改动是什么；
    · sqlite 单文件本身是串行写的，这里的「并发」是应用层的
      读-改-写交错，不是数据库层的写冲突。

### 8. 结论
  * 审计是 append-only 的**机械**保证：SQLite 触发器拒绝 UPDATE/DELETE，
    而且它是被当场试出来的，不是一句声称。
  * 审计与业务变更在**同一个事务**里；失败的写入不留痕。
  * 删除实验不会删除审计 —— 那正是最需要它的时刻。
  * 护栏的「未分析」状态出现在每一份相关报告里，并说清了原因。
  * 仍未做的（写在这里而不是留着让人误会）：
    - ~~审计没有「操作者」字段~~ **已补**：静态 token 鉴权 + `actor` 列，
      见第 1.5 节与 README 设计决策第 45 条。
      **边界**：静态 token 无过期、无轮换、无限速，token 泄露即冒充；
      读接口仍然匿名；迁移前的老记录操作者是「（迁移前未知）」。
    - ~~注册表没有并发控制~~ **已补**：`version` 列 + `If-Match` 头，
      冲突返回 412 而不是静默覆盖（见第 7.5 节）。**边界**：乐观锁是
      可选的（不带 If-Match 仍是后写覆盖），且没有自动重试与合并。
    - 护栏**仍然没有被分析**：数据模型只有主指标一条时间序列。
      要做需要数仓里另建指标表 + 停实验的判据，那是另一件事。
```

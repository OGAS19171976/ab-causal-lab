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

### 7. 护栏指标：**真的判定**，并给出「要不要停实验」的判据
  三种声明各跑一遍，因为它们的**补救办法完全不同**：
    A) 有规格、有数据（其中一条被注入了 +12% 的真实伤害）
    B) 只声明了名字，没给方向与容忍度
    C) 规格齐全、但没有数据（数仓路径还没有护栏表）

  A) 护栏那条 status = fail，health = fail
     护栏分析（3 条声明，3 条可判定，名义 alpha=0.05）
       latency_p99        fail    伤害 +0.1179（相对 +11.78%，容忍 5.00%）  CI[+0.1152, +0.1206]
       crash_rate         pass    伤害 +0.0011（相对 +0.11%，容忍 10.00%）  CI[-0.0016, +0.0038]
       revenue_per_user   pass    伤害 +0.0009（相对 +0.09%，容忍 5.00%）  CI[-0.0018, +0.0035]
       判定：stop —— **建议停止实验**：latency_p99 的伤害已越过事先声明的容忍度  **这是停实验的理由**：护栏触发不是参考信息。注意判定用的是伤害的**置信下界**超过事先声明的容忍度，而且多条护栏做了 Bonferroni 校正（alpha/K）—— 宁可少报几条，也不要因为多看几个指标就误停一次实验。

  B) 只有名字：status = warn（**用户这一次就能补**：补 direction + max_harm）
     recall_coverage    unknown  没有声明方向或容忍度（`direction` + `max_harm` 必须显式给出，不从指标名猜）—— 无法判断，**不等于通过**

  C) 有规格没数据：status = info（**平台要补**：数仓还没有护栏表）
     latency_p99        unknown  没有该护栏的数据 —— 无法判断，**不等于通过**

  判定规则（写在代码里，也写在 README 里）：
    · 把两臂之差按 direction 折算成**伤害**；越界看的是伤害的**置信下界**，
      不是点估计 —— 点估计超了但证据不足记 warn（继续观察），
      这样「停机」这个动作才是保守的；
    · K 条护栏用 Bonferroni（alpha/K）校正：要控的是「误判有害从而错误停机」；
    · **没声明方向与容忍度就不判断**（判 unknown）—— 从指标名猜方向
      会把伤害静默读成改善；
    · **缺数据判 unknown，绝不判 pass**：那正是这一块原来的毛病。

  A 组的细节值得看：latency_p99 被注入 +12% 伤害（远超 5% 容忍度），
  于是 health 直接变成 fail，报告里出现「建议停止实验」——
  Kohavi 那本书里护栏触发是**停实验的理由**，不是参考信息。

  对照（没声明护栏）：检查项里有没有护栏那条 = False（应为 False）

### 7.7 护栏判定的**校准**：这条停机规则到底有多保守
  判定规则问的不是「有没有伤害」，而是「伤害**是否超过容忍度**」，
  所以必须量出它的运行特征（误停率 / 功效），而不是只说规则合理：

  护栏判定校准（80 次/场景，n=3000，容忍度 5.00%，名义 alpha=0.05）
    H0（真实伤害 0）：误停率 **0.0000**、watch 0.0000
      判定分布：{'stop': 0, 'watch': 0, 'ok': 80, 'unknown': 0}
    H1（注入伤害 12.00%）：停实验的比例（功效）**1.0000**
      判定分布：{'stop': 80, 'watch': 0, 'ok': 0, 'unknown': 0}
    边界点（伤害正好 = 容忍度 5.00%）：{'stop': 4, 'watch': 36, 'ok': 40, 'unknown': 0}
      读法：规则问的是「伤害**是否超过容忍度**」，所以边界点上应当大约一半喊停（估计以真值 5% 为中心、左右各一半）——
      实测偏保守（点估计要越界、且置信下界也要越界才停），这是有意的。
    读法：判定阈值是**容忍度**而不是 0，所以 H0 下误停率必然**远低于** alpha ——
    这是「宁可少停」的取舍；代价是功效要靠更大的伤害或更多样本来补。

  三个工作点合起来说明：**阈值在容忍度上而不是 0 上**，
  于是 H0 下几乎不会误停，代价是伤害刚好压在容忍度附近时
  大量落在「观察」带 —— 那正是「宁可少停、也不要误停」的取舍。

### 7.9 决策层：护栏触发**真的能停实验**（而且服务端自己复核）
  前面几节做到的是「报告里写着建议停止实验」。从一句话到一个动作之间
  隔着一次判断，所以这个端点**不信任调用方递过来的结论**：
  它自己重跑一遍分析，确认护栏确实越界才执行。

  三种情形各跑一遍：
    A) 护栏越界 -> HTTP 200，状态 stopped，tripped=True
    B) 护栏没越界 -> HTTP 409（拒绝，实验状态不变）
    C) 人工强制：editor -> HTTP 403；admin -> HTTP 200（forced=True）

  审计里留下的依据（**动作名单独记为 stop**，理由进 note）：
    stop_editor | 护栏停止：stop —— **建议停止实验**：latency_p99 的伤害已越过事先声明的容忍度（服务端重新分析确认）；v1 -> v2

  为什么值得单独做一层：停实验是这套平台里**最不可逆**的动作
  （分流随时能重开，已经造成的伤害收不回来）。所以它比「改个状态」厚：
  服务端复核、理由必填、人工叫停要 admin、成功与拒绝都留痕。

### 7.11 「没做」的清单：从「靠人偶然发现」到「检查集里直接红」
  这个仓库栽过三次同一类跟头：**功能做完了，README 还写着没做** ——
  簇级 CUPED（被三处代码拒绝了两轮）、M2 决策层、数仓比值链路。
  共同点是「没做」是一句**无法被核对**的话：数字有人对（声明清单），
  「没做」没人对，于是它只朝一个方向漂移。
  这一轮把每一句「没做」变成一条**带证据**的记录：证据必须现在还成立
  （某个符号确实不存在 / 某个串搜不到 / 某个文件不存在），
  一旦不成立就在检查集里报错并指出该改哪一句。


  机检 0 条『没做』：0 条仍成立，0 条已经过时。
  另有 2 条**无法机检**、只能人读 ——清单不假装覆盖它们。
  退出码：0（0 = 清单与事实一致）

  第一次跑就报了**一个假阳性**：清单里写着 `target=load_real_traffic`，
  于是「在 src/ 下搜这个串」搜到了**清单自己** —— 检查器也会骗人，
  所以搜索时显式跳过清单文件（代码里写了原因）。
  更早一步，这一轮还顺手抓到**三条已经过时的声明**：
    · M2「没做决策层」（护栏决策层已做，见 7.9）
    · 「整簇路径只支持 post-only」（簇级 CUPED 已做并校准）
    · 「数仓不支持比值指标」（06/07 两条 SQL 早已上线）

### 7.12 环境来源与三方库类型：两句「只能人看」变成两张表
  已知边界里原先有两句话是**无法核对**的：
    · 「三方库没有类型保证：取决于上游是否带 py.typed，只能人看」；
    · （隐含的）「锁文件 == 环境」—— 它比的是**版本**，看不见**来源**。
  这一轮把它们各变成一条机器检查。

  第一句的实测（`scripts/check_typed_deps.py`）：
    三方库类型清单（判据：覆盖 100% 的锁文件条目 + 每个无类型的包都有处置）
      锁文件条目 48 个（另有 0 个因平台 marker 不适用：无）；带 py.typed 的 32 个；本项目实际 import 的 11 个

      包                     版本          py.typed  stub 包          用到它的源码文件
      numpy                 2.5.3       有         -                     70
      scipy                 1.18.1      **没有**    -                     31
      pytest                9.1.1       有         -                     20
      pandas                3.0.3       **没有**    -                     10
      duckdb                1.5.5       有         -                      7
      fastapi               0.141.1     有         -                      7
      scikit-learn          1.9.1       **没有**    -                      4
      packaging             26.3        有         -                      2
      matplotlib            3.11.1      有         -                      1
      pydantic              2.13.5      有         -                      1
      uvicorn               0.53.0      有         -                      1
      cloudpickle           3.1.2       **没有**    -                      0
      colorama              0.4.6       **没有**    -                      0
      fonttools             4.64.0      **没有**    -                      0
      joblib                1.6.0       **没有**    -                      0
      mypy-extensions       1.1.0       **没有**    -                      0
      pyarrow               25.0.1      **没有**    -                      0
      pygments              2.21.0      **没有**    -                      0
      python-dateutil       2.9.0.post0 **没有**    -                      0
      ruff                  0.16.8      **没有**    -                      0
      six                   1.17.0      **没有**    -                      0
      threadpoolctl         3.7.0       **没有**    -                      0
      typing-extensions     4.16.0      **没有**    -                      0
      tzdata                2026.2      **没有**    -                      0

      没带类型、但被本项目用到的（逐包处置）：
        scipy              31 个文件   处置：stub
          用到的是 stats.norm / optimize.minimize / spatial 的几个函数；本地最小 stub 只声明这些签名，上游改名会在 mypy 里报
        pandas             10 个文件   处置：accept-any
          DataFrame 的类型在无 stub 时基本退化成 Any；本仓库对它的用法集中在数仓 IO 与列选择，靠测试与 schema 检查兜底
        scikit-learn        4 个文件   处置：accept-any
          只作为 nuisance 学习器（Ridge / RandomForest）出现，接口窄且被测试覆盖；上游一旦补上顶层 py.typed，本条会被判过时

    清单覆盖锁文件全部条目，且每个无类型的包都有明确处置

  第二句实测出来的是一个**藏了很久的事实**：本地 venv 曾经是混合环境
  （`include-system-site-packages = true`），锁文件 48 个包里有
  **14 个**（pandas / matplotlib / pytest / packaging…）实际解析自
  **系统 Python 的 site-packages**，venv 里根本没有它们 —— 而
  `lock_requirements.py --check` 照样全绿，因为**版本号恰好一样**。
  也就是说「本地跑的东西」与「CI 装的东西」不是同一套文件。
  修法两步：按锁文件把缺的包装进 venv（必须 `--ignore-installed`，
  否则 pip 看到系统里的同名包就认为「已满足」——这一步实测踩过），
  再把 `include-system-site-packages` 置为 false（CI 用裸解释器，
  那一条会显式打印「不适用」而不是静默跳过；判据是"跑测试的这套包
  必须都来自当前解释器的 purelib"）。修完的读数：
    环境来源检查（锁文件管版本，这一条管**来源**）
      解释器: .venv/Scripts/python.exe
      当前解释器的 purelib: .venv/Lib/site-packages（venv）
      开关 include-system-site-packages：适用：.venv/pyvenv.cfg 里读到 false

      一、锁文件里的 48 个发行版装在哪
        当前解释器内 48 个 / 外面 0 个

        另有 0 个因平台 marker 不适用（本平台不该装）：无

      二、源码 import 的顶层模块：共 22 个，没人锁的 0 个
        标准库         14
        锁文件         8

    环境来源与锁文件一致：所有包都来自当前解释器，源码没有未锁的 import

  为什么值得单独记一条：这个漏检**不是**版本错，是**拓扑**错 ——
  版本对、来源错，所有基于版本的自检都会说「没问题」（设计决策 53）。

### 7.13 前端契约：495 行单文件页面，靠静态契约挡住两类「不报错的坏」
  `src/ablab/platform/static/index.html` 是**无构建步骤**的单文件页面
  （一个 `api(path, opts)` helper 包住 fetch）。它最容易出的两类事故
  都**不报错**：后端改了路径、前端静默 404；改了某个 id、控件静默失效。
  这一轮给它配了四条机检（`scripts/check_frontend.py`，秒级、进快速组）：
    1. **路由契约**：页面调用的每条端点都必须在服务端路由表里（含方法）；
    2. **反向契约**：路由表里没出现在页面上的端点，必须有**写下来的决定**；
    3. **选择器契约**：`$(「#x」)` / `querySelector(「.y」)` 指向的 id/class 必须存在；
    4. **语法**：`node --check` 对页面里的 JS 做一次真正的解析。
  路由表走**两条独立路径**取（静态解析装饰器 + 真的构造一次 app 读 routes），
  两边算出来不一样就报错 —— 以后有人改用 include_router，静态那一条会漏。

    前端契约检查（页面：src/ablab/platform/static/index.html，无构建步骤）
      页面 495 行，内联 JS 301 行；应用路由 16 条（静态解析 16 条，两条路径一致）；另有 4 条 FastAPI 自带（文档/OpenAPI，不参与判据）

      一、UI → API：页面调用的端点
        方法     端点                                        行
        GET    /api/experiments                          L75
        POST   /api/experiments/{param}/analyze          L121
        POST   /api/experiments                          L248
        POST   /api/validate/aa                          L267

      二、API → UI：路由表里没有出现在页面上的端点（每条都有决定）
        DELETE /api/experiments/{param}                  有决定
               └ 删除是破坏性操作，只留 admin 的 curl 路径
        GET    /                                         有决定
               └ 静态页面本身（服务端把 index.html 发出来），不是页面要调的 API
        GET    /api/events                               有决定
               └ 同上（全局审计流）
        GET    /api/experiments/{param}                  有决定
               └ 详情页用列表返回的字段直接渲染；这条是给脚本/curl 单取一条用的
        GET    /api/experiments/{param}/events           有决定
               └ 审计留痕：页面没有审计页，运维用 curl 查
        GET    /api/warehouse/experiments                有决定
               └ 数仓里有哪几条实验：属于运维探索，页面不做
        GET    /healthz                                  有决定
               └ 探活接口，给运维与 CI 用，页面不需要
        PATCH  /api/experiments/{param}/status           有决定
               └ 改实验状态（draft/running/...）：页面只做创建与查看，状态流转留给脚本
        POST   /api/design/power                         有决定
               └ 设计期算功效：设计期用 Python API，页面只管在跑的实验
        POST   /api/experiments/{param}/bind             有决定
               └ 绑定数仓需要选表与确认，交给脚本
        POST   /api/experiments/{param}/estimator        有决定
               └ 改判定口径会**改变结论的解释**，页面刻意不提供入口（只留带 token 的脚本调用）
        POST   /api/experiments/{param}/stop             有决定
               └ 停实验是不可逆动作：宁可不在页面上放一个容易被误点的按钮

      三、选择器：页面里引用 38 处，其中指向不存在的 0 处
      四、语法：node --check 通过

    页面与接口的契约一致：端点都在、决定都齐、选择器都能落地、JS 语法通过

  第一次跑就抓出三件事，都不是页面写错了，而是**没人写下来的事实**：
    · FastAPI **自带** 4 条路由（/docs、/redoc、/openapi.json 与 oauth2-redirect）
      —— 第一版把它们报成「界面没用到且没有决定」4 条假阳性，现在单独归类；
    · 路由表里有 12 条**界面有意不做**的端点，这一轮逐条写下理由
      （删除/停实验/改判定口径这类不可逆或改变解释的动作，页面刻意不给入口）；
    · 我自己的决定表里有一条其实**已经被 UI 用着**（`POST /api/validate/aa`），
      被反向判据当场抓出来 —— 决定表与「没做」清单一样会漂，所以也要有反向检查。
  诚实的边界：这四条检查的是**契约**（路径/方法/选择器/语法），不是**行为**。
  「点了按钮会不会真的做对」靠 HTTP 层测试（`tests/test_platform_*.py`：状态码、
  乐观锁 412、审计留痕）与人工看一眼；Playwright 那类端到端与它们重叠度高，
  **刻意不做** —— 这一条也写在脚本的模块文档里，免得后来人以为漏了。
  契约检查自己也有故障注入测试：拿一份**故意写坏**的页面（调用不存在的端点、
  方法写错、选择器指向不存在、语法错）跑一遍，断言它**确实报红**。

### 7.14 真实数据：把「没有真实流量」变成一道有证据的门
  已知边界里原先写着「数据里到底有没有真实流量，只能人看」——
  这一轮把它变成 `unimplemented.py` 里一条**机检项**：
  `kind=file_absent`、`target=data/real/provenance.json`。
  也就是说：只要那份 provenance 不存在，README 那句话就成立；
  一旦有人把外部数据接进来，检查集立刻红，并逼着 README 改口径。
  这是把一个**无法核对**的问题换成可核对的证据（与 7.11 节同一个机制）。

  门本身还查三层（`scripts/check_real_traffic.py`）：
    1. **契约**：source / exported_at / external_generator / experiments /
       tables 都在，实验的设计权重和为 1（口径必须来自声明，不能从数据反推）；
    2. **字节与声明一致**：每张表的 sha256 与行数都要和磁盘上的文件对得上；
    3. **反冒充**：`external_generator` 必须为 true，且源目录里不许出现
       本仓库合成器的痕迹（`.generated` 标记、`config_fingerprint` 列）——
       没有这一条，「合成数据 + 手写 provenance」就能把这句话骗过去。
  三层都过了才**真接入**：load_real_traffic 归一化 → build_warehouse
  (generate=False) 跑同一套 SQL。**现在目录里真的有数据**（MovieLens：
  610 用户 / 100,836 条真实评分 / 1996–2018），所以这一步在 CI 里
  每个 push 都会真的把外部数据接进数仓并跑完整条链路 ——
  真实分布上的 A/A 读数：control 315 / treatment 295（期望各 305，
  χ² = 0.6557）。在这之前它只打印'没有接入'并返回 0。

    真实数据的门（契约 + 反冒充 + 真接入）
      目录：data/real

      契约通过：source='MovieLens ml-latest-small（GroupLens Research，外部分发）'，exported_at=2026-09-22，3 张表，1 个实验声明
      真接入完成：归一化 → build/real_traffic；SQL → build/warehouse_real.duckdb
      实验：ml_aa
      注意口径：这只能叫「**接入过**外部数据」，不等于在生产流量上验证过 ——
      差异审计（哪些数字变了、哪些没变）才是这一步真正买到的东西。

  契约被实测修正过一次，值得记：第一版把 `user_profile.reg_ds` 写成「可选」，
  依据是 `_normalize_profile` 里那个 `if reg_ds in out.columns` 的写法。
  端到端测试（拿最小 fixture 真的跑一遍链路）当场给出
  `KeyError: ['reg_ds'] not in index` —— **归一化器认为可选，链路认为必需**。
  所以现在门会先查必需列，缺列时报清楚哪张表缺哪列。
  教训（写进 README 决策 55）：**契约要从链路写，不是从某一段代码写**。

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
    - ~~护栏没有被分析~~ **已补**：合成路径现在**真的判定**护栏，
      越界（伤害的置信下界超过事先声明的容忍度）会让 health 变 fail
      并给出「建议停止实验」（见第 7 节）。**边界**：数仓路径还没有
      护栏表，那里的护栏判 unknown（不是通过）；方向与容忍度必须显式声明。
```

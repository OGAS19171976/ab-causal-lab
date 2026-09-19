"""Honest 因果森林（简化版 GRF）。

两个关键设计
------------
**1. 分裂准则：最大化子节点之间 CATE 的差异**

    分数 = (n_L * n_R) / (n_L + n_R) * (tau_L - tau_R)^2

而不是最小化结果的方差。这是因果树和预测树的根本区别：
后者关心"把 Y 预测准"，前者关心"把效应区分开"。

**2. Honest splitting（诚实分裂）**

把样本劈成两半：一半 ``structure`` 用来**选分裂点**，另一半 ``estimation``
用来**估叶子里的效应**。这样叶子效应的估计与树结构独立，
CATE 估计才是渐近正态的（Wager & Athey 2018）。

不诚实会怎样：树结构本身就是在"找效应差异最大"的地方切，
再用同一批数据估叶子均值，等于**对噪声做最大化**，
叶子效应被系统性放大。这个问题在验证台里会被直接量出来。

实现上分裂用排序 + 累积和做到 ``O(m log m)``：
对每个特征排序后，任意一个切点两侧的 ``tau`` 都能 O(1) 算出来。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CausalTree", "CausalForest", "ForestConfig"]

#: 影响函数合成能承受的 ψ 矩阵单元数上限（``n_pred × N``）。
#: 约 1.6 GB（float64）—— 到这一步就不是"算得慢"，而是"内存没了"，
#: 所以宁可在这里明确报错，也不要让它把机器拖死。
_MAX_IF_CELLS = 2 * 10**8


@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None
    tau: float = 0.0
    #: 叶子 τ̂ 的抽样方差（在 estimation 半样本上算，见 ``_tau_of``）。
    #: 它让森林能给出**区间**而不只是点估计 —— 这是 M4 从"排序可用"走到
    #: "水平可用"的那一步。
    var: float = 0.0
    n_struct: int = 0
    n_est: int = 0
    #: 该叶子的**影响函数原料**（在 estimation 半样本上算）：
    #: 单元下标、是否处置、以及臂内残差 ``Y_i − 该臂均值``。
    #:
    #: 为什么要在叶子上留这三样：跨树方差的正确做法是把各棵树的**影响函数
    #: 相加**（协方差自动进来），而不是把方差按独立合成。
    #: 这与本仓库在 CS 聚合、SA 聚合、事件研究上修过三次的是同一个错误 ——
    #: 森林是第四处，也是最后一处还活着的地方。
    est_idx: np.ndarray | None = None
    est_treated: np.ndarray | None = None
    est_resid: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0

    #: 「非叶节点一定有左右子节点」这条不变式，收在这里说一次。
    #:
    #: 字段本身是 ``_Node | None``（叶节点没有子节点），而下面所有递归函数都在
    #: ``not is_leaf`` 之后直接用 ``node.left`` —— 类型检查器当然看不到这条推理，
    #: 于是 12 处报错。与其在 12 个地方写 ``assert node.left is not None``
    #: （``-O`` 下会被删掉，报错也没有上下文），不如把不变式讲一次：
    #: 真缺了子节点说明**树的构造坏了**，那时需要的是一句能读懂的话。
    @property
    def children(self) -> tuple["_Node", "_Node"]:
        """``(左, 右)``；非叶节点一定有。"""
        if self.left is None or self.right is None:
            raise ValueError(
                f"内部节点缺少子节点（feature={self.feature}, threshold={self.threshold}）——"
                "树构造有问题，不该走到这里"
            )
        return self.left, self.right


@dataclass(frozen=True)
class ForestConfig:
    """因果森林的超参数。"""

    n_trees: int = 100
    max_depth: int = 5
    min_leaf: int = 20
    #: 每个节点随机考虑的特征数（None = 用 sqrt(p)）
    n_features_per_split: int | None = None
    #: 每棵树的子采样比例
    subsample: float = 0.7
    #: 候选切点的分位数个数
    n_thresholds: int = 20
    #: 是否诚实分裂。**关掉它就能量出不诚实带来的偏置。**
    honest: bool = True
    #: 叶子效应的收缩强度：``tau = (n*tau_leaf + k*tau_global) / (n + k)``。
    #: ``0`` 表示不收缩。叶子越小、噪声越大时越该收缩 ——
    #: 这也是"要不要用弹性模型"这个问题的连续版本：``k=inf`` 就退化成常数 ATE。
    shrinkage: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_trees < 1:
            raise ValueError("n_trees 必须为正")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample 必须落在 (0, 1]")
        if self.n_thresholds < 2:
            raise ValueError("n_thresholds 至少为 2")


def _tau_of(D: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> tuple[float, float, int, int]:
    """子样本上的 CATE、它的**抽样方差**、以及两组人数。

    方差取两臂均值差的标准公式 ``S1²/n1 + S0²/n0``（组内样本方差，ddof=1）——
    它就是"这个叶子的 τ̂ 有多不确定"的直接估计，不需要额外假设。

    **必须在 estimation 半样本上算**：honest 分裂保证"选叶子的样本"与
    "估效应的样本"不同，所以这个方差不会被"挑到最漂亮的叶子"这件事污染。
    用 training 半样本算会**偏小** —— 同一批数据既选叶子又估效应，
    挑中的叶子天然是效应看起来最大的那个。
    """
    d = D[mask]
    y = Y[mask]
    n1 = int(d.sum())
    n0 = int(d.size - n1)
    if n1 == 0 or n0 == 0:
        return 0.0, float("inf"), n1, n0
    y1 = y[d > 0.5]
    y0 = y[d < 0.5]
    tau = float(y1.mean() - y0.mean())
    # ddof=1：单元素那一臂没有方差可言 → inf，表示"这个叶子给不出区间"
    v1 = float(y1.var(ddof=1)) if n1 > 1 else float("inf")
    v0 = float(y0.var(ddof=1)) if n0 > 1 else float("inf")
    var = v1 / n1 + v0 / n0 if np.isfinite(v1) and np.isfinite(v0) else float("inf")
    return tau, var, n1, n0


def _best_split(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    idx: np.ndarray,
    features: np.ndarray,
    n_thresholds: int,
    min_leaf: int,
) -> tuple[int, float, float] | None:
    """在 ``idx`` 上找最优切点，返回 ``(feature, threshold, score)``。"""
    m = idx.size
    if m < 2 * min_leaf:
        return None

    x_sub = X[idx]
    d_sub = D[idx]
    y_sub = Y[idx]

    best: tuple[int, float, float] | None = None

    for f in features:
        x = x_sub[:, f]
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ds = d_sub[order]
        ys = y_sub[order]

        cum_d = np.cumsum(ds)
        cum_dy = np.cumsum(ds * ys)
        cum_y = np.cumsum(ys)

        total_d = cum_d[-1]
        total_dy = cum_dy[-1]
        total_y = cum_y[-1]

        # 切点：前 k 个在左边（k = 1..m-1）
        k = np.arange(1, m)
        # 必须真的切开（相邻值不同），且两侧都够大
        valid = (xs[k - 1] < xs[k]) & (k >= min_leaf) & (m - k >= min_leaf)
        if not valid.any():
            continue

        kk = k[valid]
        dL = cum_d[kk - 1]
        dyL = cum_dy[kk - 1]
        yL = cum_y[kk - 1]
        nL = kk.astype(float)
        nR = m - nL

        dR = total_d - dL
        dyR = total_dy - dyL
        yR = total_y - yL

        ok = (dL > 0) & (nL - dL > 0) & (dR > 0) & (nR - dR > 0)
        if not ok.any():
            continue

        kk, dL, dyL, yL, nL, nR, dR, dyR, yR = (
            arr[ok] for arr in (kk, dL, dyL, yL, nL, nR, dR, dyR, yR)
        )

        tau_L = dyL / dL - (yL - dyL) / (nL - dL)
        tau_R = dyR / dR - (yR - dyR) / (nR - dR)
        score = (nL * nR) / (nL + nR) * (tau_L - tau_R) ** 2

        j = int(np.argmax(score))
        if best is None or score[j] > best[2]:
            threshold = float((xs[kk[j] - 1] + xs[kk[j]]) / 2.0)
            best = (int(f), threshold, float(score[j]))

    return best


def _build(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    idx: np.ndarray,
    *,
    depth: int,
    config: ForestConfig,
    rng: np.random.Generator,
    n_features: int,
) -> _Node:
    """在 ``idx``（structure 半样本）上递归建树。"""
    node = _Node(n_struct=int(idx.size))

    if depth >= config.max_depth or idx.size < 2 * config.min_leaf:
        return node

    k = config.n_features_per_split or max(1, int(np.sqrt(n_features)))
    features = rng.choice(n_features, size=min(k, n_features), replace=False)

    split = _best_split(X, D, Y, idx, features, config.n_thresholds, config.min_leaf)
    if split is None:
        return node

    f, threshold, _score = split
    left_mask = X[idx, f] <= threshold
    left_idx = idx[left_mask]
    right_idx = idx[~left_mask]
    if left_idx.size < config.min_leaf or right_idx.size < config.min_leaf:
        return node

    node.feature = f
    node.threshold = threshold
    node.left = _build(
        X, D, Y, left_idx, depth=depth + 1, config=config, rng=rng, n_features=n_features
    )
    node.right = _build(
        X, D, Y, right_idx, depth=depth + 1, config=config, rng=rng, n_features=n_features
    )
    return node


def _assign_leaves(node: _Node, X: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
    """把 ``idx`` 里的样本路由到叶子，写入 ``out``（叶子 id）。"""
    if node.is_leaf:
        out[idx] = id(node)
        return
    mask = X[idx, node.feature] <= node.threshold
    left, right = node.children
    if mask.any():
        _assign_leaves(left, X, idx[mask], out)
    if (~mask).any():
        _assign_leaves(right, X, idx[~mask], out)


def _fill_leaf_effects(
    node: _Node,
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    idx: np.ndarray,
    index_map: np.ndarray | None = None,
) -> None:
    """在 estimation 半样本上给叶子填效应值。

    ``index_map`` 把"树内部的下标"翻回**原始训练样本**的下标。森林里每棵树
    只吃到 subsample，叶子里存的 ``est_idx`` 如果停留在本地坐标，各棵树的
    影响函数就**没法相加**（同一个单元在不同树里编号不同）—— 而相加正是这
    一轮要修的事。
    """
    if node.is_leaf:
        tau, var, n1, n0 = _tau_of(D, Y, idx)
        node.tau = tau
        node.var = var
        node.n_est = n1 + n0
        # 影响函数原料：臂内残差（保证按臂中心化，正是 IF 的定义）
        treated = D[idx] > 0.5
        resid = np.array(Y[idx], dtype=float)
        if treated.any():
            resid[treated] -= resid[treated].mean()
        if (~treated).any():
            resid[~treated] -= resid[~treated].mean()
        node.est_idx = np.array(idx, dtype=int) if index_map is None else index_map[idx]
        node.est_treated = treated
        node.est_resid = resid
        return
    mask = X[idx, node.feature] <= node.threshold
    left, right = node.children
    if mask.any():
        _fill_leaf_effects(left, X, D, Y, idx[mask], index_map)
    if (~mask).any():
        _fill_leaf_effects(right, X, D, Y, idx[~mask], index_map)


def _predict_tree(node: _Node, X: np.ndarray, out: np.ndarray, idx: np.ndarray) -> None:
    if node.is_leaf:
        out[idx] = node.tau
        return
    mask = X[idx, node.feature] <= node.threshold
    left, right = node.children
    if mask.any():
        _predict_tree(left, X, out, idx[mask])
    if (~mask).any():
        _predict_tree(right, X, out, idx[~mask])


def _accumulate_tree_if(
    node: _Node, X: np.ndarray, psi: np.ndarray, idx: np.ndarray
) -> None:
    """把**这棵树对预测值的影响函数**累加到 ``psi`` 上。

    叶子 L 上的 τ̂ 就是两臂均值之差，所以单元 i 在 τ̂_L 里的**系数**就是
    它那一臂在叶子里的个数分之一。逐树的贡献写成（在 L 的 estimation
    半样本上按臂中心化）：

        ψ_i = +(Y_i − Ȳ_{t,L}) / n_{t,L}     若 i 在 L 且是处置组
        ψ_i = −(Y_i − Ȳ_{c,L}) / n_{c,L}     若 i 在 L 且是对照组
        ψ_i = 0                               否则（没落进这棵树）

    这就是 GRF 里的森林权重 ``α_i(x) = (1/B)Σ_b α_{b,i}(x)`` 乘上残差，
    方差直接是 ``Σ_i ψ̄_i²`` —— **不除 N**（归一化已经藏在 ``1/n_臂`` 里）。

    **踩过的坑（必须留着）**：第一版写成 ``(n_L/n_臂)·残差``，然后
    ``SE = sqrt(Σψ̄²)/N``。单看一棵树它恰好等于
    ``sqrt(Σψ̄²)/N = SE_真 · n_L/N``，也就是**每多切一层就凭空小一截**
    （实测 16 片叶子 → 单树 SE 偏小约 20 倍）。量出来才发现，改回来。

    于是"预测点在 x 处"的森林估计量是若干 τ̂_L 的平均，它的影响函数就是这些
    ψ 的平均 —— **跨树协方差自动进来**，因为同一个单元会同时出现在多棵树的
    ψ 里（而按独立合成的 ``Σ σ²_b`` 把它丢掉了）。

    **两个下标空间必须分清**（这里踩过一次）：``idx`` 是**预测点**的行号
    （``0..n_pred-1``），``node.est_idx`` 是**训练样本**的行号
    （``0..N-1``，已经翻回原始坐标）。所以这里落笔的是一个
    ``(n_pred, N)`` 矩阵里的**子块**：哪些预测点 × 哪些训练样本。
    """
    if node.is_leaf:
        est_idx = node.est_idx
        if est_idx is None or est_idx.size == 0 or idx.size == 0:
            return
        treated = node.est_treated
        resid = node.est_resid
        assert treated is not None and resid is not None
        n_t = float(treated.sum())
        n_c = float((~treated).sum())
        contrib = np.zeros(est_idx.size)
        if n_t > 0:
            contrib[treated] = resid[treated] / n_t
        if n_c > 0:
            contrib[~treated] = -resid[~treated] / n_c
        # 落进同一个叶子的预测点共享同一根 ψ；不同预测点之间**不会**互相污染，
        # 因为第 0 维的 index 互不相同（`np.ix_` 的笛卡尔积正好是这个意思）。
        psi[np.ix_(idx, est_idx)] += contrib[None, :]
        return
    mask = X[idx, node.feature] <= node.threshold
    left, right = node.children
    if mask.any():
        _accumulate_tree_if(left, X, psi, idx[mask])
    if (~mask).any():
        _accumulate_tree_if(right, X, psi, idx[~mask])


def _predict_tree_var(node: _Node, X: np.ndarray, out: np.ndarray, idx: np.ndarray) -> None:
    """与 ``_predict_tree`` 同路，但落的是**叶子的 τ̂ 方差**。

    单独写一个而不是给 ``_predict_tree`` 加参数：那条路径在预测里被调用多次，
    多一个"要不要算方差"的分支只会让两条语义混在一起（这也是本项目
    "别让一个入口干两件事"的一贯做法）。
    """
    if node.is_leaf:
        out[idx] = node.var
        return
    mask = X[idx, node.feature] <= node.threshold
    left, right = node.children
    if mask.any():
        _predict_tree_var(left, X, out, idx[mask])
    if (~mask).any():
        _predict_tree_var(right, X, out, idx[~mask])


class CausalTree:
    """单棵 honest 因果树。"""

    def __init__(self, config: ForestConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config or ForestConfig(n_trees=1)
        self.rng = rng or np.random.default_rng(0)
        self.root: _Node | None = None
        #: 这棵树用到的样本在**原始训练集**里的下标（森林 subsample 时才有）。
        #: 影响函数要跨树相加，就必须先说清"每棵树说的是谁的下标"。
        self.obs_idx: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        D: np.ndarray,
        Y: np.ndarray,
        obs_idx: np.ndarray | None = None,
    ) -> "CausalTree":
        n = X.shape[0]
        idx = np.arange(n)

        if self.config.honest:
            perm = self.rng.permutation(n)
            half = n // 2
            struct_idx, est_idx = perm[:half], perm[half:]
        else:
            struct_idx = est_idx = idx

        self.root = _build(
            X, D, Y, struct_idx,
            depth=0, config=self.config, rng=self.rng, n_features=X.shape[1],
        )
        self.obs_idx = None if obs_idx is None else np.asarray(obs_idx, dtype=int)
        _fill_leaf_effects(self.root, X, D, Y, est_idx, self.obs_idx)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("先调用 fit")
        out = np.empty(X.shape[0])
        _predict_tree(self.root, X, out, np.arange(X.shape[0]))
        return out


@dataclass
class CausalForest:
    """Bagged honest 因果森林。"""

    config: ForestConfig = field(default_factory=ForestConfig)
    trees: list[CausalTree] = field(default_factory=list)
    _fitted: bool = False
    #: 训练样本量 N。影响函数的方差是 ``Var(ψ)/N``，这个 N 必须是**原始训练
    #: 样本量**，不是某棵树 subsample 之后的大小。
    _n_train: int = 0

    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> "CausalForest":
        X = np.asarray(X, dtype=float)
        D = np.asarray(D, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n = X.shape[0]

        rng = np.random.default_rng(self.config.seed)
        self.trees = []
        for _ in range(self.config.n_trees):
            size = max(int(n * self.config.subsample), 2 * self.config.min_leaf)
            idx = rng.choice(n, size=min(size, n), replace=False)
            tree = CausalTree(self.config, rng)
            tree.fit(X[idx], D[idx], Y[idx], obs_idx=idx)
            self.trees.append(tree)

        if self.config.shrinkage > 0:
            self._shrink(X, D, Y)

        self._n_train = n
        self._fitted = True
        return self

    def _shrink(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> None:
        """把每棵树的叶子效应朝**全局 ATE** 收缩。

        叶子越小、越靠近噪声，收缩越强。这就是"弹性模型 vs 常数 ATE"
        这条光谱上的连续调节：``k = 0`` 完全不收缩，``k -> inf`` 退化成常数。
        """
        k = float(self.config.shrinkage)
        n1 = float((D > 0.5).sum())
        n0 = float((D < 0.5).sum())
        global_tau = (
            float(Y[D > 0.5].mean() - Y[D < 0.5].mean()) if n1 > 0 and n0 > 0 else 0.0
        )

        def walk(node: _Node) -> None:
            if node.is_leaf:
                w = node.n_est / (node.n_est + k) if node.n_est + k > 0 else 0.0
                node.tau = w * node.tau + (1.0 - w) * global_tau
                return
            left, right = node.children
            walk(left)
            walk(right)

        for t in self.trees:
            if t.root is not None:
                walk(t.root)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("先调用 fit")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        preds = np.vstack([t.predict(X) for t in self.trees])
        return preds.mean(axis=0)

    def predict_with_se(
        self, X: np.ndarray, combine: str = "influence"
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(tau_hat, se)`` —— M4 从"排序"走到"水平"的那一步。

        ``combine`` 有两种：

        * ``"influence"``（默认）：把各棵树的**影响函数相加**再做方差。
          ``ψ̄ = (1/B)Σ_b ψ_b``、``SE = sqrt(Σ_i ψ̄_i²)``。
          各棵树用同一份数据训练、它们**不独立**，而影响函数一相加，
          跨树协方差就自动进来了 —— 这正是 GRF 那套权重的方差写法。
        * ``"independent"``：旧的 ``sqrt(Σ_b σ²_b)/B``，把各棵树当独立量。
          **它是反保守的**（本仓库在 CS 聚合、SA 聚合、事件研究上已经修过
          三次同一个错误，森林是第四处），保留它**只是为了量出差别**，
          不要在产品路径上用它。

        仍然**没有**做到的：这个 SE 只覆盖"估计量的抽样变异"，
        **不覆盖点估计本身的偏差**（叶子内部的效应异质性 + 平滑偏差）。
        实测覆盖率仍远低于名义值，原因在偏差而不是方差
        （见 ``run_cate_coverage_audit`` 与 ``cate_interval_report.md``）。
        所以别把这个 SE 当成"校准好的区间"，它是"方差那一半修对了"。

        代价：``"influence"`` 需要一张 ``(n_pred, N)`` 的 ψ 矩阵，
        空间是 ``n_pred × N`` 个 float。这就是"跨树协方差不是免费的"——
        超限时直接报错，而不是偷偷退回旧算法。
        """
        if not self._fitted:
            raise RuntimeError("先调用 fit")
        if combine not in ("influence", "independent"):
            raise ValueError(f"combine 只能是 influence / independent，收到 {combine!r}")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n_pred = X.shape[0]
        n_trees = len(self.trees)
        taus = np.empty((n_trees, n_pred))
        vars_ = np.empty((n_trees, n_pred))
        idx = np.arange(n_pred)
        for b, tree in enumerate(self.trees):
            assert tree.root is not None
            _predict_tree(tree.root, X, taus[b], idx)
            _predict_tree_var(tree.root, X, vars_[b], idx)

        if combine == "independent":
            var_sum = np.where(np.isinf(vars_), np.inf, vars_).sum(axis=0)
            se = np.sqrt(var_sum) / n_trees
        else:
            n_train = self._n_train
            if n_train <= 0:
                raise RuntimeError("森林没有训练样本量（_n_train），无法做影响函数合成")
            budget = n_pred * n_train
            if budget > _MAX_IF_CELLS:
                raise ValueError(
                    f"影响函数合成要 (n_pred={n_pred} × N={n_train}) 的 ψ 矩阵，"
                    f"共 {budget} 个单元，超过上限 {_MAX_IF_CELLS}。"
                    "要么减少预测点、要么减小森林/样本，"
                    "要么显式用 combine='independent'（反保守，只建议用于对照）。"
                )
            psi = np.zeros((n_pred, n_train))
            for tree in self.trees:
                assert tree.root is not None
                _accumulate_tree_if(tree.root, X, psi, idx)
            psi /= n_trees
            # Σ_i ψ̄_i = 0 精确成立（每片叶子的残差按臂中心化），所以
            # Var(ψ̄) 就是均方；而 1/n_臂 已经把"样本量"算进权重里了，
            # 所以 Var(τ̂(x)) = Σ_i ψ̄_i²，**不再除 N**。
            se = np.sqrt((psi**2).sum(axis=1))

        # inf 表示某个叶子样本太少、给不出方差 → 该单元的 SE 也必须是 inf：
        # 影响函数合成会给出一个**有限**的数，但那不是"有方差"，而是"没材料"。
        # 这条判断必须保留，否则最不可靠的那批人反而拿到了最窄的区间。
        any_inf = np.isinf(vars_).any(axis=0)
        se = np.where(any_inf, np.inf, se)
        return taus.mean(axis=0), se

    @property
    def feature_importance(self) -> np.ndarray:
        """分裂次数按特征统计 —— 只是粗粒度的重要性，不做因果解读。"""
        if not self._fitted:
            raise RuntimeError("先调用 fit")
        p = None
        counts: dict[int, int] = {}

        def walk(node: _Node) -> None:
            if node.is_leaf:
                return
            counts[node.feature] = counts.get(node.feature, 0) + 1
            left, right = node.children
            walk(left)
            walk(right)

        for t in self.trees:
            if t.root is not None:
                walk(t.root)

        p = max(counts) + 1 if counts else 0
        out = np.zeros(p)
        for f, c in counts.items():
            out[f] = c
        total = out.sum()
        return out / total if total else out

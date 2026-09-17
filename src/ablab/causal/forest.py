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


@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None
    tau: float = 0.0
    n_struct: int = 0
    n_est: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0


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


def _tau_of(D: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> tuple[float, int, int]:
    """子样本上的 CATE 与两组人数。"""
    d = D[mask]
    y = Y[mask]
    n1 = int(d.sum())
    n0 = int(d.size - n1)
    if n1 == 0 or n0 == 0:
        return 0.0, n1, n0
    return float(y[d > 0.5].mean() - y[d < 0.5].mean()), n1, n0


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
    if mask.any():
        _assign_leaves(node.left, X, idx[mask], out)
    if (~mask).any():
        _assign_leaves(node.right, X, idx[~mask], out)


def _fill_leaf_effects(
    node: _Node, X: np.ndarray, D: np.ndarray, Y: np.ndarray, idx: np.ndarray
) -> None:
    """在 estimation 半样本上给叶子填效应值。"""
    if node.is_leaf:
        tau, n1, n0 = _tau_of(D, Y, idx)
        node.tau = tau
        node.n_est = n1 + n0
        return
    mask = X[idx, node.feature] <= node.threshold
    if mask.any():
        _fill_leaf_effects(node.left, X, D, Y, idx[mask])
    if (~mask).any():
        _fill_leaf_effects(node.right, X, D, Y, idx[~mask])


def _predict_tree(node: _Node, X: np.ndarray, out: np.ndarray, idx: np.ndarray) -> None:
    if node.is_leaf:
        out[idx] = node.tau
        return
    mask = X[idx, node.feature] <= node.threshold
    if mask.any():
        _predict_tree(node.left, X, out, idx[mask])
    if (~mask).any():
        _predict_tree(node.right, X, out, idx[~mask])


class CausalTree:
    """单棵 honest 因果树。"""

    def __init__(self, config: ForestConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config or ForestConfig(n_trees=1)
        self.rng = rng or np.random.default_rng(0)
        self.root: _Node | None = None

    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> "CausalTree":
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
        _fill_leaf_effects(self.root, X, D, Y, est_idx)
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
            tree.fit(X[idx], D[idx], Y[idx])
            self.trees.append(tree)

        if self.config.shrinkage > 0:
            self._shrink(X, D, Y)

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
            walk(node.left)
            walk(node.right)

        for t in self.trees:
            walk(t.root)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("先调用 fit")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        preds = np.vstack([t.predict(X) for t in self.trees])
        return preds.mean(axis=0)

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
            walk(node.left)
            walk(node.right)

        for t in self.trees:
            walk(t.root)

        p = max(counts) + 1 if counts else 0
        out = np.zeros(p)
        for f, c in counts.items():
            out[f] = c
        total = out.sum()
        return out / total if total else out

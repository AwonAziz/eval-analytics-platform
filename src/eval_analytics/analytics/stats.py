"""Statistics for the analytical queries.

Two ideas do most of the work here.

**Paired bootstrap.** The two arms are scored on the same documents, so the
difference in accuracy is a paired statistic: resampling documents (not
runs) and recomputing the delta gives a confidence interval that is far
tighter than treating the two arms as independent samples. Resampling
documents is the right unit because the pairing is at document level.

**McNemar exact test.** Accuracy differences between two models on the same
examples are driven by the *discordant* pairs -- examples exactly one model
gets right. McNemar's test is built on that 2x2 table and is the correct
significance test for paired binary outcomes, where a two-proportion z-test
would overstate the evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

#: Bootstrap resamples. 4000 puts the 2.5th percentile at ~100 effective
#: samples in the tail, enough for a stable 95% interval.
DEFAULT_N_BOOTSTRAP = 4000
DEFAULT_SEED = 20261005
_BOOTSTRAP_BLOCK = 250


@dataclass
class PairedResult:
    """A paired comparison between two models over a shared example set."""

    n: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_a_only: int
    n_b_only: int
    n_both: int
    n_neither: int
    n_boot: int

    @property
    def significant(self) -> bool:
        """Whether the paired delta excludes zero at the 95% level."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["significant"] = self.significant
        return out


def _as_bool_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == object:
        return arr.astype(str).map({"1": True, "0": False, "True": True, "False": False}).fillna(False).to_numpy(bool)
    return arr.astype(bool)


def mcnemar_exact(n_a_only: int, n_b_only: int) -> float:
    """Two-sided exact McNemar p-value for the discordant pairs.

    Under the null, the direction of each discordant pair is a fair coin, so
    the count of ``a_only`` wins is Binomial(n_a_only + n_b_only, 0.5). With
    zero discordant pairs the two models agree everywhere and the test is
    undefined, reported as p = 1.0.
    """
    from scipy.stats import binomtest

    n_discordant = n_a_only + n_b_only
    if n_discordant == 0:
        return 1.0
    k = min(n_a_only, n_b_only)
    return float(binomtest(k, n_discordant, 0.5, alternative="two-sided").pvalue)


def paired_bootstrap(
    a: Any,
    b: Any,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> PairedResult:
    """Paired bootstrap of ``mean(a) - mean(b)`` with an exact McNemar test.

    ``a`` and ``b`` are aligned correctness vectors for the same examples in
    the same order.
    """
    a_arr = _as_bool_array(a)
    b_arr = _as_bool_array(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired inputs must align: {a_arr.shape} vs {b_arr.shape}")
    if a_arr.size == 0:
        raise ValueError("paired comparison needs at least one example")

    a_f, b_f = a_arr.astype(np.float64), b_arr.astype(np.float64)
    n = a_f.size
    diff = a_f - b_f
    delta = float(diff.mean())

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, _BOOTSTRAP_BLOCK):
        stop = min(start + _BOOTSTRAP_BLOCK, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        deltas[start:stop] = diff[idx].mean(axis=1)

    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))

    a_only = int((a_arr & ~b_arr).sum())
    b_only = int((~a_arr & b_arr).sum())

    return PairedResult(
        n=n,
        mean_a=float(a_f.mean()),
        mean_b=float(b_f.mean()),
        delta=delta,
        ci_low=lo,
        ci_high=hi,
        p_value=mcnemar_exact(a_only, b_only),
        n_a_only=a_only,
        n_b_only=b_only,
        n_both=int((a_arr & b_arr).sum()),
        n_neither=int((~a_arr & ~b_arr).sum()),
        n_boot=n_boot,
    )


@dataclass
class CalibrationBucket:
    """One confidence bin, with its share of total ECE."""

    bin_low: float
    bin_high: float
    n: int
    accuracy: float
    avg_confidence: float
    gap: float
    ece_contribution: float
    share_of_ece: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationSummary:
    """Calibration for one arm, decomposed by confidence bucket."""

    model_id: str
    arm: str
    quantisation: str
    n: int
    accuracy: float
    mean_confidence: float
    ece: float
    mce: float
    brier: float
    overconfidence: float
    buckets: list[CalibrationBucket]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["buckets"] = [b.to_dict() for b in self.buckets]
        return out


def calibration(
    model_id: str,
    arm: str,
    quantisation: str,
    confidence: Any,
    correct: Any,
    prob_true: Any,
    n_bins: int = 15,
) -> CalibrationSummary:
    """Binned reliability diagram and the ECE decomposition.

    The decomposition is the point: ``ece_contribution`` is
    ``n_bin * |accuracy - confidence| / N``, so the buckets that dominate ECE
    are visible directly rather than inferred. ``overconfidence`` is
    ``mean_confidence - accuracy`` -- positive means the model states more
    confidence than it earns.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    corr = _as_bool_array(correct)
    p_true = np.asarray(prob_true, dtype=np.float64) if prob_true is not None else None

    if conf.shape != corr.shape:
        raise ValueError(f"confidence/correct must align: {conf.shape} vs {corr.shape}")

    n = conf.size
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    buckets: list[CalibrationBucket] = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # The first bucket is closed on both ends so a confidence of exactly
        # 0 is counted rather than falling outside every bucket.
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(corr[mask].mean())
        avg_conf = float(conf[mask].mean())
        gap = abs(acc - avg_conf)
        ece += count * gap / n
        mce = max(mce, gap)
        buckets.append(
            CalibrationBucket(
                bin_low=lo,
                bin_high=hi,
                n=count,
                accuracy=acc,
                avg_confidence=avg_conf,
                gap=gap,
                ece_contribution=count * gap / n,
                share_of_ece=0.0,
            )
        )

    for bucket in buckets:
        bucket.share_of_ece = bucket.ece_contribution / ece if ece > 0 else 0.0

    brier = float(np.mean((p_true - corr.astype(np.float64)) ** 2)) if p_true is not None else float("nan")
    mean_conf = float(conf.mean())
    accuracy = float(corr.mean())

    return CalibrationSummary(
        model_id=model_id,
        arm=arm,
        quantisation=quantisation,
        n=n,
        accuracy=accuracy,
        mean_confidence=mean_conf,
        ece=float(ece),
        mce=float(mce),
        brier=brier,
        overconfidence=mean_conf - accuracy,
        buckets=buckets,
    )

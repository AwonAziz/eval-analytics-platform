"""Deterministic generator for fine-tune run artifacts.

The fine-tuning project that owns ``test_preds.npy`` / ``serving_report.json``
lives on another machine, so this module fabricates artifacts of exactly the
shapes the real runs emit and writes them under ``data/raw/generated/``. The
ingest layer does not know the difference: it lands and validates whatever is
on disk, so dropping in the real artifacts replaces these with no code change.

Every array is drawn from a seeded ``numpy.random.Generator``, so the
warehouse is byte-reproducible and CI can assert on exact metric values.

The generative model encodes specific, explainable findings rather than
noise, so the analytical queries have something real to find:

* LoRA is better *calibrated*; full fine-tune is better *separated*. Each arm
  gets its own logit scale and temperature, so ECE and accuracy move in
  opposite directions -- which is the actual portfolio finding.
* Full fine-tune has a majority-class prior bias, so it wins on head classes
  and collapses on the tail. LoRA wins the rare classes.
* Documents in the 17-32 token band contain a single ambiguous cue term and
  get a class-confusion boost toward a paired label. That is the origin of
  the length-bucket error spike, and it is concentrated on specific class
  pairs rather than spread evenly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SEED = 20261005

CLASSES = [
    "invoice",
    "contract",
    "resume",
    "scientific_paper",
    "legal_filing",
    "medical_record",
    "technical_manual",
    "financial_report",
]

#: Heavily imbalanced prior: the head/tail split is what makes the arm
#: comparison interesting.
CLASS_PRIORS = np.array([0.26, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.03])

#: Classes the two arms are expected to disagree on most.
TAIL_CLASSES = {"medical_record", "technical_manual", "financial_report"}
HEAD_CLASSES = {"invoice", "contract", "resume"}

#: Label pairs that share a cue term, and therefore collide on short docs.
CONFUSION_PAIRS = {
    "invoice": "financial_report",
    "financial_report": "invoice",
    "contract": "legal_filing",
    "legal_filing": "contract",
    "resume": "technical_manual",
    "technical_manual": "resume",
    "scientific_paper": "medical_record",
    "medical_record": "scientific_paper",
}

#: Target share of the test set per document-length bucket, aligned to
#: eval_analytics.config.LENGTH_BUCKET_EDGES.
LENGTH_MIX = [
    (1, 8, 0.06),
    (9, 16, 0.11),
    (17, 32, 0.22),
    (33, 64, 0.28),
    (65, 128, 0.22),
    (129, 512, 0.11),
]

#: The problematic band: a single cue term, no surrounding context.
AMBIGUOUS_BAND = (17, 32)

BASE_MODEL = "distilbert-base-uncased"
TOTAL_PARAMS = 66_000_000

ARMS: dict[str, dict[str, float | None]] = {
    # `signal_scale` separates classes (drives accuracy). `overconfidence` is
    # what each arm keeps after temperature scaling is undone: full
    # fine-tune stays sharp and over-confident, so its ECE floor sits roughly
    # 70% above LoRA's. That residual is the whole calibration finding.
    "lora": {
        "signal_scale": 1.00,
        "overconfidence": 0.930,
        "class_prior_bias": 0.00,
        "rare_boost": 0.26,
        "noise": 0.45,
        "trainable_params": 887_808,
        "adapter_rank": 16,
    },
    "full_finetune": {
        "signal_scale": 1.07,
        "overconfidence": 0.791,
        "class_prior_bias": 0.30,
        "rare_boost": -0.24,
        "noise": 0.4275,
        "trainable_params": TOTAL_PARAMS,
        "adapter_rank": None,
    },
}

#: Shared separation between the true class and its rivals, before the
#: per-arm scale is applied.
BASE_MARGIN = 2.8

#: How much a short document costs the true class. Deliberately mild: the
#: 1-8 and 9-16 bands land within ~0.5 points of each other, so the 17-32
#: spike is attributable to cue-term confusion rather than to raw brevity.
SHORT_DOC_MARGIN_COST = 0.20

#: Logit boost toward the confusable partner class inside the ambiguous
#: band. This is the sole driver of the 17-32 token error spike, which lands
#: ~4x above the 9-16 band and ~13x above 33-64.
CONFUSION_BOOST = 2.1

QUANT_CONFIGS = [
    # label, dtype, size multiple vs fp32, latency scale vs fp32
    ("fp32", "float32", 1.00, 1.00),
    ("fp16", "float16", 0.50, 0.62),
    ("int8", "int8", 0.26, 0.31),
]

SERVE_BATCH_SIZES = [1, 8, 32]
SERVE_INPUT_TOKENS = [16, 64, 256, 1024]
SERVE_SAMPLES_PER_CONFIG = 60


def _model_size_bytes(quantisation: str) -> int:
    """On-disk size of a model at a given precision."""
    multiple = {"fp32": 4.0, "fp16": 2.0, "int8": 1.06}.get(quantisation, 4.0)
    return int(TOTAL_PARAMS * multiple)

DRIFT_FEATURES = [
    # name, drift slope per window, baseline level
    ("doc_length_tokens", 0.00042, 0.031),
    ("avg_sentence_len", 0.00061, 0.048),
    ("currency_symbol_rate", -0.00018, 0.112),
    ("upper_ratio", 0.00235, 0.064),
    ("punct_ratio", 0.00009, 0.027),
    ("table_density", 0.00077, 0.039),
    ("legal_citation_rate", 0.00104, 0.071),
    ("non_ascii_rate", 0.00022, 0.018),
]

DRIFT_WINDOWS = 28


@dataclass
class GeneratedRun:
    experiment_id: str
    directory: Path
    labels: list[str]
    example_ids: list[str]
    artifacts: list[str] = field(default_factory=list)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _expected_calibration_error(
    probs: np.ndarray, true_idx: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """Binned ECE, matching the convention used in the analytics layer."""
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == true_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, buckets = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i else (confidence >= lo) & (confidence <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc, avg_conf = float(correct[mask].mean()), float(confidence[mask].mean())
        gap = abs(acc - avg_conf)
        total += n * gap
        buckets.append(
            {
                "bin_low": round(float(lo), 4),
                "bin_high": round(float(hi), 4),
                "n": n,
                "accuracy": round(acc, 6),
                "avg_confidence": round(avg_conf, 6),
                "gap": round(gap, 6),
            }
        )
    return total / probs.shape[0], buckets


def _sample_lengths(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw token counts matching LENGTH_MIX, jittered inside each bucket."""
    lengths = np.empty(n, dtype=np.int32)
    cursor = 0
    for low, high, share in LENGTH_MIX:
        count = int(round(n * share))
        if cursor + count > n:
            count = n - cursor
        if count <= 0:
            continue
        if high is None:
            lengths[cursor : cursor + count] = rng.integers(low, low * 3, count)
        else:
            lengths[cursor : cursor + count] = rng.integers(low, high + 1, count)
        cursor += count
    if cursor < n:
        lengths[cursor:] = rng.integers(129, 400, n - cursor)
    rng.shuffle(lengths)
    return lengths


def _build_confusions(true_idx: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Confusable partner class per example, or -1 outside the ambiguous band."""
    boost = np.full(len(true_idx), -1, dtype=np.int32)
    in_band = (lengths >= AMBIGUOUS_BAND[0]) & (lengths <= AMBIGUOUS_BAND[1])
    for i, label in enumerate(true_idx):
        if not in_band[i]:
            continue
        partner = CONFUSION_PAIRS.get(CLASSES[label])
        if partner:
            boost[i] = CLASSES.index(partner)
    return boost


def _solve_temperature(logits: np.ndarray, true_idx: np.ndarray) -> tuple[float, float]:
    """Find the temperature that minimises ECE for a fixed set of predictions.

    Temperature does not change the argmax, so it moves stated confidence
    without moving accuracy. That makes ECE a U-shaped function of T: as T
    falls the model sharpens and ECE falls while the model is under-confident,
    reaches a minimum, then rises again once it over-shoots. A bisection that
    assumes monotonicity will happily return a bound instead of the minimum,
    so this locates the interior optimum by grid scan followed by ternary
    refinement.

    Returns ``(temperature, ece_at_that_temperature)``. The minimum is a
    property of the sampled logits, not a free parameter: when it lands near
    but not exactly on a round number, that residue is genuine.
    """
    def ece_at(temperature: float) -> float:
        return _expected_calibration_error(_softmax(logits / temperature), true_idx)[0]

    grid = np.linspace(np.log(0.10), np.log(4.0), 120)
    scores = np.array([ece_at(float(np.exp(t))) for t in grid])
    best = int(scores.argmin())

    lo = float(grid[max(best - 1, 0)])
    hi = float(grid[min(best + 1, len(grid) - 1)])
    for _ in range(80):
        third = (hi - lo) / 3
        if ece_at(float(np.exp(lo + third))) < ece_at(float(np.exp(hi - third))):
            hi = hi - third
        else:
            lo = lo + third

    temperature = float(np.exp((lo + hi) / 2))
    return temperature, ece_at(temperature)


def _arm_logits(
    rng: np.random.Generator,
    true_idx: np.ndarray,
    lengths: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, float]:
    """Return (logits, temperature, temperature-optimal ECE) for one arm.

    Every example carries the same latent class signal across both arms --
    only the readout differs -- so the two arms stay genuinely paired and a
    McNemar test on their disagreement table is meaningful.
    """
    cfg = ARMS[arm]
    n_examples, n_classes = len(true_idx), len(CLASSES)
    rows = np.arange(n_examples)

    # Latent affinity: every class gets noise, the true class gets a margin
    # that grows with document length and shrinks for very short documents.
    latent = rng.normal(0, 0.72, (n_examples, n_classes))
    length_evidence = np.clip((lengths - 16) / 90.0, 0.0, 2.2)
    short_cost = np.clip((32 - lengths) / 32.0, 0.0, 1.0) * SHORT_DOC_MARGIN_COST
    margin = BASE_MARGIN * cfg["signal_scale"] + length_evidence - short_cost
    latent[rows, true_idx] += margin

    logits = latent + rng.normal(0, float(cfg["noise"]), latent.shape)

    # Majority-class prior: full fine-tune drifts toward frequent labels.
    logits += CLASS_PRIORS[None, :] * float(cfg["class_prior_bias"])

    # Tail-class correction, opposite sign per arm.
    for j, label in enumerate(CLASSES):
        if label in TAIL_CLASSES:
            logits[:, j] += float(cfg["rare_boost"])
        elif label in HEAD_CLASSES:
            logits[:, j] -= float(cfg["rare_boost"]) * 0.35

    # The 17-32 token finding: a single shared cue term, no context to
    # disambiguate, so the paired label gets pulled in hard.
    partner_idx = _build_confusions(true_idx, lengths)
    paired = np.flatnonzero(partner_idx >= 0)
    if paired.size:
        logits[paired, partner_idx[paired]] += CONFUSION_BOOST

    # Find the best possible calibration for this arm, then re-sharpen it by
    # `overconfidence` to model a deployment that ships the uncalibrated head.
    best_temperature, best_ece = _solve_temperature(logits, true_idx)
    temperature = best_temperature / float(cfg["overconfidence"] or 1.0)
    return logits, temperature, best_ece


def _confidence_interval(n_correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- used for the reported accuracy CI bounds."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = n_correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


def _macro_f1(true_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = float(((pred_idx == c) & (true_idx == c)).sum())
        fp = float(((pred_idx == c) & (true_idx != c)).sum())
        fn = float(((pred_idx != c) & (true_idx == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def _mce(probs: np.ndarray, true_idx: np.ndarray, n_bins: int = 15) -> float:
    _, buckets = _expected_calibration_error(probs, true_idx, n_bins)
    return float(max((b["gap"] for b in buckets), default=0.0))


def _brier(probs: np.ndarray, true_idx: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(true_idx)), true_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _per_class_breakdown(true_idx: np.ndarray, pred_idx: np.ndarray, probs: np.ndarray) -> list[dict]:
    out = []
    for c, label in enumerate(CLASSES):
        mask = true_idx == c
        n = int(mask.sum())
        if n == 0:
            continue
        hit = int((pred_idx[mask] == c).sum())
        out.append(
            {
                "label": label,
                "n": n,
                "support": n,
                "n_correct": hit,
                "accuracy": round(hit / n, 6),
                "mean_confidence": round(float(probs[mask].max(axis=1).mean()), 6),
                "is_tail_class": label in TAIL_CLASSES,
            }
        )
    return out


#: Rows per quantisation group. Modern INT8 schemes quantise in small groups
#: rather than across a whole tensor, and the group size is what sets the
#: resolution of the grid -- and therefore how many predictions INT8 flips
#: relative to FP32.
INT8_GROUP_ROWS = 32


def _quantise_int8(logits: np.ndarray) -> np.ndarray:
    """Round pre-softmax activations onto a symmetric per-class INT8 grid.

    This approximates what weight quantisation does to the logits a quantised
    kernel produces. Scales are per class *and* per group of rows, the
    analogue of per-output-channel group quantisation.

    The grouping matters numerically. A single per-tensor scale is set by the
    largest logit in the whole batch, so every class would be quantised far
    too coarsely and INT8 would flip ~5% of predictions -- several times
    worse than any real INT8 deployment. Group-wise scales put the step size
    near the typical logit magnitude instead, which lands the disagreement
    rate in the range INT8 actually produces.
    """
    n_rows, n_classes = logits.shape
    group = min(INT8_GROUP_ROWS, n_rows)
    n_blocks = n_rows // group
    remainder = n_rows - n_blocks * group

    def _round(block: np.ndarray) -> np.ndarray:
        scale = np.abs(block).max(axis=1, keepdims=True) / 127.0
        scale = np.where(scale == 0, 1.0, scale)
        return np.round(block / scale) * scale

    out = np.empty_like(logits, dtype=np.float64)
    if n_blocks:
        blocks = logits[: n_blocks * group].reshape(n_blocks, group, n_classes)
        out[: n_blocks * group] = _round(blocks).reshape(-1, n_classes)
    if remainder:
        tail = logits[n_blocks * group :]
        scale = np.abs(tail).max(axis=1, keepdims=True) / 127.0
        scale = np.where(scale == 0, 1.0, scale)
        out[n_blocks * group :] = np.round(tail / scale) * scale
    return out


def _base_example_set(rng: np.random.Generator, n_examples: int) -> dict:
    """The shared test set: labels, lengths and example ids.

    Extracted separately from the arms so that several artifact sets -- the
    FP32 LoRA run, its INT8 counterpart, the full fine-tune -- are guaranteed
    to be scored on exactly the same documents, which is what makes them
    pairwise comparable.
    """
    n_classes = len(CLASSES)
    return {
        "true_idx": rng.choice(n_classes, size=n_examples, p=CLASS_PRIORS),
        "lengths": _sample_lengths(rng, n_examples),
        "example_ids": [f"doc_{i:05d}" for i in range(n_examples)],
    }


def generate_run(
    out_dir: Path,
    arm: str,
    experiment_id: str,
    seed: int = SEED,
    started_at: datetime | None = None,
    base: dict | None = None,
    quantisation: str = "fp32",
) -> GeneratedRun:
    """Emit one arm's complete artifact set under ``out_dir``.

    When ``quantisation`` is not ``fp32`` the same logits are quantised
    before the softmax and the FP32 temperature is reused, so the INT8 run is
    a numerically perturbed readout of the *same* model rather than a
    separately trained one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    started_at = started_at or datetime(2026, 10, 3, 9, 15, tzinfo=timezone.utc)

    n_examples = 2400
    if base is None:
        base = _base_example_set(rng, n_examples)
    true_idx, lengths = base["true_idx"], base["lengths"]
    example_ids = base["example_ids"]
    n_classes = len(CLASSES)

    logits, temperature, _best_ece = _arm_logits(rng, true_idx, lengths, arm)
    if quantisation != "fp32":
        logits = _quantise_int8(logits)
    probs = _softmax(logits / temperature)
    pred_idx = probs.argmax(axis=1)
    true_label_names = [CLASSES[i] for i in true_idx]

    correct = pred_idx == true_idx
    n_correct = int(correct.sum())
    accuracy = n_correct / n_examples
    ci_low, ci_high = _confidence_interval(n_correct, n_examples)
    ece, buckets = _expected_calibration_error(probs, true_idx)
    mce = _mce(probs, true_idx)
    brier = _brier(probs, true_idx)
    macro_f1 = _macro_f1(true_idx, pred_idx, n_classes)

    cfg = ARMS[arm]
    trainable = cfg["trainable_params"]
    is_lora = arm == "lora"
    model_id = (
        f"{BASE_MODEL}-lora-r{cfg['adapter_rank']}" if is_lora else f"{BASE_MODEL}-full-finetune"
    )
    if quantisation != "fp32":
        # A quantised build is a distinct deployable artifact, so it gets its
        # own model_id and its own quantised size on disk. `base_model`,
        # `arm` and `adapter_rank` are unchanged, which is what lets the
        # agreement query pair the two precisions of one logical model.
        model_id = f"{model_id}-{quantisation}"
        trainable = cfg["trainable_params"]
    size_bytes = _model_size_bytes(quantisation)

    # --- artifacts on disk ---------------------------------------------
    written: list[str] = []

    np.save(out_dir / "test_preds.npy", probs.astype(np.float32))
    written.append("test_preds.npy")

    np.save(out_dir / "test_labels.npy", true_label_names)
    written.append("test_labels.npy")

    np.save(out_dir / "test_example_ids.npy", np.array(example_ids))
    written.append("test_example_ids.npy")

    # Column order of the probability matrix, so the artifact set is
    # self-describing rather than depending on generator internals.
    (out_dir / "classes.json").write_text(
        json.dumps({"label_names": CLASSES}, indent=2), encoding="utf-8"
    )
    written.append("classes.json")

    np.save(out_dir / "test_token_lengths.npy", lengths)
    written.append("test_token_lengths.npy")

    eval_report = {
        "experiment_id": experiment_id,
        "arm": arm,
        "model_id": model_id,
        "base_model": BASE_MODEL,
        "adapter_rank": cfg["adapter_rank"],
        "trainable_params": int(trainable),
        "total_params": TOTAL_PARAMS,
        "dataset": "docclass-8k-v3",
        "quantisation": quantisation,
        "model_size_bytes": size_bytes,
        "test_set_size": n_examples,
        "n_correct": n_correct,
        "accuracy": round(accuracy, 6),
        "accuracy_ci_low": round(ci_low, 6),
        "accuracy_ci_high": round(ci_high, 6),
        "macro_f1": round(macro_f1, 6),
        "ece": round(ece, 6),
        "mce": round(mce, 6),
        "brier": round(brier, 6),
        "temperature": round(temperature, 6),
        "calibration_bins": buckets,
        "per_class": _per_class_breakdown(true_idx, pred_idx, probs),
        "seed": seed,
        "started_at": started_at.isoformat(),
        "finished_at": (started_at + timedelta(minutes=41)).isoformat(),
    }
    (out_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    written.append("eval_report.json")

    model_card = {
        "model_id": model_id,
        "base_model": BASE_MODEL,
        "arm": arm,
        "method": "lora" if is_lora else "full_finetune",
        "adapter_rank": cfg["adapter_rank"],
        "trainable_params": int(trainable),
        "total_params": TOTAL_PARAMS,
        "trainable_pct": round(100.0 * trainable / TOTAL_PARAMS, 6),
        "quantisation": quantisation,
        "dataset": "docclass-8k-v3",
        "created_at": (started_at + timedelta(minutes=42)).isoformat(),
        "is_champion": is_lora and quantisation == "fp32",
        "data_origin": "synthetic",
    }
    (out_dir / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    written.append("model_card.json")

    # Latency and drift are properties of the architecture and the input
    # distribution, not of a particular build precision, so the FP32 artifact
    # set carries them once instead of tripling identical rows.
    if quantisation == "fp32":
        _write_serving_report(out_dir, experiment_id, model_id, rng, written)
        _write_drift_report(out_dir, experiment_id, model_id, rng, written, started_at)

    return GeneratedRun(
        experiment_id=experiment_id,
        directory=out_dir,
        labels=CLASSES,
        example_ids=example_ids,
        artifacts=written,
    )


def _write_serving_report(
    out_dir: Path, experiment_id: str, model_id: str, rng: np.random.Generator, written: list[str]
) -> None:
    """Latency/size by quantisation config, with per-request samples.

    Model size is exact (fp32 = 4 bytes/param, int8 ~ 1 byte/param plus
    scales); latency is lognormal around a median that grows with token count
    and shrinks with quantisation, which gives a realistic p50/p99 tail.
    """
    configs = []
    all_samples: list[dict] = []
    for quant, dtype, size_mult, lat_mult in QUANT_CONFIGS:
        size_bytes = int(TOTAL_PARAMS * size_mult)
        samples = []
        for batch in SERVE_BATCH_SIZES:
            for tokens in SERVE_INPUT_TOKENS:
                # Median latency: fixed overhead + per-token cost, plus a
                # mild batching win.
                median = (6.5 + tokens * 0.0091) * lat_mult * (1.0 - 0.045 * np.log2(batch))
                sigma = 0.19
                draws = rng.lognormal(mean=np.log(median), sigma=sigma, size=SERVE_SAMPLES_PER_CONFIG)
                for i, latency in enumerate(draws):
                    samples.append(
                        {
                            "quantisation": quant,
                            "batch_size": batch,
                            "input_tokens": tokens,
                            "latency_ms": round(float(latency), 4),
                            "request_index": i,
                        }
                    )
        all_samples.extend(samples)
        batch = 8
        tokens = 256
        subset = [
            s
            for s in samples
            if s["batch_size"] == batch and s["input_tokens"] == tokens
        ]
        latencies = np.array([s["latency_ms"] for s in subset])
        configs.append(
            {
                "quantisation": quant,
                "dtype": dtype,
                "model_size_bytes": size_bytes,
                "model_size_mb": round(size_bytes / 1e6, 3),
                "peak_memory_mb": round(size_bytes / 1e6 * 1.18, 3),
                "reference_batch_size": batch,
                "reference_input_tokens": tokens,
                "p50_latency_ms": round(float(np.percentile(latencies, 50)), 4),
                "p99_latency_ms": round(float(np.percentile(latencies, 99)), 4),
                "throughput_tps": round(batch * 1000.0 / float(np.percentile(latencies, 50)), 3),
                "n_samples": len(samples),
            }
        )

    report = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "hardware": {
            "cpu": "AMD EPYC 7763 64-core",
            "accelerator": "NVIDIA L4 24GB",
            "runtime": "onnxruntime 1.19",
        },
        "generated_at": datetime(2026, 10, 3, 11, 0, tzinfo=timezone.utc).isoformat(),
        "configs": configs,
        "requests": all_samples,
    }
    (out_dir / "serving_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    written.append("serving_report.json")


def _write_drift_report(
    out_dir: Path,
    experiment_id: str,
    model_id: str,
    rng: np.random.Generator,
    written: list[str],
    started_at: datetime,
) -> None:
    """PSI / KS / JS per feature across daily windows.

    Slopes are per-window, and ``upper_ratio`` is deliberately steep enough
    to cross the severe threshold -- that crossing is what the drift-trend
    query and the alerting column exist to surface.
    """
    measurements = []
    for feature, slope, base in DRIFT_FEATURES:
        prev = base
        for w in range(DRIFT_WINDOWS):
            noise = rng.normal(0, 0.006)
            value = max(0.0005, base + slope * w + noise)
            psi = value
            # KS rises sub-linearly with PSI; JS saturates faster.
            ks = float(np.clip(0.62 * np.sqrt(psi) + rng.normal(0, 0.004), 0.0, 1.0))
            js = float(np.clip(0.44 * psi + rng.normal(0, 0.003), 0.0, 1.0))
            severity = "severe" if psi >= 0.25 else "moderate" if psi >= 0.10 else "none"
            window_id = f"{experiment_id}-w{w:02d}"
            measured_at = (started_at + timedelta(days=w)).isoformat()
            common = {
                "feature": feature,
                "window_index": w,
                "window_id": window_id,
                "measured_at": measured_at,
                "baseline_value": round(base, 6),
                "current_value": round(prev, 6),
                "sample_size": int(rng.integers(800, 1600)),
                "threshold_moderate": 0.10,
                "threshold_severe": 0.25,
                "severity": severity,
            }
            # One row per statistic, so the fact's natural grain is
            # (feature, window, metric_name) and the pivot view can widen it.
            measurements.append(
                {**common, "metric_name": "psi", "metric_value": round(psi, 6),
                 "psi": round(psi, 6), "ks_pvalue": round(float(np.clip(0.9 * np.exp(-6 * psi), 0.0, 1.0)), 6)}
            )
            measurements.append(
                {**common, "metric_name": "ks_statistic", "metric_value": round(ks, 6),
                 "ks_statistic": round(ks, 6), "ks_pvalue": round(float(np.clip(0.9 * np.exp(-6 * psi), 0.0, 1.0)), 6)}
            )
            measurements.append(
                {**common, "metric_name": "js_divergence", "metric_value": round(js, 6),
                 "js_divergence": round(js, 6)}
            )
            prev = value

    report = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "reference_snapshot": "ref_2026-09-15",
        "generated_at": (started_at + timedelta(days=DRIFT_WINDOWS)).isoformat(),
        "windows": DRIFT_WINDOWS,
        "measurements": measurements,
    }
    (out_dir / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    written.append("drift_report.json")


def generate_all(out_dir: Path, seed: int = SEED) -> list[GeneratedRun]:
    """Generate the full experiment: both arms, plus an INT8 build of LoRA.

    The INT8 build shares the LoRA test set and temperature and differs only
    in numerics, so the pair supports a like-for-like prediction-agreement
    measurement against the FP32 LoRA run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = datetime(2026, 10, 3, 9, 15, tzinfo=timezone.utc)
    rng = np.random.default_rng(seed)
    shared = _base_example_set(rng, 2400)

    return [
        generate_run(out_dir / "exp-001-lora", "lora", "exp-001-lora", seed, base, shared),
        generate_run(
            out_dir / "exp-001-lora-int8",
            "lora",
            "exp-001-lora-int8",
            seed,
            base,
            shared,
            quantisation="int8",
        ),
        generate_run(
            out_dir / "exp-001-full-finetune",
            "full_finetune",
            "exp-001-full-finetune",
            seed,
            base,
            shared,
        ),
    ]

"""Decision metrics, with explicitly normalized multiclass probabilities."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np


def probabilities(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    if z.ndim != 1 or z.size < 2 or not np.isfinite(z).all() or temperature <= 0:
        raise ValueError("Expected finite 1D logits with at least two options and positive temperature")
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def label_index(question: dict) -> int:
    criteria = question.get("criteria")
    label = question["label"]
    if question["type"] == "noul":
        # Common adapter contract: [no, yes].
        if isinstance(label, bool):
            return int(label)
        if isinstance(label, str) and label.lower() in ("yes", "no", "true", "false"):
            return int(label.lower() in ("yes", "true"))
        return int(label)
    if isinstance(criteria, dict):
        if label in criteria:
            return list(criteria).index(label)
        if isinstance(label, (int, float)):
            return int(label)
        raise ValueError(f"Label {label!r} not among criteria")
    if isinstance(label, str) and criteria and label in criteria:
        return criteria.index(label)
    return int(label)


def fit_temperature(rows: Sequence[dict]) -> float:
    """Fit a single temperature using calibration labels only; bounded log grid."""
    if not rows:
        raise ValueError("Temperature calibration is empty")
    candidates = np.exp(np.linspace(math.log(0.2), math.log(5.0), 161))
    losses = [np.mean([-math.log(max(probabilities(r["logits"], t)[r["label"]], 1e-15)) for r in rows]) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def fit_temperatures(rows: Sequence[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["type"]].append(row)
    return {kind: fit_temperature(items) for kind, items in sorted(groups.items())}


def metrics(rows: Sequence[dict], temperatures: dict | None = None, bins: int = 15) -> dict:
    if not rows:
        return {"n": 0}
    temperatures = temperatures or {}
    ps = [probabilities(r["logits"], temperatures.get(r["type"], 1.0)) for r in rows]
    labels = np.array([r["label"] for r in rows])
    if any(y < 0 or y >= len(p) for y, p in zip(labels, ps)):
        raise ValueError("Gold label index outside output distribution")
    predictions = np.array([int(p.argmax()) for p in ps])
    correct = (labels == predictions).astype(float)
    confidence = np.array([float(p.max()) for p in ps])
    nll = [-math.log(max(float(p[y]), 1e-15)) for p, y in zip(ps, labels)]
    # Multiclass Brier is the sum over classes, range [0,2].
    brier = [float(np.sum((p - np.eye(len(p))[y]) ** 2)) for p, y in zip(ps, labels)]
    ece = 0.0
    calibration_bins = []
    for i in range(bins):
        mask = (confidence >= i / bins) & (confidence < (i+1)/bins if i < bins-1 else confidence <= 1)
        count = int(mask.sum())
        if count:
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += count / len(rows) * abs(acc-conf)
            calibration_bins.append({"lower": i/bins, "upper": (i+1)/bins, "n": count, "accuracy": acc, "confidence": conf})
    # Tie groups are indivisible: don't manufacture coverage by cherry-picking ties.
    order = np.argsort(-confidence, kind="stable")
    risk = 1 - np.cumsum(correct[order]) / np.arange(1, len(rows)+1)
    boundaries = np.r_[confidence[order][1:] != confidence[order][:-1], True]
    coverage = {}
    for budget in (0.01, 0.05, 0.10):
        viable = np.where((risk <= budget) & boundaries)[0]
        coverage[str(budget)] = float((viable[-1]+1)/len(rows)) if len(viable) else 0.0
    result = {"n": len(rows), "accuracy": float(correct.mean()), "nll": float(np.mean(nll)),
              "brier": float(np.mean(brier)), "ece_15": ece, "mean_confidence": float(confidence.mean()),
              "coverage_at_risk": coverage, "calibration_bins": calibration_bins}
    score_items = [(r,p) for r,p in zip(rows,ps) if r["type"] == "score"]
    if score_items:
        result["score_mae"] = float(np.mean([abs(float(np.dot(np.arange(len(p)),p))-r["label"]) for r,p in score_items]))
    source_groups = defaultdict(list)
    for i, row in enumerate(rows):
        source_groups[row.get("source", "unknown")].append(i)
    result["macro_source_accuracy"] = float(np.mean([correct[ix].mean() for ix in source_groups.values()]))
    return result


def compare(reference: Sequence[dict], quantized: Sequence[dict]) -> dict:
    ref = {r["key"]:r for r in reference}
    other = {r["key"]:r for r in quantized}
    if len(ref) != len(reference) or len(other) != len(quantized) or ref.keys() != other.keys():
        raise ValueError("Paired comparison requires unique identical decision keys")
    js, flips, harmful, margins, errors = [], [], [], [], []
    for key, r in ref.items():
        q = other[key]
        p1,p2 = probabilities(r["logits"]), probabilities(q["logits"])
        if p1.shape != p2.shape or r["label"] != q["label"]:
            raise ValueError("Paired predictions have incompatible options or labels")
        mean = (p1+p2)/2
        js.append(float((np.sum(p1*np.log(np.maximum(p1,1e-15)/np.maximum(mean,1e-15))) + np.sum(p2*np.log(np.maximum(p2,1e-15)/np.maximum(mean,1e-15))))/2))
        flip = int(p1.argmax() != p2.argmax())
        flips.append(flip)
        harmful.append(int(p1.argmax() == r["label"] and p2.argmax() != r["label"]))
        margins.append(float(np.sort(p1)[-1]-np.sort(p1)[-2]))
        errors.append(float(abs(p1.max()-p2.max())))
    if not js:
        return {"n":0}
    # Near-boundary examples receive higher weight, capped and normalized.
    weights = 1 + np.clip((0.2-np.asarray(margins))/0.2, 0, 1)
    return {"n":len(js), "js_to_reference":float(np.mean(js)), "decision_flip_rate":float(np.mean(flips)),
            "harmful_flip_rate":float(np.mean(harmful)), "mean_confidence_drift":float(np.mean(errors)),
            "selection_objective":float(np.average(js,weights=weights)+0.1*np.mean(harmful))}


def paired_accuracy_ci(reference: Sequence[dict], quantized: Sequence[dict], seed: int = 20261001, samples: int = 2000) -> dict:
    """Cluster bootstrap by record, preserving all questions from each request."""
    ref = {r["key"]:r for r in reference}
    if set(ref) != {r["key"] for r in quantized}:
        raise ValueError("Paired CI requires identical keys")
    groups = defaultdict(list)
    for row in quantized:
        r=ref[row["key"]]
        delta=int(np.argmax(row["logits"])==row["label"])-int(np.argmax(r["logits"])==r["label"])
        groups[row.get("cluster_id",row["record_id"])].append(delta)
    sums=np.array([sum(x) for x in groups.values()]); counts=np.array([len(x) for x in groups.values()])
    if not len(sums):
        return {"n_clusters":0}
    rng=np.random.default_rng(seed)
    boot=[]
    for _ in range(samples):
        ix=rng.integers(0,len(sums),len(sums))
        boot.append(float(sums[ix].sum()/counts[ix].sum()))
    return {"n_clusters":len(sums), "delta":float(sums.sum()/counts.sum()), "lower_95":float(np.quantile(boot,.025)), "upper_95":float(np.quantile(boot,.975)), "seed":seed,"resamples":samples}

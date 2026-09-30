"""Export static paper figures from reviewed completed-run aggregates.

    python scripts/plot_results.py --input results/aggregate.json --output-dir docs/assets

Requires Matplotlib and NumPy. Uses stored paired intervals and actual selected
profiles; no model loading, method selection, fabricated missing observations,
or significance tests are performed. SVG/PDF exports and PNG previews are saved
with fixed metadata. Repeated calls in the same runtime are byte deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORDER = {"kev-0.8b": 0, "kev-4b": 1, "kev-9b": 2, "nanojev": 3, "laya": 4}
NAMES = {"kev-0.8b": "Kev-0.8B", "kev-4b": "Kev-4B", "kev-9b": "Kev-9B", "nanojev": "NanoJev", "laya": "Laya"}
BLUE, ORANGE, INK, GREY = "#245A81", "#B66B24", "#252525", "#737373"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "axes.edgecolor": "#8A8A8A", "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none", "svg.hashsalt": "S1Q-results-v1", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "savefig.transparent": False,
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_rank(run: dict) -> tuple:
    match = re.search(r"-v(\d+)", run["run"])
    split = "external_test" if run.get("external") else "test"
    n = run.get("evaluations", {}).get("native", {}).get(split, {}).get("raw", {}).get("n", 0)
    return (int(match.group(1)) if match else 0, "validation" in run["run"], n, run["run"])


def latest_runs(report: dict, *, external: bool = False) -> list[dict]:
    chosen = {}
    for run in report["runs"]:
        if bool(run.get("external")) != external:
            continue
        model = run["model"]
        if model not in chosen or latest_rank(run) > latest_rank(chosen[model]):
            chosen[model] = run
    return sorted(chosen.values(), key=lambda run: (ORDER.get(run["model"], 99), run["model"]))


def point(run: dict, split: str) -> dict | None:
    """Return a reviewed point and its actual asymmetric paired interval."""
    interval = run.get("paired_comparisons", {}).get(split, {}).get("selected_minus_matched_rtn", {})
    if interval.get("status") != "computed" or not run.get("matched_rtn"):
        return None
    values = [interval.get(key) for key in ("delta", "lower_95", "upper_95")]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError(f"Invalid paired interval: {run['run']} {split}")
    delta, lower, upper = values
    if lower > upper or any(abs(value) > 1 for value in values):
        raise ValueError(f"Invalid accuracy-difference bounds: {run['run']} {split}")
    selected = run["selection"]["selected"]["name"]
    raw = run["evaluations"][selected][split]["raw"]
    rtn = run["evaluations"][run["matched_rtn"]][split]["raw"]
    if raw["n"] != rtn["n"] or not math.isclose(delta, raw["accuracy"] - rtn["accuracy"], abs_tol=1e-12):
        raise ValueError(f"Paired delta/count differs from displayed metrics: {run['run']} {split}")
    return {"model": run["model"], "run": run["run"], "split": split,
            "selected": selected, "matched_rtn": run["matched_rtn"], "n": raw["n"],
            "n_clusters": interval["n_clusters"], "cluster_basis": interval.get("cluster_basis"),
            "delta_pp": delta * 100, "lower_95_pp": lower * 100, "upper_95_pp": upper * 100,
            "summary_sha256": run["summary_sha256"]}


def interval_axis(ax, *, limits: tuple[float, float], rows: int):
    ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(*limits)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_xlabel("S1Q − matched RTN accuracy (percentage points)", labelpad=8)


def common_limits(points: list[dict]) -> tuple[float, float]:
    if not points:
        return (-1.0, 1.0)
    low = min(0, *(item["lower_95_pp"] for item in points))
    high = max(0, *(item["upper_95_pp"] for item in points))
    padding = max((high - low) * 0.12, 0.5)
    return low - padding, high + padding


def draw_point(ax, item: dict, y: float, *, color=BLUE, marker="o", filled=True):
    lower, upper, delta = item["lower_95_pp"], item["upper_95_pp"], item["delta_pp"]
    ax.hlines(y, lower, upper, color=color, lw=1.35, zorder=2)
    ax.vlines([lower, upper], y - 0.055, y + 0.055, color=color, lw=1.1, zorder=2)
    ax.plot(delta, y, marker=marker, markersize=5.5, markeredgewidth=1.15,
            markeredgecolor=color, markerfacecolor=color if filled else "white", zorder=3)


def export(fig, *, output_dir: Path, name: str, title: str, description: str) -> dict:
    output = {}
    for suffix in ("svg", "pdf", "png"):
        path = output_dir / f"{name}.{suffix}"
        if suffix == "svg":
            metadata = {"Date": None, "Creator": "S1Q", "Title": title, "Description": description}
        elif suffix == "pdf":
            metadata = {"CreationDate": None, "ModDate": None, "Creator": "S1Q", "Producer": "Matplotlib", "Title": title, "Subject": description}
        else:
            metadata = {"Software": "S1Q", "Title": title, "Description": description}
        fig.savefig(path, format=suffix, dpi=220, bbox_inches="tight", pad_inches=0.16, metadata=metadata)
        if suffix == "svg":
            # Canonical text bytes keep Git newline conversion from invalidating
            # the figure manifest on Windows or Linux checkouts.
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        output[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    plt.close(fig)
    return output


def paired_figure(runs: list[dict], *, output_dir: Path, report: dict) -> dict | None:
    if not runs:
        return None
    main = [point(run, "test") for run in runs]
    secondary = [[item for item in (point(run, "transfer_test"), point(run, "ood")) if item is not None] for run in runs]
    points = [item for item in main if item is not None] + [item for items in secondary for item in items]
    if not points:
        return None
    limits = common_limits(points)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.45), gridspec_kw={"wspace": 0.68})
    for ax in axes:
        interval_axis(ax, limits=limits, rows=len(runs))
        ax.set_yticks(np.arange(len(runs)))
    axes[0].set_title("Main test", loc="left", pad=12)
    axes[1].set_title("Source transfer / native game OOD", loc="left", pad=12)
    labels_left, labels_right = [], []
    for i, (run, item, others) in enumerate(zip(runs, main, secondary)):
        display = NAMES.get(run["model"], run["model"])
        labels_left.append(f"{display}\n" + (f"n={item['n']}, G={item['n_clusters']}" if item else "paired result unavailable"))
        labels_right.append(" / ".join(f"{'OOD' if value['split']=='ood' else 'Transfer'}: n={value['n']}, G={value['n_clusters']}" for value in others) or "paired result unavailable")
        if item:
            draw_point(axes[0], item, i)
        else:
            axes[0].text(0, i, "unavailable", color=GREY, ha="center", va="center", fontsize=8)
        for j, other in enumerate(others):
            offset = (j - (len(others) - 1) / 2) * 0.18
            is_ood = other["split"] == "ood"
            draw_point(axes[1], other, i + offset, color=ORANGE if is_ood else BLUE,
                       marker="D" if is_ood else "o", filled=not is_ood)
        if not others:
            axes[1].text(0, i, "unavailable", color=GREY, ha="center", va="center", fontsize=8)
    axes[0].set_yticklabels(labels_left)
    axes[1].set_yticklabels(labels_right, fontsize=8)
    axes[1].legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=1.2, markersize=5, label="Source transfer"),
                            Line2D([], [], color=ORANGE, marker="D", markerfacecolor="white", lw=1.2, markersize=5, label="Native game OOD")],
                   frameon=False, loc="upper right", bbox_to_anchor=(1.02, -0.18), ncol=2, fontsize=8)
    fig.suptitle("Selected S1Q relative to the same-format RTN baseline", x=0.04, ha="left", y=1.04, fontsize=12)
    fig.text(0.04, -0.055, f"Paired 95% percentile cluster intervals; {report['bootstrap_samples']:,} resamples. n = questions; G = clusters.\n"
             "Exploratory search after pilot inspection. NanoJev targets are expert-action agreement; other cohorts use Kev source labels.", fontsize=8, color=GREY)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.87, bottom=0.12)
    files = export(fig, output_dir=output_dir, name="paired_accuracy", title="S1Q paired accuracy differences",
                   description="Selected S1Q minus matched RTN, main versus transfer/native game OOD. Paired95% cluster bootstrap intervals, exploratory post-pilot search.")
    return {"name": "paired_accuracy", "files": files, "chart_points": points,
            "axis_limits_pp": list(limits), "interpretation": "Exploratory paired changes; no significance annotations or pooled cross-model rate."}


def storage_figure(runs: list[dict], *, output_dir: Path) -> dict | None:
    points = []
    for run in runs:
        selected = run["selection"]["selected"]["name"]
        storage = run.get("methods", {}).get(selected, {}).get("storage", {})
        if not storage:
            continue
        ratio = storage.get("estimated_parameter_storage_ratio")
        native, packed = storage.get("native_parameters_bytes"), storage.get("estimated_complete_packed_parameters_bytes")
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in (ratio, native, packed)) or native <= 0 or packed < 0:
            raise ValueError(f"Invalid parameter storage estimate: {run['run']}")
        if not math.isclose(ratio, packed / native, rel_tol=1e-12):
            raise ValueError(f"Storage ratio does not reconcile: {run['run']}")
        points.append({"model": run["model"], "run": run["run"], "selected": selected,
                       "ratio": ratio, "quantized_parameter_fraction": storage["quantized_parameter_fraction"],
                       "native_parameters_bytes": native, "estimated_complete_packed_parameters_bytes": packed,
                       "summary_sha256": run["summary_sha256"]})
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(8.7, 4.0))
    y = np.arange(len(points))
    bars = ax.barh(y, [item["ratio"] for item in points], color=BLUE, height=0.48)
    ax.set_yticks(y, [NAMES.get(item["model"], item["model"]) for item in points])
    ax.invert_yaxis()
    limit = max(1.1, max(item["ratio"] for item in points) + 0.35)
    ax.set_xlim(0, limit)
    ax.set_xlabel("Estimated complete parameter bytes / native parameter bytes", labelpad=9)
    ax.axvline(1, color=INK, ls=(0, (3, 3)), lw=0.9)
    ax.text(1, -0.62, "native = 1", ha="center", va="center", fontsize=8, color=GREY)
    for bar, item in zip(bars, points):
        ax.text(bar.get_width() + 0.025, bar.get_y() + bar.get_height()/2,
                f"{item['ratio']:.3f}   ({item['quantized_parameter_fraction']*100:.1f}% of parameters quantized)",
                ha="left", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=8)
    fig.suptitle("Parameter storage estimates for the selected profiles", x=0.05, ha="left", y=1.02, fontsize=12)
    fig.text(0.05, -0.04, "Includes scales and retained embeddings, heads, biases and recurrent parameters.\n"
             "These estimates describe parameter storage; measured CUDA allocation is reported separately. Floating point execution remains.", fontsize=8, color=GREY)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.86)
    files = export(fig, output_dir=output_dir, name="parameter_storage", title="S1Q complete parameter storage ratios",
                   description="Selected mixed precision profiles. Native parameter bytes are the denominator; retained parameters and scale metadata included. Storage estimates, no speedup inference.")
    return {"name": "parameter_storage", "files": files, "chart_points": points,
            "interpretation": "Parameter estimate only; no measured memory or speedup inference."}


def external_figure(runs: list[dict], *, output_dir: Path, report: dict) -> dict | None:
    points = [item for run in runs if (item := point(run, "external_test")) is not None]
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(8.7, max(2.4, 1.9 + len(points)*0.42)))
    limits = common_limits(points)
    interval_axis(ax, limits=limits, rows=len(points))
    ax.set_yticks(np.arange(len(points)), [f"{NAMES.get(item['model'],item['model'])} (n={item['n']}, G={item['n_clusters']})" for item in points])
    for i, item in enumerate(points):
        draw_point(ax, item, i)
    ax.set_title("External public authored cohort: selected S1Q − matched RTN", loc="left", pad=14)
    fig.text(0.04, -0.1, f"Paired 95% cluster intervals; {report['bootstrap_samples']:,} resamples. Profiles frozen before external inference.\n"
             "Screened fstandhartinger/jevbench original+hard public tasks; hard labels have model-assisted authorship. No official leaderboard claim.", fontsize=8, color=GREY)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.77, bottom=0.2)
    files = export(fig, output_dir=output_dir, name="external_accuracy", title="S1Q external authored cohort paired accuracy differences",
                   description="Fixed profiles on screened public authored JevBench tasks. Completed external results only; no external fitting or official rank.")
    return {"name": "external_accuracy", "files": files, "chart_points": points, "axis_limits_pp": list(limits),
            "interpretation": "Fixed-profile external authored validation; no invented uncompleted model points."}


def render_assets_readme(manifest: dict) -> str:
    labels = {"paired_accuracy": "Paired accuracy differences: main versus source transfer/native game OOD",
              "parameter_storage": "Complete parameter storage estimates",
              "external_accuracy": "Fixed-profile external authored validation"}
    lines = ["# S1Q paper figures", "", "Generated by `scripts/plot_results.py` from `results/aggregate.json`. "
             "The exact aggregate hash, selected runs, displayed values and export hashes are in `figure-manifest.json`. "
             "SVG files retain editable text; PDF files embed the default sans serif font. PNG files are inspection previews.", ""]
    for figure in manifest["figures"]:
        name = figure["name"]
        lines.extend([f"## {labels[name]}", "", f"![{labels[name]}]({name}.png)", "",
                      f"[SVG]({name}.svg) · [PDF]({name}.pdf)", "", figure["interpretation"], ""])
    lines.extend(["The paired intervals are computed from saved paired predictions using saved cluster IDs. Older files without them use record-ID clusters, as recorded in the figure manifest. "
                  "Models and datasets have different target semantics and sample sizes; their rates are never pooled. "
                  "The expanded search followed inspected pilot results. Storage ratios include retained native parameters and are separate from measured CUDA allocation. "
                  "The runtime uses transient dequantization and floating point matrix multiplication; these figures imply no native integer speedup.", "",
                  "```sh", "python scripts/summarize_results.py", "python scripts/plot_results.py --input results/aggregate.json --output-dir docs/assets", "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/aggregate.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    source_payload = args.input.read_bytes()
    report = json.loads(source_payload.decode("utf-8"))
    if report.get("schema") != "s1q-results-aggregate.v1":
        parser.error("Unsupported aggregate schema; run scripts/summarize_results.py first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "s1q-paper-figures.v1", "aggregate_sha256": hashlib.sha256(source_payload).hexdigest(),
                "plotter_sha256": sha256(Path(__file__)), "matplotlib_version": matplotlib.__version__,
                "bootstrap_samples": report["bootstrap_samples"], "seed": report["seed"], "figures": []}
    ordinary, external = latest_runs(report), latest_runs(report, external=True)
    for figure in (paired_figure(ordinary, output_dir=args.output_dir, report=report),
                   storage_figure(ordinary, output_dir=args.output_dir),
                   external_figure(external, output_dir=args.output_dir, report=report)):
        if figure is not None:
            manifest["figures"].append(figure)
    (args.output_dir / "figure-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (args.output_dir / "README.md").write_text(render_assets_readme(manifest), encoding="utf-8", newline="\n")
    print(json.dumps({"figures": [item["name"] for item in manifest["figures"]], "aggregate_sha256": manifest["aggregate_sha256"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

"""Build transparent post hoc cases from the complete published W4A4 quartet.

These descriptive examples supplement the full matrix; no evaluation or interval
fitting occurs here. All 10 model and 22 source candidates remain inspectable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_component_analysis import COLORS, METHODS, MODELS, SOURCES, axis_style

LABELS = {"s1q-joint": "Joint", "s1q-ac": "AC", "s1q-margin": "Margin", "s1q": "S1Q"}
ACCURACY = dict(zip(METHODS, ("accuracy_joint_percent", "accuracy_ac_percent",
                            "accuracy_margin_percent", "accuracy_s1q_percent")))
RULE = ("Post hoc: separately among all 10 single-model aggregates and all 22 "
        "single-source aggregates, require S1Q to exceed all three ablations, "
        "then maximize S1Q minus max(Joint, AC, Margin); break exact ties by case_id.")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def title(row):
    if row.case_axis == "single_model":
        return "StartLux-Decision-4B" if row.case_id == "startlux-decision-4b" else row.case_id
    return "SST-5" if row.case_id == "sst5" else row.case_id


def analyze(data):
    cells = data.pivot(index=["model", "dataset_id"], columns="method", values="accuracy") * 100
    cells = cells.loc[:, list(METHODS)]
    metadata = data.drop_duplicates(["model", "dataset_id"]).set_index(["model", "dataset_id"])
    for m, label in (("s1q-joint", "joint"), ("s1q-ac", "ac"), ("s1q-margin", "margin")):
        cells[f"gain_vs_{label}_pp"] = cells.s1q - cells[m]
    cells["best_ablation_accuracy_percent"] = cells[list(METHODS[:-1])].max(axis=1)
    cells["gap_to_best_ablation_pp"] = cells.s1q - cells.best_ablation_accuracy_percent
    cells["best_ablation_method"] = cells[list(METHODS[:-1])].idxmax(axis=1).map(LABELS)
    cells["n"] = metadata.n
    cells["n_clusters"] = metadata.n_clusters
    records = []
    for axis, level, ids in (("single_model", "model", MODELS), ("single_source", "dataset_id", SOURCES)):
        for case_id in ids:
            subset = cells.xs(case_id, level=level)
            means = subset[list(METHODS)].mean()
            best = means[list(METHODS[:-1])].max()
            row = dict(case_axis=axis, case_id=case_id,
                       aggregation="unweighted mean of all 22 source accuracies" if axis == "single_model" else "unweighted mean of all 10 model accuracies",
                       model_count=1 if axis == "single_model" else 10,
                       source_count=22 if axis == "single_model" else 1,
                       complete_cells=len(subset),
                       decision_evaluations_per_method=int(subset.n.sum()),
                       sum_reported_clusters_per_method=int(subset.n_clusters.sum()),
                       positive_cells_vs_ac=int((subset.gain_vs_ac_pp > 1e-10).sum()),
                       negative_cells_vs_ac=int((subset.gain_vs_ac_pp < -1e-10).sum()),
                       positive_cells_vs_best_ablation=int((subset.gap_to_best_ablation_pp > 1e-10).sum()),
                       negative_cells_vs_best_ablation=int((subset.gap_to_best_ablation_pp < -1e-10).sum()),
                       **{ACCURACY[m]: float(means[m]) for m in METHODS},
                       **{f"gain_vs_{name}_pp": float(means.s1q - means[m]) for m, name in zip(METHODS[:-1], ("joint", "ac", "margin"))},
                       best_ablation_accuracy_percent=float(best),
                       gap_to_best_ablation_pp=float(means.s1q - best),
                       s1q_strict_best=bool(means.s1q > best + 1e-10),
                       best_ablation_method=LABELS[means[list(METHODS[:-1])].idxmax()])
            records.append(row)
    candidates = pd.DataFrame(records)
    for metric, rank_col in (("gap_to_best_ablation_pp", "rank_gap_to_best_ablation"),
                             ("gain_vs_ac_pp", "rank_gain_vs_ac"), ("gain_vs_joint_pp", "rank_gain_vs_joint")):
        order = candidates.sort_values(["case_axis", metric, "case_id"], ascending=[True, False, True])
        candidates.loc[order.index, rank_col] = order.groupby("case_axis").cumcount() + 1
        candidates[rank_col] = candidates[rank_col].astype(int)
    candidates["selected_descriptive_case"] = candidates.s1q_strict_best & (candidates.rank_gap_to_best_ablation == 1)
    chosen = candidates[candidates.selected_descriptive_case].sort_values("case_axis")
    if len(chosen) != 2:
        raise ValueError("Both axes must have a strictly positive complete-scope example")
    return cells, candidates.sort_values(["case_axis", "rank_gap_to_best_ablation"]), chosen


def figures(root, selected):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for i, (ax, row) in enumerate(zip(axes, selected.itertuples())):
        values = [getattr(row, ACCURACY[m]) for m in METHODS]
        yy = np.arange(4)[::-1]
        ax.barh(yy, values, height=.59, color=[COLORS[m] for m in METHODS])
        ax.set_yticks(yy, [LABELS[m] for m in METHODS], fontsize=11)
        ax.set_xlim(0, 80)
        ax.set_ylim(-.6, 3.6)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_xlabel("Decision accuracy (%)", fontsize=11)
        scope = "All 22 sources; one model" if row.case_axis == "single_model" else "All 10 models; one source"
        ax.set_title(f"({chr(97+i)}) {title(row)}\n{scope}", fontsize=12, loc="left", pad=17)
        axis_style(ax)
        for y, value, method in zip(yy, values, METHODS):
            ax.text(value + .8, y, f"{value:.2f}", va="center", fontsize=11,
                    fontweight="bold" if method == "s1q" else "normal", color=COLORS[method])
        ax.text(.01, 1.02, f"S1Q vs best ablation: +{row.gap_to_best_ablation_pp:.2f} pp",
                transform=ax.transAxes, fontsize=10, color=COLORS["s1q"], fontweight="bold")
    fig.text(.02, .045, "W4A4 · Post hoc illustrative cases; selected separately from 10 model / 22 source aggregates.", fontsize=9, color="#4B5661")
    fig.text(.02, .005, "All four configurations shown. No selected-case confidence intervals; full rankings and regressions are retained.", fontsize=9, color="#4B5661")
    fig.subplots_adjust(left=.085, right=.98, top=.73, bottom=.21, wspace=.30)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(root / f"case_studies.{suffix}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    tikz = [r"\begin{tikzpicture}"]
    for i, row in enumerate(selected.itertuples()):
        position = "" if i == 0 else r"at={(.54\textwidth,0)},anchor=south west,"
        tikz += [r"\begin{axis}[" + position + r"width=.44\textwidth,height=5.6cm,xmin=0,xmax=80,",
                 r"ymin=-.6,ymax=3.6,xtick={0,20,40,60,80},ytick={0,1,2,3},yticklabels={S1Q,Margin,AC,Joint},",
                 r"xlabel={Decision accuracy (\%)},bar width=10pt,axis lines*=left,xmajorgrids,grid style={gray!15},",
                 r"tick label style={font=\scriptsize},label style={font=\footnotesize},clip=false,",
                 rf"title={{({chr(97+i)}) {title(row)}}},title style={{font=\small}}]"]
        for j, method in enumerate(METHODS):
            y = 3-j
            value = getattr(row, ACCURACY[method])
            color = f"casecolor{i}{j}"
            tikz += [rf"\definecolor{{{color}}}{{HTML}}{{{COLORS[method][1:]}}}",
                     rf"\addplot+[xbar,mark=none,bar shift=0pt,fill={color},draw=none,forget plot] coordinates {{({value:.10f},{y})}};",
                     rf"\node[anchor=west,font=\scriptsize] at (axis cs:{value+.8:.10f},{y}) {{{value:.2f}}};"]
        scope = "All 22 sources; one model" if row.case_axis == "single_model" else "All 10 models; one source"
        tikz += [rf"\node[anchor=north,font=\scriptsize] at (rel axis cs:.5,-.23) {{{scope}}};",
                 rf"\node[anchor=north,font=\scriptsize] at (rel axis cs:.5,-.33) {{S1Q vs best ablation: $+{row.gap_to_best_ablation_pp:.2f}$ pp}};",
                 r"\end{axis}"]
    tikz += [r"\end{tikzpicture}"]
    (root / "case_studies.tikz.tex").write_text("\n".join(tikz)+"\n", encoding="utf-8")


def table_tex(rows, scope=False):
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Case / complete scope & Joint & AC & Margin & S1Q & $\Delta$ best (pp) \\", r"\midrule"]
    for row in rows.itertuples():
        name = title(row) if scope else row.case_id.replace("_", r"\_")
        if scope:
            name += " (22 sources)" if row.case_axis == "single_model" else " (10 models)"
        vals = [getattr(row, ACCURACY[m]) for m in METHODS]
        best = max(round(v, 2) for v in vals)
        printed = [rf"\textbf{{{v:.2f}}}" if round(v, 2) == best else f"{v:.2f}" for v in vals]
        lines.append(name + " & " + " & ".join(printed) + rf" & {row.gap_to_best_ablation_pp:+.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, root = args.source_root.resolve() / "accuracy_by_dataset.csv", args.output_dir.resolve()
    if root.exists():
        raise FileExistsError("Fresh output directory required")
    data = pd.read_csv(source)
    data = data[(data.comparison_group == "all12") & data.method.isin(METHODS)].copy()
    if (set(data.model) != set(MODELS) or set(data.dataset_id) != set(SOURCES)
            or len(data) != 880 or data.duplicated(["model", "dataset_id", "method"]).any()
            or set(data.precision) != {"W4A4"} or not data.accuracy.between(0, 1).all()):
        raise ValueError("Expected the complete 10 x 22 x 4 W4A4 matrix")
    for field, expected in (("suite_id", "expanded-v3-w4a4-22source-20261007"),
                            ("experiment_batch_id", "expanded-20261007-v3")):
        if set(data[field]) != {expected}:
            raise ValueError(f"Unexpected {field}")
    for field in ("n", "n_clusters", "source_category", "label_kind"):
        if data.groupby(["model", "dataset_id"])[field].nunique().max() != 1:
            raise ValueError(f"Mismatched component metadata: {field}")
    for field in ("n", "n_clusters"):
        if data.groupby("dataset_id")[field].nunique().max() != 1:
            raise ValueError(f"Cross-model source counts differ: {field}")
    cells, candidates, selected = analyze(data)
    root.mkdir(parents=True)
    candidates.to_csv(root / "candidate_rankings.csv", index=False)
    selected.to_csv(root / "selected_cases.csv", index=False)
    for row in selected.itertuples():
        level = "model" if row.case_axis == "single_model" else "dataset_id"
        cells.xs(row.case_id, level=level).reset_index().to_csv(root / f"{row.case_axis}_complete_breakdown.csv", index=False)
    figures(root, selected)
    (root / "case_studies_table.tex").write_text(table_tex(selected, scope=True), encoding="utf-8")
    for axis in ("single_model", "single_source"):
        (root / f"{axis}_candidate_ranking.tex").write_text(table_tex(candidates[candidates.case_axis == axis]), encoding="utf-8")
    markdown = ["# Post hoc illustrative component cases", "", RULE, "",
        "These examples supplement the full **10 models × 22 sources W4A4** quartet; they do not replace it. "
        "The rule was chosen after viewing the results. There are no selected-case confidence intervals or significance claims. "
        "All four configurations, all 32 candidate rankings and the selected cases' negative cells are retained. No new model evaluation was performed.", "",
        "| Case / complete scope | Joint | AC | Margin | S1Q | S1Q − best ablation (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in selected.itertuples():
        scope = "all 22 sources" if row.case_axis == "single_model" else "all 10 models"
        vals = [getattr(row, ACCURACY[m]) for m in METHODS]
        markdown.append(f"| {title(row)}, {scope} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | **{vals[3]:.2f}** | **+{row.gap_to_best_ablation_pp:.2f}** |")
    markdown += ["", "![Complete four-configuration post hoc examples](case_studies.png)", "",
        "[PDF](case_studies.pdf) · [SVG](case_studies.svg) · [Standalone TikZ](case_studies.tikz.tex) · "
        "[Selected values](selected_cases.csv) · [All 32 candidate ranks](candidate_rankings.csv) · "
        "[Single-model complete breakdown](single_model_complete_breakdown.csv) · "
        "[Single-source complete breakdown](single_source_complete_breakdown.csv) · [Manifest](figure_manifest.json)", "",
        "Accuracy is a percentage; differences are percentage points. Model cases average all 22 source accuracies equally; "
        "source cases average all 10 model accuracies equally. No source-size weighting or isolated model-by-source extreme-cell selection is used.", "",
        "StartLux-Decision-4B has 2,871 decisions per method. Its published source cluster counts sum to 2,469, "
        "which is not a globally unique independent-cluster count. SST-5 reuses the same 128 source inputs across all 10 models; "
        "1,280 model-decision evaluations are not 1,280 independent examples.", "",
        "Joint and Margin evaluate 22 candidates per layer; AC and S1Q evaluate 44. The S1Q-versus-AC comparison matches candidate count, "
        "but S1Q also incurs margin backpropagation cost. This is not an equal-total-compute causal isolation or a proof of component synergy. "
        "These are floating-point QDQ accuracy measurements, with no new packed-checkpoint or integer-kernel speedup claim.", "",
        "The gain pattern is descriptive. These aggregates do not establish why a particular source benefits, nor a universal winner. "
        "See the [whole-matrix component analysis](../components-20261008/README.md) for primary evidence.", "",
        "Reproduce from repository root (choose a fresh output directory):", "", "```sh",
        "python scripts/build_component_cases.py --source-root results/benchmarks/expanded-20261007 --output-dir work/component-cases", "```", "",
        "## Complete candidate rankings", ""]
    for axis in ("single_model", "single_source"):
        markdown += ["### " + ("All 10 single-model candidates" if axis == "single_model" else "All 22 single-source candidates"), "",
                     "| Rank | Candidate | Joint | AC | Margin | S1Q | Gap to best (pp) |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in candidates[candidates.case_axis == axis].itertuples():
            vals = [getattr(row, ACCURACY[m]) for m in METHODS]
            best = max(round(v, 2) for v in vals)
            fmt = [f"**{v:.2f}**" if round(v, 2) == best else f"{v:.2f}" for v in vals]
            markdown.append(f"| {row.rank_gap_to_best_ablation} | {row.case_id} | " + " | ".join(fmt) + f" | {row.gap_to_best_ablation_pp:+.2f} |")
        markdown.append("")
    (root / "README.md").write_text("\n".join(markdown).rstrip()+"\n", encoding="utf-8")
    metadata = dict(status="complete_descriptive_post_hoc_cases", selection_rule=RULE,
        no_case_confidence_intervals_or_significance_claims=True,
        primary_full_matrix_is_not_replaced=True, candidate_counts=dict(single_model=10, single_source=22),
        component_cells=880, precision="W4A4", methods=list(METHODS),
        input_sha256={"expanded-20261007/accuracy_by_dataset.csv": sha(source)},
        author_script_sha256=sha(Path(__file__)),
        selected=selected.to_dict(orient="records"),
        files={p.name: sha(p) for p in sorted(root.iterdir())})
    (root / "figure_manifest.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(status="passed", candidates=len(candidates), selected=selected[["case_axis", "case_id", "gap_to_best_ablation_pp"]].to_dict(orient="records")), indent=2))


if __name__ == "__main__":
    main()

"""Render fixed-scope S1Q component evidence from published aggregates.

No model evaluation, subset selection, or confidence-interval fitting occurs here.
The complete ten-model, twenty-two-source quartet is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

METHODS = ("s1q-joint", "s1q-ac", "s1q-margin", "s1q")
LABELS = {"s1q-joint": "Joint", "s1q-ac": "+AC", "s1q-margin": "+Margin", "s1q": "S1Q"}
MODELS = ("kev-0.8b", "kev-4b", "kev-9b", "laya", "intern-decision-0.8b",
          "intern-decision-2b", "intern-decision-4b", "startlux-decision-0.8b",
          "startlux-decision-2b", "startlux-decision-4b")
MODEL_LABELS = ("Kev 0.8B", "Kev 4B", "Kev 9B", "Laya", "Intern 0.8B", "Intern 2B",
                "Intern 4B", "StartLux 0.8B", "StartLux 2B", "StartLux 4B")
SOURCES = ("agnews", "amazon", "arc", "boolq", "csqa", "emotion", "imdb", "jevbench",
           "mmlu", "mnli", "openbookqa", "paws", "qnli", "sciq", "semif", "sst5",
           "toolace", "trec", "tweet-offensive", "wanli", "wildjailbreak", "yelp")
COLORS = {"s1q-joint": "#7A8792", "s1q-ac": "#477DA6", "s1q-margin": "#85ACC8", "s1q": "#B87814"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, root, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(root / f"{name}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def axis_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#BCC4CB")
    ax.spines["bottom"].set_color("#BCC4CB")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E6E9EC", lw=.7)


def write_tikz(root, means, model, ci, rankings, deltas):
    # Self-contained alternatives for the built-in standalone LaTeX compiler.
    overview = [r"\begin{tikzpicture}",
        r"\begin{axis}[width=.45\textwidth,height=6.6cm,xbar,xmin=0,xmax=62,",
        r"ytick={0,1,2,3},yticklabels={Joint,+AC,+Margin,S1Q},xlabel={Accuracy (\%)},",
        r"bar width=11pt,ymin=-.6,ymax=3.6,axis lines*=left,xmajorgrids,grid style={gray!15},",
        r"tick label style={font=\footnotesize},label style={font=\footnotesize},title={(a) All 10 models, 22 sources},title style={font=\small}]"]
    for j, method in enumerate(METHODS):
        color = "orange!75!black" if method == "s1q" else "blue!45!gray"
        point, lo, hi = [float(ci.loc[method, k]) for k in ("accuracy_percent", "lower_95_percent", "upper_95_percent")]
        overview += [rf"\addplot+[xbar,bar shift=0pt,fill={color},draw=none,forget plot] coordinates {{({point:.8f},{j})}};",
                     rf"\addplot+[sharp plot,black,mark=none,forget plot] coordinates {{({lo:.8f},{j}) ({hi:.8f},{j})}};",
                     rf"\node[anchor=west,font=\scriptsize] at (axis cs:{hi+1:.8f},{j}) {{{point:.2f}}};"]
    overview += [r"\end{axis}", r"\begin{axis}[at={(.54\textwidth,0)},anchor=south west,width=.45\textwidth,height=6.6cm,",
                 r"xmin=-3,xmax=10,ymin=-.6,ymax=9.6,ytick={0,1,2,3,4,5,6,7,8,9},",
                 "yticklabels={" + ",".join(MODEL_LABELS) + "},",
                 r"xlabel={S1Q $-$ Joint (pp)},axis lines*=left,clip=false,xmajorgrids,grid style={gray!15},",
                 r"tick label style={font=\scriptsize},label style={font=\footnotesize},title={(b) Every model; descriptive gains},title style={font=\small}]",
                 r"\addplot[gray,dashed] coordinates {(0,-.5) (0,9.5)};"]
    for j, name in enumerate(MODELS):
        val = float(model.loc[name, "s1q"] - model.loc[name, "s1q-joint"])
        overview += [rf"\addplot[orange!75!black,only marks,mark=*] coordinates {{({val:.8f},{j})}};",
                     rf"\node[anchor=west,font=\scriptsize] at (axis cs:{val+.15:.8f},{j}) {{{val:+.2f}}};"]
    overview += [r"\end{axis}", r"\end{tikzpicture}"]
    (root / "ablation_overview.tikz.tex").write_text("\n".join(overview)+"\n", encoding="utf-8")
    lines = [r"\begin{tikzpicture}", r"\begin{axis}[width=.96\textwidth,height=8cm,xbar,xmin=0,xmax=86,",
             "ytick={" + ",".join(map(str,range(13))) + "},",
             "yticklabels={" + ",".join(str(x).replace("*",r"$^*$").replace("S1Q (current)","S1Q") for x in rankings.method_label) + "},",
             r"xlabel={Accuracy (\%)},bar width=7pt,ymin=-.6,ymax=12.6,axis lines*=left,",
             r"xmajorgrids,grid style={gray!15},tick label style={font=\scriptsize},label style={font=\footnotesize}]"]
    for j, row in rankings.iterrows():
        method = row.method
        p, lo, hi = [float(ci.loc[method, k]) for k in ("accuracy_percent", "lower_95_percent", "upper_95_percent")]
        col = "orange!75!black" if method == "s1q" else ("gray!55" if method == "native" else "blue!45!gray")
        lines += [rf"\addplot+[xbar,bar shift=0pt,fill={col},draw=none,forget plot] coordinates {{({p:.8f},{j})}};",
                  rf"\addplot+[sharp plot,black,mark=none,forget plot] coordinates {{({lo:.8f},{j}) ({hi:.8f},{j})}};",
                  rf"\node[anchor=west,font=\scriptsize] at (axis cs:{hi+1:.8f},{j}) {{{p:.2f}}};"]
    lines += [r"\end{axis}",r"\end{tikzpicture}"]
    (root / "overall_comparison.tikz.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    cmap = LinearSegmentedColormap.from_list("signed",["#225F91","#FAFBFC","#BD7918"])
    limit = max(1., float(np.abs(deltas.values).max()))
    heat = [r"\begin{tikzpicture}[x=.39cm,y=.48cm]"]
    for y, name in enumerate(MODELS):
        heat += [rf"\node[anchor=east,font=\scriptsize] at (-.1,{y+.5}) {{{MODEL_LABELS[y]}}};"]
        for x, source in enumerate(SOURCES):
            val = float(deltas.loc[name, source]); rgb = cmap((val+limit)/(2*limit))[:3]
            col = f"cell{x}_{y}"
            printed = "0" if round(val) == 0 else f"{val:+.0f}"
            heat += [rf"\definecolor{{{col}}}{{rgb}}{{{rgb[0]:.5f},{rgb[1]:.5f},{rgb[2]:.5f}}}",
                     rf"\fill[{col}] ({x},{y}) rectangle ({x+1},{y+1});",
                     rf"\node[font=\tiny,text={'white' if abs(val)>.6*limit else 'black'}] at ({x+.5},{y+.5}) {{{printed}}};"]
    for x, name in enumerate(SOURCES):
        heat += [rf"\node[anchor=east,rotate=55,font=\tiny] at ({x+.5},-.2) {{{name}}};"]
    heat += [r"\node[anchor=west,font=\scriptsize] at (0,10.4) {S1Q $-$ Joint accuracy (pp); all 220 cells, symmetric color scale};",r"\end{tikzpicture}"]
    (root / "source_robustness.tikz.tex").write_text("\n".join(heat)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    src, root = args.source_root.resolve(), args.output_dir.resolve()
    if root.exists():
        raise FileExistsError("Fresh output directory required")
    files = {k:src/k for k in ("accuracy_by_dataset.csv","model_scores.csv","rankings.csv",
              "accuracy-uncertainty/method-accuracy-ci.csv","accuracy-uncertainty/pairwise-accuracy-ci.csv")}
    data = pd.read_csv(files["accuracy_by_dataset.csv"])
    data = data[(data.comparison_group=="all12") & data.method.isin(METHODS)].copy()
    assert set(data.model)==set(MODELS) and set(data.dataset_id)==set(SOURCES)
    assert len(data)==880 and not data.duplicated(["model","dataset_id","method"]).any()
    assert set(data.precision)=={"W4A4"} and data.accuracy.notna().all()
    model = data.groupby(["model","method"]).accuracy.mean().unstack()*100
    means = model.mean()
    rank = pd.read_csv(files["rankings.csv"])
    rank = rank[(rank.comparison_group=="all12") & (rank.family=="All")].sort_values("rank_accuracy")
    assert len(rank)==12
    for method in METHODS:
        assert abs(float(rank[rank.method==method].accuracy_percent.iloc[0])-means[method])<1e-9
    model_original = pd.read_csv(files["model_scores.csv"])
    model_original = model_original[(model_original.comparison_group=="all12") & model_original.method.isin(METHODS)]
    assert len(model_original)==40
    for row in model_original.itertuples():
        assert abs(model.loc[row.model,row.method]-row.accuracy*100)<1e-9
    ci = pd.read_csv(files["accuracy-uncertainty/method-accuracy-ci.csv"]).set_index("method")
    pairs = pd.read_csv(files["accuracy-uncertainty/pairwise-accuracy-ci.csv"])
    paired = []
    for reference in METHODS[:-1]:
        matched = pairs[(pairs.method=="s1q") & (pairs.reference==reference)]
        if len(matched)==1:
            row=matched.iloc[0]; values=[row.difference_pp,row.lower_95_pp,row.upper_95_pp]
        else:
            matched=pairs[(pairs.method==reference)&(pairs.reference=="s1q")]
            assert len(matched)==1
            row=matched.iloc[0]; values=[-row.difference_pp,-row.upper_95_pp,-row.lower_95_pp]
        assert abs(values[0]-(means["s1q"]-means[reference]))<1e-9
        paired.append(dict(method="s1q",reference=reference,difference_pp=values[0],lower_95_pp=values[1],upper_95_pp=values[2]))
    root.mkdir(parents=True)
    data.to_csv(root/"component_by_source.csv",index=False)
    records=[]
    for name in (*MODELS,"All"):
        values=means if name=="All" else model.loc[name]
        for method in METHODS:
            records.append(dict(model=name,method=method,margin=method in ("s1q-margin","s1q"),
                         activation_compensation=method in ("s1q-ac","s1q"),
                         candidates_per_layer=44 if method in ("s1q-ac","s1q") else 22,
                         accuracy_percent=float(values[method]),delta_joint_pp=float(values[method]-values["s1q-joint"]),
                         best_displayed=format(values[method],".2f")==format(max(values),".2f")))
    pd.DataFrame(records).to_csv(root/"component_accuracy.csv",index=False)
    pd.DataFrame(paired).to_csv(root/"component_paired_ci.csv",index=False)
    model_out=model.reindex(MODELS).copy()
    model_out["s1q_minus_joint_pp"]=model_out["s1q"]-model_out["s1q-joint"]
    model_out["interaction_pp"]=model_out["s1q"]-model_out["s1q-ac"]-model_out["s1q-margin"]+model_out["s1q-joint"]
    model_out.to_csv(root/"component_model_summary.csv",index_label="model")
    notes=["# Complete S1Q component analysis", "", "W4A4; all 10 fixed models and all 22 source suites. Sources are averaged equally within models, then models equally. No favorable model or source subset is selected.", "",
           "| Model | Configuration | Margin | AC | Candidates/layer | Accuracy % | Gain vs Joint (pp) |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in records:
        value=f"{row['accuracy_percent']:.2f}"
        if row["best_displayed"]: value="**"+value+"**"
        notes.append(f"| {row['model']} | {LABELS[row['method']]} | {int(row['margin'])} | {int(row['activation_compensation'])} | {row['candidates_per_layer']} | {value} | {row['delta_joint_pp']:+.2f} |")
    notes += ["", "Bold denotes the best displayed value within a model, including rounding ties; it is not a significance claim. All configurations share W/A reconstruction, layer scope and 128 calibration requests with 128 token rows/layer (64 fit, 64 selection). Enabling AC adds corrected candidates: 44 versus 22. This is a component/recipe comparison, not an equal-compute causal isolation.", "",
              "Conditional paired 95% intervals are reused unchanged from the frozen shared-cluster bootstrap (2,000 draws, seed 20261007). Models and sources are fixed; no multiple-comparison adjustment. Model gains and source cells have no newly computed intervals. The interaction is a descriptive finite difference, not a synergy test.", "",
              f"S1Q exceeds Joint on {int((model_out.s1q_minus_joint_pp>0).sum())}/10 model averages; all negative outcomes are retained. S1Q-AC has better overall NLL. External quantizers are repository adaptations; the current study uses floating-point QDQ and makes no integer-kernel speedup claim.", "",
              "![Complete component comparison](ablation_overview.png)", "", "![All models and sources](source_robustness.png)", "", "![All twelve implementations](overall_comparison.png)"]
    (root/"README.md").write_text("\n".join(notes)+"\n",encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,"axes.labelsize":10,"svg.fonttype":"none","pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(12.6,5.6),gridspec_kw={"width_ratios":[1,1.25]})
    ax=axes[0]
    for y,method in enumerate(METHODS):
        p=means[method]; lo,hi=ci.loc[method,["lower_95_percent","upper_95_percent"]]
        ax.barh(y,p,height=.6,color=COLORS[method])
        ax.errorbar(p,y,xerr=[[p-lo],[hi-p]],fmt="none",ecolor="#273746",capsize=3,lw=1)
        ax.text(hi+1,y,f"{p:.2f}%",va="center",weight="bold" if method=="s1q" else "normal")
    ax.set(yticks=range(4),yticklabels=[LABELS[m] for m in METHODS],xlim=(0,63),xlabel="Accuracy (%)",title="(a) Complete aggregate with conditional 95% intervals")
    ax.invert_yaxis(); axis_style(ax)
    ax=axes[1]
    for method,offset,mark in (("s1q-ac",-.19,"s"),("s1q-margin",0,"^"),("s1q",.19,"o")):
        values=model.reindex(MODELS)[method]-model.reindex(MODELS)["s1q-joint"]
        ax.scatter(values,np.arange(10)+offset,color=COLORS[method],marker=mark,s=34,label=LABELS[method])
        if method=="s1q":
            for y,val in enumerate(values):
                ax.text(val+(.15 if val>=0 else -.15),y+.19,f"{val:+.2f}",ha="left" if val>=0 else "right",va="center",fontsize=8)
    ax.axvline(0,color="#56616A",lw=1,ls="--")
    ax.set(yticks=range(10),yticklabels=MODEL_LABELS,xlim=(-5,11),xlabel="Gain over Joint (percentage points)",title="(b) All models; gains are descriptive")
    ax.set_ylim(10.55,-.8); axis_style(ax); ax.legend(frameon=False,ncol=3,loc="lower right",fontsize=9)
    fig.suptitle("S1Q component analysis",x=.07,ha="left",fontsize=17,weight="bold")
    fig.text(.07,.9,"W4A4 | 10 models x 22 sources | equal source/model weights | all outcomes retained",fontsize=10,color="#53616D")
    fig.text(.07,.015,"AC: bounded activation compensation. Margin: native decision-margin token sampling. AC adds candidates: 44 vs 22.",fontsize=9,color="#53616D")
    fig.subplots_adjust(top=.81,bottom=.15,left=.08,right=.985,wspace=.55)
    save(fig,root,"ablation_overview")
    piv=data.pivot(index=["model","dataset_id"],columns="method",values="accuracy")
    deltas=((piv["s1q"]-piv["s1q-joint"])*100).unstack().reindex(index=MODELS,columns=SOURCES)
    deltas.to_csv(root/"source_gain_matrix.csv",index_label="model")
    cmap=LinearSegmentedColormap.from_list("signed",["#225F91","#FAFBFC","#BD7918"])
    limit=max(1,float(np.abs(deltas.values).max()))
    fig,ax=plt.subplots(figsize=(15.2,6.2))
    im=ax.imshow(deltas.values,cmap=cmap,vmin=-limit,vmax=limit,aspect="auto")
    ax.set(xticks=range(22),xticklabels=SOURCES,yticks=range(10),yticklabels=MODEL_LABELS)
    plt.setp(ax.get_xticklabels(),rotation=50,ha="right",rotation_mode="anchor",fontsize=9)
    for y in range(10):
        for x in range(22):
            val=deltas.iloc[y,x]
            printed="0" if round(val)==0 else f"{val:+.0f}"
            ax.text(x,y,printed,ha="center",va="center",fontsize=8,color="white" if abs(val)>.6*limit else "#24313B")
    for spine in ax.spines.values(): spine.set_visible(False)
    colorbar=fig.colorbar(im,ax=ax,pad=.015,shrink=.86)
    colorbar.set_label("S1Q - Joint accuracy (pp)")
    fig.suptitle("Complete source robustness",x=.09,ha="left",fontsize=17,weight="bold")
    fig.text(.09,.89,"W4A4 | every model/source cell | signed gains; symmetric color scale",fontsize=10,color="#53616D")
    fig.text(.09,.018,"220 cells; values rounded to whole pp for readability. Cells are descriptive, with no source-level significance claim.",fontsize=9,color="#53616D")
    fig.subplots_adjust(top=.83,bottom=.25,left=.1,right=.96)
    save(fig,root,"source_robustness")
    ranked=pd.concat([pd.DataFrame([dict(method="native",method_label="Native reference",accuracy_percent=float(ci.loc["native","accuracy_percent"]))]),rank],ignore_index=True)
    fig,ax=plt.subplots(figsize=(10.6,7.4))
    for y,row in ranked.iterrows():
        method=row.method;p=float(ci.loc[method,"accuracy_percent"]);lo,hi=ci.loc[method,["lower_95_percent","upper_95_percent"]]
        ax.barh(y,p,height=.62,color=COLORS.get(method,"#CCD9E3") if method!="native" else "#87949E")
        ax.errorbar(p,y,xerr=[[p-lo],[hi-p]],fmt="none",ecolor="#253540",capsize=2,lw=.9)
        ax.text(hi+1,y,f"{p:.2f}%",va="center",fontsize=9,weight="bold" if method=="s1q" else "normal")
    ax.set(yticks=range(13),yticklabels=ranked.method_label.str.replace("S1Q (current)","S1Q",regex=False),xlim=(0,86),xlabel="Accuracy (%)")
    ax.invert_yaxis();axis_style(ax)
    fig.suptitle("Complete W4A4 comparison",x=.06,ha="left",fontsize=17,weight="bold")
    fig.text(.06,.905,"10 models x 22 sources | all 12 quantized implementations | Native is a reference",fontsize=10,color="#53616D")
    fig.text(.06,.035,"Intervals: conditional shared-cluster bootstrap, 2,000 draws. * Repository adaptations; not official reproductions.\nAccuracy uses QDQ. Remaining Native-to-S1Q accuracy gap: 22.27 pp. New OmniQuant/AdaRound/HQQ matrix is pending.",fontsize=9,color="#53616D")
    fig.subplots_adjust(top=.84,bottom=.14,left=.23,right=.96)
    save(fig,root,"overall_comparison")
    write_tikz(root,means,model,ci,ranked,deltas)
    manifest={"schema":"s1q.complete-component-figures.v1","scope":"expanded-v3-w4a4-22source-20261007",
              "precision":"W4A4","models":list(MODELS),"sources":list(SOURCES),"methods":list(METHODS),
              "selection":"Complete common-coverage scope; no favorable model/source subset",
              "aggregation":"Sources equal within model, then models equal", "source_cells":220,"component_cells":880,
              "candidate_counts":{"s1q-joint":22,"s1q-ac":44,"s1q-margin":22,"s1q":44},
              "intervals":"Existing conditional shared-cluster bootstrap; 2000 draws; seed 20261007; fixed models/sources; no multiplicity adjustment",
              "s1q_vs_joint_model_wins":int((model_out.s1q_minus_joint_pp>0).sum()),
              "inputs":{k:sha(v) for k,v in files.items()},"script_sha256":sha(Path(__file__)),
              "outputs":{p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()}}
    (root/"figure_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"complete","component_cells":880,"all_source_cells":220,"model_wins":manifest["s1q_vs_joint_model_wins"],"accuracy_percent":means.to_dict(),"outputs":len(manifest["outputs"])},allow_nan=False))


if __name__=="__main__":
    main()

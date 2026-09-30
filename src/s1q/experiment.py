"""Development-selected, paired evaluation. No final-test tuning."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import torch

from .data import assert_disjoint, load_records
from .metrics import compare, fit_temperatures, label_index, metrics, paired_accuracy_ci
from .models import UnsupportedRecord, load_model
from .quantization import CalibrationCollector, quantize_model


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n" for row in rows),encoding="utf-8")


def synchronize(adapter):
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)


@torch.inference_mode()
def predict(adapter, records, *, allow_rejection=False):
    rows, accepted, rejected, timings = [], [], [], []
    for index, record in enumerate(records):
        rid=str(record.get("id",record.get("_meta",{}).get("id",record.get("request_id",index))))
        synchronize(adapter)
        start=time.perf_counter()
        try:
            outputs=adapter.infer(record)
            if len(outputs) != len(record["questions"]):
                raise ValueError("Native output count must equal the number of evaluation questions")
        except (UnsupportedRecord, ValueError) as error:
            if not allow_rejection:
                raise
            rejected.append({"record_id":rid,"reason":str(error),"exception":type(error).__name__})
            continue
        synchronize(adapter)
        timings.append((time.perf_counter()-start)*1000)
        accepted.append(record)
        for (qid,q),output in zip(record["questions"].items(),outputs):
            if "label" not in q:
                raise ValueError("Evaluation questions require independent labels")
            rows.append({"key":f"{rid}::{qid}","record_id":rid,"qid":qid,"type":q["type"],
                         "source":q.get("src",record.get("source",record.get("_meta",{}).get("source","unknown"))),
                         "cluster_id":record.get("_meta",{}).get("group_id",rid),
                         "label":label_index(q),"label_kind":q.get("label_kind",q.get("gold_label_kind","dataset_gold")),
                         "logits":output.detach().float().cpu().tolist()})
        if (index+1)%50 == 0:
            print(json.dumps({"stage":"predict","records":index+1,"total":len(records)}),flush=True)
    return rows,accepted,rejected,timings


def evaluate(rows, temperatures, reference=None):
    result={"raw":metrics(rows),"temperature_calibrated":metrics(rows,temperatures),"temperatures":temperatures}
    kinds=sorted({row.get("label_kind","dataset_gold") for row in rows})
    if kinds != ["dataset_gold"]:
        result["by_label_kind"]={kind:{"raw":metrics([r for r in rows if r.get("label_kind","dataset_gold")==kind]),
                                     "temperature_calibrated":metrics([r for r in rows if r.get("label_kind","dataset_gold")==kind],temperatures)} for kind in kinds}
        result["interpretation"]="Expert actions measure target agreement; event-probability calibration must be read on observed_outcome/deterministic labels separately."
    if reference is not None:
        result["paired"]=compare(reference,rows)
        result["paired_accuracy_ci"]=paired_accuracy_ci(reference,rows)
    return result


def storage_report(adapter,session):
    quantized_numel=sum(x.integer_weight.numel() for x in session.layers.values())
    payload=sum((x.integer_weight.numel()*x.bits+7)//8+x.scales.numel()*4+x.input_scale.numel()*4 for x in session.layers.values())
    native=sum(p.numel()*p.element_size() for p in adapter.model.parameters())
    quant_names={id(module.weight) for name,module in session._modules.items()}
    retained=sum(p.numel()*p.element_size() for p in adapter.model.parameters() if id(p) not in quant_names)
    total_params=sum(p.numel() for p in adapter.model.parameters())
    return {"quantized_parameter_count":quantized_numel,"total_parameter_count":total_params,
            "quantized_parameter_fraction":quantized_numel/total_params,
            "packed_linears_bytes":payload,"retained_native_parameters_bytes":retained,
            "estimated_complete_packed_parameters_bytes":payload+retained,
            "native_parameters_bytes":native,"estimated_parameter_storage_ratio":(payload+retained)/native,
            "note":"Storage estimate, not measured GPU memory or integer-kernel speedup. Retained embeddings, heads, biases and recurrent parameters included."}


def run_experiment(model_name, data_dir, output_dir, *, device="cuda", dtype="bf16", source_dir=None,
                   bits=4, group_size=128, activation_search=True, export=True, seed=20261001,
                   fisher_enabled=False, fisher_records=32, reuse_baseline=None):
    out=Path(output_dir)
    if (out/"summary.json").exists():
        raise FileExistsError("Completed experiment exists; use a fresh output directory")
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed); torch.set_num_threads(4)
    print(json.dumps({"stage":"load","model":model_name}),flush=True)
    adapter=load_model(model_name,device=device,dtype=dtype,source_dir=source_dir)
    adapter.model.eval()
    packages={}
    for pkg in ("torch","transformers","peft","datasets","huggingface-hub","numpy"):
        try: packages[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: pass
    data_dir=Path(data_dir)
    manifest=data_dir/"manifest.json"
    env={"model":adapter.metadata,"packages":packages,"platform":platform.platform(),"seed":seed,
         "data_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
         "gpu":torch.cuda.get_device_name(adapter.device) if adapter.device.type=="cuda" else None}
    env["source_files_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    # Validate every prepared partition, including transfer development even
    # when it is not used for model selection or final evaluation in this run.
    all_splits={path.stem:load_records(path) for path in sorted(data_dir.glob("*.jsonl"))}
    split_hashes={path.stem:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(data_dir.glob("*.jsonl"))}
    manifest_data=json.loads(manifest.read_text(encoding="utf-8-sig")) if manifest.exists() else None
    manifest_binds_splits=False
    if manifest_data is not None:
        entries=manifest_data.get("splits",{})
        for split,actual in split_hashes.items():
            if not isinstance(entries.get(split),dict) or entries[split].get("sha256") != actual:
                raise ValueError(f"Data manifest does not bind current split bytes: {split}")
        manifest_binds_splits=True
    assert_disjoint(all_splits)
    env["split_files_sha256"]=split_hashes
    write_json(out/"environment.json",env)
    splits={split:all_splits[split] for split in ("calibration","temperature_calibration","development","test","transfer_test","ood") if split in all_splits}
    if not all(splits.get(s) for s in ("calibration","temperature_calibration","development","test")):
        raise ValueError("Need nonempty disjoint calibration, temperature, development and test sets")
    baseline, eligibility, exclusions, baseline_timings={}, {}, {}, {}
    # Establish eligibility before quantization. Failures after this point abort.
    reuse=Path(reuse_baseline) if reuse_baseline else None
    if reuse:
        if not (reuse/"summary.json").exists():
            raise ValueError("Cached baseline experiment must be complete")
        previous=json.loads((reuse/"environment.json").read_text())
        if previous["model"]["name"] != model_name or previous["model"]["revision"] != adapter.metadata["revision"] or previous["model"]["dtype"] != adapter.metadata["dtype"] or previous["data_manifest_sha256"] != env["data_manifest_sha256"]:
            raise ValueError("Cached baseline model/dtype/data manifest differs")
        # Device placement may change. All remaining pinned model/adapter
        # metadata affect inference eligibility or raw-logit semantics.
        current_identity=json.loads(json.dumps({k:v for k,v in adapter.metadata.items() if k != "device"}))
        previous_identity={k:v for k,v in previous["model"].items() if k != "device"}
        if previous_identity != current_identity:
            raise ValueError("Cached baseline model/adapter inference metadata differs")
        if not manifest_binds_splits or env["data_manifest_sha256"] is None:
            raise ValueError("Cached baseline reuse requires a manifest binding every current data split")
        if "split_files_sha256" in previous and previous["split_files_sha256"] != split_hashes:
            raise ValueError("Cached baseline input split fingerprints differ")
        # Legacy complete runs recorded only the immutable manifest hash. The
        # equality check above plus its verified per-split hashes binds bytes.
        previous_eligibility=json.loads((reuse/"eligibility.json").read_text())
        if any(any(rid is None for rid in ids) for ids in previous_eligibility["accepted"].values()):
            raise ValueError("Cannot reuse a baseline without stable request IDs")
        env["reused_baseline_environment"]=previous
        write_json(out/"environment.json",env)
    for split,records in splits.items():
        print(json.dumps({"stage":"baseline","split":split,"records":len(records)}),flush=True)
        if reuse:
            rows=load_records(reuse/f"baseline/{split}.jsonl")
            ids=set(previous_eligibility["accepted"][split])
            accepted=[r for r in records if r.get("id",r.get("_meta",{}).get("id",r.get("request_id"))) in ids]
            rejected=previous_eligibility["excluded"][split]
            timing=[]
            # Older v1 rows lacked episode clustering; restore it from source metadata.
            clusters={str(r.get("id",r.get("_meta",{}).get("id",r.get("request_id")))):r.get("_meta",{}).get("group_id") for r in accepted}
            for row in rows:
                if clusters.get(row["record_id"]): row["cluster_id"]=clusters[row["record_id"]]
        else:
            rows,accepted,rejected,timing=predict(adapter,records,allow_rejection=True)
        if not accepted:
            raise ValueError(f"No eligible native inputs in {split}; inspect exclusions")
        baseline[split]=rows; eligibility[split]=accepted; exclusions[split]=rejected; baseline_timings[split]=timing
        write_rows(out/f"baseline/{split}.jsonl",rows)
    write_json(out/"eligibility.json",{"accepted":{s:[r.get("id",r.get("_meta",{}).get("id",r.get("request_id"))) for r in rows] for s,rows in eligibility.items()},"excluded":exclusions})
    temperatures=fit_temperatures(baseline["temperature_calibration"])
    baseline_report={s:evaluate(rows,temperatures) for s,rows in baseline.items() if s not in ("calibration","temperature_calibration")}
    baseline_report["observed_model_time_ms"]={s:{"mean":sum(t)/len(t),"requests":len(t),"includes_encode":True} for s,t in baseline_timings.items() if t}
    if reuse: baseline_report["reused_baseline"]=str(reuse)
    write_json(out/"baseline/metrics.json",baseline_report)
    print(json.dumps({"stage":"collect_activation_statistics","records":len(eligibility["calibration"])}),flush=True)
    with torch.inference_mode(), CalibrationCollector(adapter.backbone,reservoir_size=32,seed=seed) as collector:
        for rec in eligibility["calibration"]: adapter.infer(rec)
    stats=collector.statistics()
    fisher_stats=None
    if fisher_enabled:
        from .decision_stats import DecisionFisherCollector
        print(json.dumps({"stage":"decision_fisher","records":min(fisher_records,len(eligibility["calibration"]))}),flush=True)
        with DecisionFisherCollector(adapter,seed=seed,probes_per_record=2,temperature=1.0) as fisher:
            for record in eligibility["calibration"][:fisher_records]: fisher.collect(record)
        fisher_stats=fisher.attach_input_statistics(stats)
        write_json(out/"decision_fisher.json",fisher.metadata())
    # Same explicit backbone/head scope and group size for the base RTN comparison.
    profiles=[{"name":"rtn","method":"rtn","activation_bits":None,"sensitive_fraction":0},
              {"name":"s1q-local","method":"s1q","activation_bits":None,"sensitive_fraction":0},
              {"name":"s1q-protected","method":"s1q","activation_bits":None,"sensitive_fraction":.05}]
    if activation_search:
        profiles += [{"name":"s1q-a8","method":"s1q","activation_bits":8,"sensitive_fraction":0},
                     {"name":"s1q-a4","method":"s1q","activation_bits":4,"sensitive_fraction":0}]
    if fisher_enabled:
        profiles += [{"name":"s1q-fisher","method":"s1q","activation_bits":None,"sensitive_fraction":0,"fisher":True},
                     {"name":"s1q-fisher-protected","method":"s1q","activation_bits":None,"sensitive_fraction":.05,"fisher":True}]
    candidates=[]
    for profile in profiles:
        print(json.dumps({"stage":"candidate","profile":profile["name"]}),flush=True)
        start=time.perf_counter()
        with quantize_model(adapter.backbone,bits=bits,group_size=group_size,statistics=fisher_stats if profile.get("fisher") else stats,
                            method=profile["method"],activation_bits=profile["activation_bits"],
                            sensitive_fraction=profile["sensitive_fraction"]) as session:
            rows,_,_,_=predict(adapter,eligibility["development"])
            paired=compare(baseline["development"],rows)
            report={"profile":profile,"development":evaluate(rows,{}),"paired":paired,
                    "quantization":session.report(),"storage":storage_report(adapter,session),
                    "candidate_seconds":time.perf_counter()-start}
            write_json(out/f"candidates/{profile['name']}.json",report)
            write_rows(out/f"candidates/{profile['name']}-development.jsonl",rows)
            candidates.append(report)
    s1q_candidates=[c for c in candidates if c["profile"]["method"]=="s1q"]
    # Preserve decisions first. Within 1e-12 ties prefer fewer retained high-bit layers.
    selected=min(s1q_candidates,key=lambda c:(c["paired"]["selection_objective"],c["storage"]["estimated_complete_packed_parameters_bytes"]))
    selection={"selected":selected["profile"],"bits":bits,"group_size":group_size,
               "objective":"development boundary-weighted JS + 0.1 harmful flip rate",
               "selected_before_quantized_test":True,
               "candidates":[{"profile":c["profile"],"objective":c["paired"]["selection_objective"],"development_accuracy":c["development"]["raw"]["accuracy"]} for c in candidates]}
    write_json(out/"selection.json",selection)
    finals={}
    # Test fixed RTN, unprotected local ablation, and development-selected S1Q only.
    final_profiles=[profiles[0],profiles[1]]
    if selected["profile"] != profiles[1]: final_profiles.append(selected["profile"])
    if selected["profile"]["sensitive_fraction"] > 0 or selected["profile"]["activation_bits"] is not None:
        final_profiles.append({**selected["profile"],"name":"rtn-matched","method":"rtn"})
    for profile in final_profiles:
        print(json.dumps({"stage":"final","profile":profile["name"]}),flush=True)
        with quantize_model(adapter.backbone,bits=bits,group_size=group_size,statistics=fisher_stats if profile.get("fisher") else stats,
                            method=profile["method"],activation_bits=profile["activation_bits"],
                            sensitive_fraction=profile["sensitive_fraction"]) as session:
            temp_rows,_,_,_=predict(adapter,eligibility["temperature_calibration"])
            fitted=fit_temperatures(temp_rows)
            report={"profile":profile,"storage":storage_report(adapter,session),"quantization":session.report(),"splits":{}}
            for split in ("test","transfer_test","ood"):
                if split not in eligibility: continue
                rows,_,_,timing=predict(adapter,eligibility[split])
                report["splits"][split]=evaluate(rows,fitted,baseline[split])
                report["splits"][split]["observed_model_time_ms"]={"mean":sum(timing)/len(timing),"requests":len(timing),"includes_encode":True,"execution":"dequantized research simulation"}
                write_rows(out/f"{profile['name']}/{split}.jsonl",rows)
            if export and profile == selected["profile"]:
                run_identity=hashlib.sha256(str(out.resolve()).encode("utf-8")).hexdigest()[:12]
                artifact_dir=Path("artifacts")/run_identity/out.name; artifact_dir.mkdir(parents=True,exist_ok=True)
                artifact_path=artifact_dir/"packed-linears.pt"
                session.export(artifact_path)
                report["artifact"]={"path":str(artifact_path),"bytes":artifact_path.stat().st_size,
                                    "sha256":hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                                    "scope":"linears only; same pinned base/head/tokenizer required; research floating execution"}
            write_json(out/f"{profile['name']}/metrics.json",report)
            finals[profile["name"]]=report
    summary={"status":"complete","model":model_name,"environment":env,"data_directory":str(data_dir),
             "bits":bits,"group_size":group_size,"selection":selection,"baseline":baseline_report,"quantized":finals,
             "limitations":["Not native integer GEMM; activation quantization is simulated.","Storage estimates include retained parameters; no speedup claim.","Eligibility is frozen before quantization and exclusions are published.","No claim of first quantization of this model class."]}
    write_json(out/"summary.json",summary)
    print(json.dumps({"stage":"complete","model":model_name,"selected":selected["profile"]["name"],"summary":str(out/"summary.json")}),flush=True)
    return summary

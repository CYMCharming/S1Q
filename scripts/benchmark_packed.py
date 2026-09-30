"""Measure packed persistent storage and transient GPU peak; no INT4-GEMM claim."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from s1q.data import load_records
from s1q.experiment import predict,write_json
from s1q.metrics import compare
from s1q.models import load_model
from s1q.packed import apply_packed,packed_report
from s1q.quantization import load_quantized_artifact

parser=argparse.ArgumentParser()
parser.add_argument("--model",required=True)
parser.add_argument("--artifact",required=True)
parser.add_argument("--data",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--records",type=int,default=32)
args=parser.parse_args()
torch.set_num_threads(4)
adapter=load_model(args.model)
records=load_records(args.data)[:args.records]
artifact=torch.load(args.artifact,map_location="cpu",weights_only=True)
with torch.inference_mode():
    # Reference path reconstructed from exactly the same artifact.
    session=load_quantized_artifact(adapter.backbone,artifact)
    reference,accepted,rejected,_=predict(adapter,records,allow_rejection=True)
    session.restore()
    del session
    gc.collect();torch.cuda.empty_cache()
    dense_bytes=torch.cuda.memory_allocated()
    adapter.backbone=apply_packed(adapter.backbone,artifact)
    gc.collect();torch.cuda.empty_cache()
    packed_bytes=torch.cuda.memory_allocated()
    for record in accepted[:3]:adapter.infer(record)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    rows,_,_,times=predict(adapter,accepted)
    peak_bytes=torch.cuda.max_memory_allocated()
report={"model":args.model,"model_metadata":adapter.metadata,"gpu":torch.cuda.get_device_name(),
        "persistent_dense_cuda_bytes":dense_bytes,"persistent_packed_cuda_bytes":packed_bytes,
        "peak_packed_cuda_bytes":peak_bytes,"requests":len(accepted),"questions":len(rows),"rejected":rejected,
        "paired":compare(reference,rows),"mean_request_time_ms":sum(times)/len(times),
        "storage":packed_report(adapter.model),"artifact_sha256":hashlib.sha256(Path(args.artifact).read_bytes()).hexdigest(),
        "peak_scope":"Inference after native loading and packed conversion; initial full-checkpoint allocation excluded.",
        "execution":"transient per-layer dequantization + floating GEMM; A4/A8 simulated if enabled; no native low-bit speed claim"}
write_json(Path(args.output),report)
print(json.dumps({k:v for k,v in report.items() if k not in ("model_metadata","storage","rejected")},indent=2))

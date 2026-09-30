import argparse
import json
import torch
from s1q.data import load_records
from s1q.models import load_model
from s1q.quantization import CalibrationCollector,quantize_model
from s1q.decision_stats import DecisionFisherCollector

parser=argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--data",default="work/data-validation/calibration.jsonl")
args=parser.parse_args()
torch.set_num_threads(4)
adapter=load_model(args.model)
records=load_records(args.data)[:4]
with torch.inference_mode(),CalibrationCollector(adapter.backbone,reservoir_size=0) as collector:
    for record in records: adapter.infer(record)
stats=collector.statistics()
with DecisionFisherCollector(adapter,seed=20261001,probes_per_record=2) as fisher:
    for record in records: fisher.collect(record)
combined=fisher.attach_input_statistics(stats)
print(json.dumps(fisher.metadata(),indent=2),flush=True)
with torch.inference_mode(),quantize_model(adapter.backbone,method="s1q",statistics=combined) as session:
    logits=adapter.infer(records[0])
    print(json.dumps({"quantized_layers":len(session.layers),"logits":[z.cpu().tolist() for z in logits]}),flush=True)

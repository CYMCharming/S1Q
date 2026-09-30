import argparse
import json
import torch
from s1q.models import load_model

parser=argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--dtype",default="bf16")
args=parser.parse_args()
torch.set_num_threads(4)
adapter=load_model(args.model,dtype=args.dtype)
record={"state":"The customer asks for a refund.","questions":{
    "refund":{"type":"noul","instructions":"Does the customer request a refund?"},
    "department":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","shipping":"Delivery status"}}}}
with torch.inference_mode():
    logits=adapter.infer(record)
print(json.dumps({"metadata":adapter.metadata,"logits":[x.cpu().tolist() for x in logits],"linears":len(adapter.linear_modules())}),flush=True)

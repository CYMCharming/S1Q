"""Download pinned checkpoints into the configured HF cache (no credentials printed)."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

parser=argparse.ArgumentParser()
parser.add_argument("models",nargs="+")
parser.add_argument("--manifest",default="configs/models.json")
args=parser.parse_args()
models=json.loads(Path(args.manifest).read_text())
for name in args.models:
    cfg=models[name]
    patterns=["*.json","*.safetensors","*.pt","*.txt","*.jinja","tokenizer/*","backbone_config/*"]
    path=snapshot_download(cfg["repo_id"],revision=cfg["revision"],allow_patterns=patterns)
    print(json.dumps({"model":name,"checkpoint":path}),flush=True)
    if name.startswith("kev-"):
        import torch
        meta=torch.load(Path(path)/"head.pt",map_location="cpu",weights_only=True)
        print(json.dumps({"base":meta.get("base"),"base_revision":meta.get("base_revision")}),flush=True)
        snapshot_download(meta["base"],revision=meta.get("base_revision"),allow_patterns=["*.json","*.safetensors","*.txt","*.model","merges.txt"],ignore_patterns=["*vision*","*.bin"])

import importlib.metadata as metadata
import json
import os
import shutil
from pathlib import Path

result = {"packages": {}, "disk": {}}
for package in ["torch", "transformers", "peft", "datasets", "huggingface-hub", "safetensors", "pytest", "einops", "laya"]:
    try:
        result["packages"][package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        result["packages"][package] = None
for root in ["/home/cym", "/mnt/sata2/cym", "/mnt/disk1"]:
    if Path(root).exists():
        result["disk"][root] = {"free_gb": round(shutil.disk_usage(root).free / 2**30, 2)}
result["hub_models"] = [p.name for p in (Path.home()/".cache/huggingface/hub").glob("models--*")]
print(json.dumps(result, indent=2))

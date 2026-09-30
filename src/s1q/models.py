"""Native, revision-pinned adapters for open System One decision models.

``infer`` returns raw logits in question insertion order. Labels and targets are
never passed to the upstream inference model. Quantizers should modify only
``adapter.backbone``; decision heads and recurrent state arithmetic stay native.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


MODEL_REVISIONS = {
    "kev-0.8b": ("jaredpalmer/kev-0.8b", "9a45d25eb2ab761841196625383fa1dff0e56c1e"),
    "kev-4b": ("jaredpalmer/kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"),
    "kev-9b": ("jaredpalmer/kev-9b", "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c"),
    "laya": ("convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"),
    "nanojev": ("C-Tianyu/NanoJev", "047b927b30882a1138fc504821b82ac145a4b81a"),
}
SOURCE_REVISIONS = {
    "kev": "0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91",
    "laya": "6d942c92081fbc139e736bbd9ac0023223c29b7f",
    "NanoJev": "76fdfc9ecdca45a9bcef17991a07d3041a87685a",
}


class UnsupportedRecord(ValueError):
    """A native model cannot represent this request without losing information."""


def _dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    values = {"fp32": torch.float32, "float32": torch.float32,
              "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
              "fp16": torch.float16, "float16": torch.float16}
    try:
        return values[value]
    except KeyError as error:
        raise ValueError(f"Unsupported inference dtype: {value}") from error


def _source(root: str | Path | None, family: str) -> Path:
    if root is None:
        candidates = [Path("work/upstream") / family, Path("work/model-audit") / family]
    else:
        root = Path(root)
        candidates = [root / family, root]
    for path in candidates:
        if (path / ("scripts" if family == "NanoJev" else family.lower())).is_dir():
            path = path.resolve()
            verify_source_revision(path, family)
            return path
    raise FileNotFoundError(f"Pinned {family} source checkout missing; searched {candidates}")


def verify_source_revision(path: str | Path, family: str) -> str:
    """Refuse source drift before importing a checkout or an exported archive.

    Archives require SOURCE_REVISION.txt written by the acquisition/export step.
    A containing S1Q repository is not evidence for an upstream source revision.
    """
    path = Path(path).resolve()
    expected = SOURCE_REVISIONS[family]
    marker = path / "SOURCE_REVISION.txt"
    if (path / ".git").exists():
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        actual = result.stdout.strip()
        # Reject tracked source changes; an untracked provenance marker is fine.
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        if dirty:
            raise RuntimeError(f"{family} source has tracked modifications: {path}")
        if marker.exists() and marker.read_text(encoding="utf-8").strip() != actual:
            raise RuntimeError(f"{family} archive marker disagrees with git HEAD: {path}")
    elif marker.is_file():
        actual = marker.read_text(encoding="utf-8").strip()
    else:
        raise RuntimeError(f"{family} source provenance missing in {path}; run scripts/setup_sources.py "
                           "or preserve SOURCE_REVISION.txt when exporting the pinned checkout")
    if actual != expected:
        raise RuntimeError(f"{family} source revision mismatch: expected {expected}, found {actual}")
    return actual


def _import_source(path: Path) -> None:
    for package in ("kev", "laya"):
        if not (path / package).is_dir():
            continue
        loaded = sys.modules.get(package)
        module_file = getattr(loaded, "__file__", None)
        if loaded is not None and (module_file is None or not Path(module_file).resolve().is_relative_to(path)):
            raise RuntimeError(f"{package} is already imported from another source; start a fresh process "
                               "to use the verified checkout")
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _module_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import upstream source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def public_request(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record.get("questions"), Mapping) or not record["questions"]:
        raise UnsupportedRecord("questions must be a nonempty mapping")
    questions = {}
    for qid, question in record["questions"].items():
        q = {k: v for k, v in question.items() if k in ("type", "instructions", "criteria")}
        if q.get("type") == "boolean":
            q["type"] = "noul"
        if q.get("type") not in ("choice", "noul", "score"):
            raise UnsupportedRecord(f"Unsupported question type for {qid}: {q.get('type')}")
        q.setdefault("instructions", "")
        questions[str(qid)] = q
    return {"state": record["state"], "questions": questions}


def question_keys(question: Mapping[str, Any]) -> list[str]:
    typ = question["type"]
    if typ in ("noul", "boolean"):
        return ["false", "true"]
    if typ == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


class DecisionAdapter:
    model: nn.Module
    backbone: nn.Module

    def __init__(self, name: str, device: str, dtype: str | torch.dtype):
        self.name, self.device, self.compute_dtype = name, torch.device(device), _dtype(dtype)
        self.metadata: dict[str, Any] = {"name": name, "dtype": str(self.compute_dtype),
                                        "device": str(self.device), "raw_logits": True}
        self.stats = {"attempted": 0, "accepted": 0, "rejected": 0}

    def _autocast(self):
        if self.compute_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.compute_dtype)

    def linear_modules(self) -> dict[str, nn.Linear]:
        return {name: module for name, module in self.backbone.named_modules()
                if isinstance(module, nn.Linear)}

    def _infer(self, request: dict[str, Any]) -> list[torch.Tensor]:
        raise NotImplementedError

    def infer(self, record: Mapping[str, Any]) -> list[torch.Tensor]:
        self.stats["attempted"] += 1
        try:
            request = public_request(record)
            logits = self._infer(request)
            if len(logits) != len(request["questions"]):
                raise RuntimeError("Native inference returned the wrong number of questions")
            for (qid, q), values in zip(request["questions"].items(), logits):
                if values.ndim != 1 or len(values) != len(question_keys(q)):
                    raise RuntimeError(f"Native option/logit mapping mismatch: {qid}")
                if not torch.isfinite(values).all():
                    raise RuntimeError(f"Nonfinite native logits: {qid}")
            self.stats["accepted"] += 1
            return logits
        except Exception:
            self.stats["rejected"] += 1
            raise

    def probs(self, record: Mapping[str, Any], temperature: float = 1.0) -> list[torch.Tensor]:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        return [torch.softmax(values.float() / temperature, -1) for values in self.infer(record)]


class KevAdapter(DecisionAdapter):
    def __init__(self, name: str, device="cuda", dtype="bf16", source_dir=None,
                 checkpoint_dir=None, revision=None, max_state=384,
                 max_branch=1024, max_packed=2048):
        super().__init__(name, device, dtype)
        _import_source(_source(source_dir, "kev"))
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.api import SystemOneRequest, to_record
        self._request_class, self._to_record = SystemOneRequest, to_record
        repo, pinned_revision = MODEL_REVISIONS[name]
        revision = revision or pinned_revision
        ck = Checkpoint(str(checkpoint_dir) if checkpoint_dir else f"{repo}@{revision}")
        self.tokenizer, self.model = ck.load(str(device), LoadOptions(
            dtype=self.compute_dtype, merge=True, temperature=1.0, backend="torch",
            fused=False, cuda_graphs=False))
        self.backbone = self.model.lm
        self.context = {"max_state": max_state, "max_branch": max_branch, "max_packed": max_packed}
        self.metadata.update(repo=repo, revision=revision, base=ck.meta.base,
                             base_revision=ck.meta.base_revision,
                             native_temperature=ck.meta.temperature,
                             hybrid=self.model.hybrid, context=self.context,
                             source_revision=SOURCE_REVISIONS["kev"])

    def _infer(self, request):
        rec, _ = self._to_record(self._request_class.model_validate(request))
        enc = self.model.encode(self.tokenizer, rec, max_state=self.context["max_state"],
                                max_branch=self.context["max_branch"], strict=True)
        if len(enc["ids"]) > self.context["max_packed"]:
            raise UnsupportedRecord("Packed Kev request exceeds its configured token budget")
        return self.model.forward(enc)


class LayaAdapter(DecisionAdapter):
    def __init__(self, name="laya", device="cuda", dtype="bf16", source_dir=None,
                 checkpoint_dir=None, revision=None, subfolder=None,
                 max_len=None, head_max_len=None, reject_truncation=True):
        super().__init__(name, device, dtype)
        os.environ.setdefault("USE_TF", "0")
        _import_source(_source(source_dir, "laya"))
        import laya
        from laya.common import collate_items
        self._collate_items = collate_items
        repo, pinned_revision = MODEL_REVISIONS["laya"]
        self.agent = laya.load(str(checkpoint_dir) if checkpoint_dir else repo,
                               device=str(device), revision=revision or pinned_revision,
                               subfolder=subfolder, fast=False, compile=False)
        if self.agent.device != self.device:
            raise RuntimeError(f"Laya unexpectedly fell back to {self.agent.device}")
        self.model, self.backbone, self.tokenizer = self.agent.model, self.agent.model.encoder, self.agent.tok
        self.reject_truncation = reject_truncation
        self.max_len = max_len or self.agent.cfg.get("max_len", 512)
        self.head_max_len = head_max_len or self.agent.cfg.get("head_max_len", 192)
        self.metadata.update(repo=repo, revision=revision or pinned_revision,
                             native_temperature=self.agent.temperature,
                             native_temperature_by_options=self.agent.temperature_by_options,
                             subfolder=subfolder, max_len=self.max_len, head_max_len=self.head_max_len,
                             source_revision=SOURCE_REVISIONS["laya"])

    def _infer(self, request):
        internal = {}
        for qid, q in request["questions"].items():
            self.agent._check_question(qid, q)
            internal[qid] = self.agent._to_internal(q)
        items = self.agent._encode_state(request["state"], list(internal), internal,
                                         max_len=self.max_len, head_max_len=self.head_max_len)
        for qid, item in zip(internal, items):
            if self.reject_truncation and item["state_stats"]["truncated"]:
                raise UnsupportedRecord(f"Laya truncates the state for {qid}")
            if item["options"]["options_distinct"] != item["options"]["options"]:
                raise UnsupportedRecord(f"Laya token budget collapses distinct options for {qid}")
        batch = self._collate_items([items], self.tokenizer.pad_token_id)
        with self._autocast():
            logits, _ = self.model(*[batch[key].to(self.device) for key in
                                    ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")])
        return [row[:len(item["markers"])].float() for row, item in zip(logits, items)]


class NanoJevAdapter(DecisionAdapter):
    def __init__(self, name="nanojev", device="cuda", dtype="bf16", source_dir=None,
                 checkpoint_dir=None, revision=None, max_len=None):
        super().__init__(name, device, dtype)
        source = _source(source_dir, "NanoJev")
        predictor = _module_file(source / "scripts/predict_toy_decisions.py", "s1q_nanojev_predictor")
        self._prepare = predictor.prepare_examples
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        repo, pinned_revision = MODEL_REVISIONS["nanojev"]
        root = Path(checkpoint_dir) if checkpoint_dir else Path(snapshot_download(
            repo_id=repo, revision=revision or pinned_revision,
            allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"]))
        self.run_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if self.run_config.get("set_head") not in ("none", "attention"):
            raise ValueError("NanoJev checkpoint missing its set-head architecture")
        self.tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = AutoConfig.from_pretrained(str(root / "backbone_config"), local_files_only=True)
        config.use_cache = False
        body = AutoModel.from_config(config, attn_implementation="sdpa")
        model_class = predictor.load_decision_model_class()
        self.model = model_class(body, self.run_config["set_head"])
        self.model.load_state_dict(load_file(str(root / "best.safetensors"), device="cpu"), strict=True)
        self.model.to(self.device).float().eval()
        self.backbone = self.model.backbone
        self.max_len = max_len or self.run_config.get("max_length", 512)
        self.metadata.update(repo=repo, revision=revision or pinned_revision,
                             native_temperature=1.0, max_len=self.max_len,
                             set_head=self.run_config["set_head"], source_revision=SOURCE_REVISIONS["NanoJev"])

    def _infer(self, request):
        # Preserve supplied candidate keys/order; unsupported native schemas fail
        # validation rather than being changed into a different decision task.
        questions = {qid: {**q, "type": "boolean" if q["type"] == "noul" else q["type"]}
                     for qid, q in request["questions"].items()}
        payload = {"states": [{"id": "s1q-record", "state": request["state"], "questions": questions}]}
        examples = self._prepare(payload, self.tokenizer, self.max_len)
        with self._autocast():
            logits, _ = self.model(examples, self.tokenizer.pad_token_id)
        return [row[:len(example["candidate_ids"])].float() for row, example in zip(logits, examples)]


def load_model(name: str, device: str = "cuda", dtype: str | torch.dtype = "bf16", **kwargs) -> DecisionAdapter:
    """Load a native model; names and Hub weights are pinned in MODEL_REVISIONS."""
    key = name.lower().replace("_", "-")
    if key == "nano-jev":
        key = "nanojev"
    if key not in MODEL_REVISIONS:
        raise ValueError(f"Unknown model {name!r}; choose {', '.join(MODEL_REVISIONS)}")
    cls = KevAdapter if key.startswith("kev-") else LayaAdapter if key == "laya" else NanoJevAdapter
    return cls(name=key, device=device, dtype=dtype, **kwargs)


load_adapter = load_model

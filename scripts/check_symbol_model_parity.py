"""Gate new native adapters on candidate probability parity and margin grads.

No quantization hyperparameter is selected by this check. Synthetic interface
records and an optional fixed development prefix exercise the native compiler.
The evidence distinguishes adapter verification from benchmark accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from s1q.models import load_model, public_request
from s1q.decision_margin import DecisionMarginCollector
from s1q.quantization import selected_linear_modules


def interface_records():
    return [{"id": "native-interface-three-types", "state": {"ticket": "A refund was requested after a duplicate charge."},
             "questions": {"team": {"type": "choice", "instructions": "Which team handles this?",
                                    "criteria": {"billing": "Payments and refunds", "shipping": "Shipping and delivery"}},
                           "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
                           "urgency": {"type": "score", "instructions": "Rate how urgent this is.",
                                       "criteria": ["low", "medium", "high"]}}},
            {"id": "native-interface-reordered-choice", "state": "Paris is the capital of France.",
             "questions": {"capital": {"type": "choice", "instructions": "Which city is the capital of France?",
                                        "criteria": {"second": "London", "first": "Paris", "third": "Rome"}}}}]


def reference_probs(adapter, record):
    request = public_request(record)
    if adapter.name.startswith("intern-"):
        response = adapter.engine.predict(request)["answers"]
        result = []
        for field, q in request["questions"].items():
            probabilities = response[field]["probabilities"]
            keys = ["no", "yes"] if q["type"] == "noul" else list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            result.append(torch.tensor([probabilities[key] for key in keys]))
        return result
    rows, mappings = [], []
    for qid, q in request["questions"].items():
        row = adapter.native.from_systemone(request["state"], q, qid)
        keys = adapter.native.validate(row)
        desired = ["false", "true"] if q["type"] == "noul" else list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
        rows.append(row)
        mappings.append([keys.index(key) for key in desired])
    native, _ = adapter.engine._logits(rows)
    return [torch.softmax(values.float(), -1)[order].cpu() for values, order in zip(native, mappings)]


def public_api_error(adapter, record, actual):
    """Verify typed values against the original served API at T=1 as well."""
    request = public_request(record)
    if adapter.name.startswith("intern-"):
        answers = adapter.engine.predict(request)["answers"]
    else:
        shipped = adapter.engine.temperature
        try:
            adapter.engine.temperature = {key: 1.0 for key in shipped}
            answers, _ = adapter.engine.decide(request["state"], request["questions"])
        finally:
            adapter.engine.temperature = shipped
    maximum = 0.
    decisions = []
    for (field, question), values in zip(request["questions"].items(), actual):
        answer = answers[field]
        if question["type"] == "noul":
            maximum = max(maximum, abs(float(values[1]) - float(answer["noul"])))
        else:
            keys = list(question["criteria"]) if question["type"] == "choice" else [str(i) for i in range(len(question["criteria"]))]
            reference = torch.tensor([answer["probabilities"][key] for key in keys])
            maximum = max(maximum, float((values-reference).abs().max()))
        if adapter.name.startswith("intern-"):
            keys = ["no", "yes"] if question["type"] == "noul" else list(question["criteria"]) if question["type"] == "choice" else [str(i) for i in range(len(question["criteria"]))]
            canonical = keys[int(values.argmax())]
            ties = int((values == values.max()).sum())
            served = answer["decision"]
            if ties == 1 and canonical != served:
                raise AssertionError("Native public decision differs from canonical raw argmax without a tie")
            decisions.append({"field": field, "canonical_first_argmax": canonical,
                              "public_decision": served, "exact_maximum_ties": ties,
                              "equal": canonical == served})
    return maximum, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--development-prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    adapter = load_model(args.model, checkpoint_dir=args.checkpoint_dir,
                         checkpoint_manifest=args.checkpoint_manifest, device=args.device, dtype="bf16")
    records = interface_records()
    if args.development_prefix:
        records.extend(json.loads(line) for line in Path(args.development_prefix).read_text().splitlines()[:8])
    comparisons, max_error = [], 0.
    with torch.inference_mode():
        for record in records:
            reference = reference_probs(adapter, record)
            actual = [p.cpu() for p in adapter.probs(record)]
            errors = [float((a-b).abs().max()) for a, b in zip(actual, reference)]
            if len(reference) != len(actual):
                raise AssertionError("Native probability question count mismatch")
            api_error, decisions = public_api_error(adapter, record, actual)
            max_error = max(max_error, *errors, api_error)
            comparisons.append({"id": record.get("id"), "questions": len(reference),
                                "max_probability_error": max(errors), "public_api_max_error": api_error,
                                "public_decision_checks": decisions})
    if max_error > 1e-6:
        raise AssertionError(f"Native T=1 candidate probability parity failed: {max_error}")
    scope = selected_linear_modules(adapter.backbone)
    gradient_energy = {name: 0. for name in scope}
    handles = []
    with DecisionMarginCollector(adapter, modules=scope, reservoir_size=16, seed=20261004) as collector:
        def observe(name):
            def hook(module, args, output):
                output.register_hook(lambda grad: gradient_energy.__setitem__(name,
                                     gradient_energy[name] + float(grad.detach().float().square().sum())))
            return hook
        try:
            handles = [module.register_forward_hook(observe(name)) for name, module in scope.items()]
            collector.collect(records[0])
            statistics = collector.statistics()
            gradient = collector.metadata()
        finally:
            for handle in handles:
                handle.remove()
    if len(statistics) != len(scope):
        raise AssertionError("Margin gradients failed to reach every selected native Linear")
    if any(not energy > 0 for energy in gradient_energy.values()):
        raise AssertionError(f"Selected Linear has zero margin gradient energy: {[n for n,e in gradient_energy.items() if not e > 0]}")
    result = {"status": "passed", "model": adapter.metadata, "records": comparisons,
              "native_temperature_for_parity": 1.0, "max_probability_error": max_error,
              "margin_gradient": gradient, "per_layer_margin_gradient_energy": gradient_energy,
              "selected_linear_count": len(scope),
              "scope_complete": True, "evaluation_configuration_selection": False,
              "canonical_argmax_rule": "first maximum in the native candidate order",
              "intern_served_argmax_rule": "lexical label order for exact probability ties",
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device) if adapter.device.type == "cuda" else None,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(adapter.device) if adapter.device.type == "cuda" else None,
              "gpu": torch.cuda.get_device_name(adapter.device) if adapter.device.type == "cuda" else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "model": args.model, "max_probability_error": max_error,
                      "selected_linear_count": len(scope), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()

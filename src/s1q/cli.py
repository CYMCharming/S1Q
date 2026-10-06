import argparse
import json
from .experiment import run_experiment
from .models import MODEL_REVISIONS

def main():
    parser=argparse.ArgumentParser(description="S1Q paired decision-model quantization experiments")
    sub=parser.add_subparsers(dest="command",required=True)
    from .optimization import add_arguments
    add_arguments(sub.add_parser("optimize", help="Current S1Q and matched low-bit baseline evaluation"))
    run=sub.add_parser("run", help="Legacy v0.1 experiment workflow; use optimize for current S1Q")
    run.add_argument("--model",required=True,choices=tuple(MODEL_REVISIONS))
    run.add_argument("--data",required=True)
    run.add_argument("--output",required=True)
    run.add_argument("--device",default="cuda")
    run.add_argument("--dtype",default="bf16",choices=["fp32","bf16","fp16"])
    run.add_argument("--bits",type=int,default=4,choices=[2,3,4,8])
    run.add_argument("--group-size",type=int,default=128)
    run.add_argument("--source-dir")
    run.add_argument("--checkpoint-dir",help="Local copy of the pinned model checkpoint")
    run.add_argument("--checkpoint-manifest",help="Optional JSON with SHA-256 hashes relative to --checkpoint-dir")
    run.add_argument("--no-activation-search",action="store_true")
    run.add_argument("--no-export",action="store_true")
    run.add_argument("--fisher",action="store_true",help="Include decision-output Fisher-weighted candidates")
    run.add_argument("--fisher-records",type=int,default=32)
    run.add_argument("--reuse-baseline",help="Reuse identical pinned native predictions from a completed run")
    run.add_argument("--include-rtn-candidate",action="store_true",
                     help="Allow matched RTN to win the development-set selection")
    run.add_argument("--s1q2",action="store_true",help="Include decision-aware S1Q2 candidates")
    run.add_argument("--reservoir-blend",type=float,default=0.5)
    run.add_argument("--max-reservoir-rows",type=int,default=32)
    run.add_argument("--decision-weighting",default="teacher",choices=["teacher","correct_boundary"])
    run.add_argument("--selection-policy",default="decision_preservation",
                     choices=["decision_preservation","accuracy_first"])
    args=parser.parse_args()
    if args.command == "optimize":
        from .optimization import run as run_optimization
        return run_optimization(args)
    run_experiment(args.model,args.data,args.output,device=args.device,dtype=args.dtype,
                   source_dir=args.source_dir,checkpoint_dir=args.checkpoint_dir,
                   checkpoint_manifest=args.checkpoint_manifest,
                   bits=args.bits,group_size=args.group_size,
                   activation_search=not args.no_activation_search,export=not args.no_export,
                   fisher_enabled=args.fisher,fisher_records=args.fisher_records,reuse_baseline=args.reuse_baseline,
                   include_rtn_candidate=args.include_rtn_candidate,s1q2_enabled=args.s1q2,
                   reservoir_blend=args.reservoir_blend,max_reservoir_rows=args.max_reservoir_rows,
                   decision_weighting=args.decision_weighting,selection_policy=args.selection_policy)

if __name__ == "__main__": main()

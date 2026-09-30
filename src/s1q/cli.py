import argparse
import json
from .experiment import run_experiment

def main():
    parser=argparse.ArgumentParser(description="S1Q paired decision-model quantization experiments")
    sub=parser.add_subparsers(dest="command",required=True)
    run=sub.add_parser("run")
    run.add_argument("--model",required=True,choices=["kev-0.8b","kev-4b","kev-9b","nanojev","laya"])
    run.add_argument("--data",required=True)
    run.add_argument("--output",required=True)
    run.add_argument("--device",default="cuda")
    run.add_argument("--dtype",default="bf16",choices=["fp32","bf16","fp16"])
    run.add_argument("--bits",type=int,default=4,choices=[4,8])
    run.add_argument("--group-size",type=int,default=128)
    run.add_argument("--source-dir")
    run.add_argument("--no-activation-search",action="store_true")
    run.add_argument("--no-export",action="store_true")
    run.add_argument("--fisher",action="store_true",help="Include decision-output Fisher-weighted candidates")
    run.add_argument("--fisher-records",type=int,default=32)
    run.add_argument("--reuse-baseline",help="Reuse identical pinned native predictions from a completed run")
    args=parser.parse_args()
    run_experiment(args.model,args.data,args.output,device=args.device,dtype=args.dtype,
                   source_dir=args.source_dir,bits=args.bits,group_size=args.group_size,
                   activation_search=not args.no_activation_search,export=not args.no_export,
                   fisher_enabled=args.fisher,fisher_records=args.fisher_records,reuse_baseline=args.reuse_baseline)

if __name__ == "__main__": main()

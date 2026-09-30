import argparse
import json
from s1q.data import prepare_kev_data,prepare_nano_data

parser=argparse.ArgumentParser()
parser.add_argument("kind",choices=["shared","nanojev"])
parser.add_argument("--output",required=True)
parser.add_argument("--calibration",type=int,default=128)
parser.add_argument("--temperature-calibration",type=int,default=128)
parser.add_argument("--development",type=int,default=256)
parser.add_argument("--test",type=int,default=1024)
parser.add_argument("--source-root")
args=parser.parse_args()
kwargs=dict(calibration=args.calibration,temperature_calibration=args.temperature_calibration,
            development=args.development,test=args.test)
if args.source_root: kwargs["source_root"]=args.source_root
function=prepare_kev_data if args.kind=="shared" else prepare_nano_data
result=function(args.output,**kwargs)
print(json.dumps({"dataset":result["dataset"],"output":args.output,"splits":result.get("splits",{}),"manifest":args.output+"/manifest.json"},indent=2),flush=True)

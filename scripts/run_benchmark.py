"""Run five independent closed-loop episodes, never an open-loop trajectory."""
import argparse
import copy
import json
from run_episode import execute
from src.utils.config import load_config

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/libero_spatial.yaml")
    p.add_argument("--no-viewer",action="store_true")
    p.add_argument("--episodes",type=int,default=5)
    p.add_argument("--calibration",default="outputs/controller_repeat_calibration.json")
    args=p.parse_args()
    if args.episodes<=0: p.error("episodes must be positive")
    cfg=load_config(args.config)
    if args.no_viewer:cfg.setdefault("viewer",{})["enabled"]=False
    results=[]
    for episode in range(args.episodes):
        current=copy.deepcopy(cfg)
        current["environment"]["seed"]+=episode
        results.append(execute(current,episode=episode,calibration=args.calibration))
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()

import argparse
import json
from src.envs.libero_env import LiberoEnvWrapper
from src.utils.config import load_config

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/libero_spatial.yaml")
    args=p.parse_args()
    cfg=load_config(args.config)
    ec=dict(cfg["environment"],settle_steps=0)
    env=LiberoEnvWrapper(ec)
    try:
        env.reset(ec["task_id"],ec["seed"])
        print(json.dumps(env.action_spec,indent=2))
    finally: env.close()

if __name__=="__main__": main()

"""Run one actual LIBERO episode; dry-run also uses a real camera observation."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from src.agents.astra_agent import AstraAgent
from src.controllers.semantic_controller import ModeBasedSemanticController
from src.envs.libero_env import LiberoEnvWrapper
from src.rollout import run_rollout
from src.schemas.action import SemanticAction
from src.trajectory.logger import TrajectoryLogger
from src.utils.config import load_config
from src.utils.viewers import start_viewers

def verify_calibration(path, cfg, spec):
    report=json.loads(Path(path).read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValueError("Controller calibration failed; stop and review report")
    if report["controller"] != cfg["controller"] or report["action_spec"] != spec:
        raise ValueError("Calibration does not match current controller; recalibrate")
    for key in ("suite", "task_id", "control_freq"):
        if report["environment"][key] != cfg["environment"][key]:
            raise ValueError("Calibration environment mismatch")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def execute(cfg, *, dry_run=False, call_astra=False, episode=0,
            calibration="outputs/controller_repeat_calibration.json"):
    agent = None
    env = LiberoEnvWrapper(dict(cfg["environment"]))
    ec,rc=cfg["environment"],cfg["rollout"]
    try:
        # Fail before simulator work or motor commands if a real policy is unavailable.
        if not dry_run or call_astra:
            agent=AstraAgent(**cfg["agent"])
        if dry_run: env.config["settle_steps"]=0
        obs=env.reset(ec["task_id"],ec["seed"])
        controller=ModeBasedSemanticController(cfg["controller"],env.action_spec)
        metadata={"config":cfg}
        if not dry_run:
            metadata["calibration_sha256"]=verify_calibration(calibration,cfg,env.action_spec)
        try:
            metadata["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True,timeout=5).strip()
        except (subprocess.SubprocessError,OSError): metadata["git_commit"]=None
        if dry_run and not call_astra:
            logger=TrajectoryLogger(rc["output_root"],f"task_{ec['task_id']}",episode,
                {**metadata,"task_instruction":env.task_instruction,"dry_run":True,"model_name":None,"action_spec":env.action_spec},
                fps=ec["control_freq"],category="dry_run")
            try:
                images=logger.save_observation(obs,0)
                action=SemanticAction.noop()
                record={"step":0,"robot_state":obs.robot_state.model_dump(),"images":images,
                    "semantic_action":action.model_dump(mode="json"),"continuous_action":controller.convert(action,obs.robot_state).tolist(),
                    "execution_valid":False,"policy":"explicit diagnostic NO-OP; no Astra call"}
                logger.save_json("initial_observation.json",{"images":images,"robot_state":obs.robot_state.model_dump()})
                record.update(controller.describe(action,obs.robot_state))
                logger.log_step(**record)
                start_viewers(logger,cfg.get("viewer"))
                print(json.dumps(record,indent=2),flush=True)
            finally:
                directory=logger.finalize(success=False,termination="dry_run_no_action_executed",astra_calls=0)
            return {"episode_dir":str(directory),"success":False,"steps":0,"termination":"dry_run_no_action_executed"}
        return run_rollout(env,agent,controller,max_steps=rc["max_steps"],history_length=rc["history_length"],
            max_invalid_decisions=rc["max_invalid_decisions"],output_root=rc["output_root"],episode=episode,
            task_id=ec["task_id"],seed=ec["seed"],dry_run=dry_run,metadata=metadata,save_video=rc["save_video"],grasp_config=cfg.get("grasp"),viewer_config=cfg.get("viewer"))
    finally:
        env.close()
        if agent: agent.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/libero_spatial.yaml")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--call-astra",action="store_true",help="Call model during dry-run; execute zero actions")
    p.add_argument("--no-viewer",action="store_true")
    p.add_argument("--clear-stop",action="store_true",help="Explicitly clear prior operator stop for a NEW episode")
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--max-steps",type=int)
    p.add_argument("--seed",type=int)
    p.add_argument("--calibration",default="outputs/controller_repeat_calibration.json")
    args=p.parse_args()
    cfg=load_config(args.config)
    if args.no_viewer: cfg.setdefault("viewer",{})["enabled"]=False
    if args.clear_stop:
        flag=Path("outputs/STOP_ROLLOUT")
        if flag.exists(): flag.unlink()
    if args.max_steps is not None:
        if args.max_steps<=0: p.error("--max-steps must be positive")
        cfg["rollout"]["max_steps"]=args.max_steps
        cfg["environment"]["horizon"]=max(cfg["environment"]["horizon"],args.max_steps+cfg["environment"]["settle_steps"])
    if args.seed is not None: cfg["environment"]["seed"]=args.seed
    print(json.dumps(execute(cfg,dry_run=args.dry_run,call_astra=args.call_astra,episode=args.episode,calibration=args.calibration),indent=2))

if __name__=="__main__": main()

"""Terminal debugger using the same deterministic controller and real environment."""
import argparse
import numpy as np
from src.controllers.semantic_controller import ModeBasedSemanticController
from src.envs.libero_env import LiberoEnvWrapper
from src.schemas.action import SemanticAction
from src.trajectory.logger import TrajectoryLogger
from src.utils.config import load_config


def parse_command(text):
    parts=text.strip().lower().split()
    action=SemanticAction.noop().model_dump(mode="json")
    if parts==["noop"]: return SemanticAction.model_validate(action)
    if len(parts)!=2: raise ValueError("Use fine x+, normal z-, fine yaw+, gripper close, noop or quit")
    mode,token=parts
    if mode=="gripper": action["gripper"]=token.upper()
    else:
        if token[-1:] not in ("+","-") or token[:-1] not in ("x","y","z","roll","pitch","yaw"):
            raise ValueError("Unknown axis or sign")
        action["mode"]=mode.upper()
        action["precision"]=mode.upper()
        action[token[:-1]]="POSITIVE" if token[-1]=="+" else "NEGATIVE"
    return SemanticAction.model_validate(action)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/libero_spatial.yaml")
    args=parser.parse_args()
    cfg=load_config(args.config); ec=cfg["environment"]
    env=LiberoEnvWrapper(ec); logger=None; success=False; termination="manual_exit"
    try:
        obs=env.reset(ec["task_id"],ec["seed"])
        ctrl=ModeBasedSemanticController(cfg["controller"],env.action_spec)
        logger=TrajectoryLogger(cfg["rollout"]["output_root"],f"task_{ec['task_id']}",metadata={"config":cfg,"task":env.task_instruction,"action_spec":env.action_spec},fps=ec["control_freq"],category="manual")
        logger.save_json("setup_steps.json",env.setup_records)
        images=logger.save_observation(obs,0)
        print("Commands: fine x+, normal z-, fine yaw+, gripper close/open/hold, noop, quit")
        print("Live saved camera images:",logger.dir/"images")
        step=0
        while True:
            text=input("> ").strip()
            if text.lower() in ("quit","exit"): break
            try: semantic=parse_command(text)
            except ValueError as exc:
                print(exc); continue
            counts=ctrl.repeat_counts(semantic,force_single=semantic.gripper.value=='CLOSE')
            start=np.array(env.get_robot_state().position)
            for index in range(counts['repeat_count']):
                sub=ctrl.subaction(semantic,index,force_single=semantic.gripper.value=='CLOSE')
                state=env.get_robot_state();detail=ctrl.describe(sub,state)
                obs,reward,done,info=env.step(np.array(detail['clipped_action']))
                next_images=logger.save_observation(obs,step+1);success=env.check_success()
                logger.log_step(step=step,robot_state=state.model_dump(),semantic_action=semantic.model_dump(mode='json'),
                    executed_semantic_action=sub.model_dump(mode='json'),**detail,**counts,substep_index=index,
                    cumulative_eef_displacement=(np.array(obs.robot_state.position)-start).tolist(),
                    next_robot_state=obs.robot_state.model_dump(),images=images,next_images=next_images,
                    execution_valid=True,reward=reward,success=success)
                print('Action:',detail['clipped_action'],'state:',obs.robot_state.model_dump(),'success:',success)
                step+=1;images=next_images
                if done:break
            if done:
                termination="success" if success else "environment_done"
                break
    except BaseException:
        termination="interrupted_or_error"
        raise
    finally:
        env.close()
        if logger: print("Saved:",logger.finalize(success=success,termination=termination))

if __name__=="__main__": main()

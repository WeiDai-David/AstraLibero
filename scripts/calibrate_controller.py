"""Measure actual single-tick displacement against a seeded NO-OP baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from src.controllers.semantic_controller import ModeBasedSemanticController
from src.envs.libero_env import LiberoEnvWrapper
from src.schemas.action import SemanticAction
from src.trajectory.logger import TrajectoryLogger
from src.utils.config import load_config

AXES = ("x", "y", "z", "roll", "pitch", "yaw")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/libero_spatial.yaml")
    p.add_argument("--all-modes",action="store_true")
    p.add_argument("--rotations",action="store_true")
    p.add_argument("--output",default="outputs/controller_calibration.json")
    args=p.parse_args()
    cfg=load_config(args.config)
    ec=cfg["environment"]
    results=[]
    baseline=None
    actions=[("baseline",SemanticAction.noop())]
    for mode in ('COARSE','NORMAL','FINE') if args.all_modes else (cfg['calibration']['mode'],):
        for axis in AXES if args.rotations else AXES[:3]:
            for sign in ('POSITIVE','NEGATIVE'):
                a=SemanticAction.noop().model_dump(mode='json')
                a.update(mode=mode,precision=mode);a[axis]=sign
                actions.append((mode+'_'+axis+'_'+sign,SemanticAction.model_validate(a)))
    for command in ("OPEN","CLOSE","HOLD"):
        a=SemanticAction.noop().model_dump(mode="json")
        a["gripper"]=command
        actions.append(("gripper_"+command,SemanticAction.model_validate(a)))
    for index,(name,action) in enumerate(actions):
        env=LiberoEnvWrapper(ec)
        log=None
        try:
            obs=env.reset(ec["task_id"],ec["seed"])
            controller=ModeBasedSemanticController(cfg["controller"],env.action_spec)
            log=TrajectoryLogger(cfg["rollout"]["output_root"],name,index,
                metadata={"kind":"controller_calibration","controller":cfg["controller"],"action_spec":env.action_spec,"environment":ec},
                fps=ec["control_freq"],category="calibration")
            log.save_json("setup_steps.json",env.setup_records)
            paths=log.save_observation(obs,0)
            initial=env.get_robot_state()
            initial_target=np.array(initial.gripper_target)
            counts=controller.repeat_counts(action)
            for subindex in range(counts['repeat_count']):
                before=env.get_robot_state()
                sub=controller.subaction(action,subindex)
                detail=controller.describe(sub,before)
                motor=np.array(detail['clipped_action'])
                next_obs,reward,done,info=env.step(motor)
                after=env.get_robot_state()
                after_paths=log.save_observation(next_obs,subindex+1)
                log.log_step(step=subindex,semantic_action=action.model_dump(mode='json'),executed_semantic_action=sub.model_dump(mode='json'),
                    robot_state=before.model_dump(),**detail,**counts,substep_index=subindex,
                    cumulative_eef_displacement=(np.array(after.position)-initial.position).tolist(),
                    next_robot_state=after.model_dump(),images=paths,next_images=after_paths,
                    reward=reward,success=env.check_success(),execution_valid=True)
                paths=after_paths
                if done:break
            delta=np.r_[np.array(after.position)-initial.position,
                (Rotation.from_quat(after.orientation)*Rotation.from_quat(initial.orientation).inv()).as_rotvec()]
            if name=="baseline": baseline=delta
            corrected=delta-baseline
            expected=controller.physical_delta(action,initial)
            target_delta=np.array(after.gripper_target)-initial_target
            record={"name":name,**counts,"initial":initial.model_dump(),"after_batch":after.model_dump(),
                "continuous_action":motor.tolist(),"expected_target_delta":expected.tolist(),"measured_delta":delta.tolist(),
                "baseline_corrected_delta":corrected.tolist(),"gripper_target_delta":target_delta.tolist()}
            if name.startswith("gripper_"):
                command=action.gripper.value
                record["sign_agrees"] = bool(np.allclose(target_delta,0)) if command=="HOLD" else bool(target_delta[0]*(1 if command=="OPEN" else -1)>0)
            elif name!="baseline":
                idx=AXES.index(name.split("_")[1])
                tolerance=cfg["calibration"]["sign_tolerance_m" if idx<3 else "sign_tolerance_rad"]
                record["sign_agrees"] = bool(corrected[idx]*np.sign(expected[idx])>tolerance)
                record["batch_achieved_fraction_of_single_target"]=float(corrected[idx]/expected[idx])
            else: record["sign_agrees"]=True
            # Record every settling tick; these are separate NO-OP control updates,
            # not an assertion that the previous target stays fixed.
            for k in range(cfg["calibration"]["settle_steps"]):
                before=env.get_robot_state()
                noop=SemanticAction.noop()
                motor_noop=controller.convert(noop,before)
                frame,reward,done,info=env.step(motor_noop)
                newpaths=log.save_observation(frame,k+counts["repeat_count"]+1)
                log.log_step(step=k+counts["repeat_count"],semantic_action=noop.model_dump(mode="json"),robot_state=before.model_dump(),
                    continuous_action=motor_noop.tolist(),next_robot_state=env.get_robot_state().model_dump(),
                    images=after_paths,next_images=newpaths,reward=reward,success=env.check_success(),execution_valid=True)
                after_paths=newpaths
                if done: break
            record["after_hold_ticks"]=env.get_robot_state().model_dump()
            record["directory"]=str(log.finalize(success=False,termination="calibration_complete"))
            results.append(record)
            print(json.dumps(record),flush=True)
        finally:
            env.close()
    report={"scope":"Mode-dependent repeated actions at single initial pose; no grasp success claim","all_modes":args.all_modes, "passed":all(r["sign_agrees"] for r in results),
        "controller":cfg["controller"],"environment":ec,"action_spec":controller.spec,"results":results}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("CALIBRATION", "PASS" if report["passed"] else "STOP_FOR_REVIEW", str(out), flush=True)
    return 0 if report["passed"] else 2

if __name__=="__main__": raise SystemExit(main())

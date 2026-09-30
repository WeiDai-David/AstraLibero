"""One fresh model decision -> bounded deterministic repeats -> fresh observation."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import numpy as np
from src.agents.astra_agent import InvalidModelOutput
from src.agents.grasp_feedback import GraspFeedback
from src.trajectory.logger import TrajectoryLogger
from src.trajectory.refiner import TrajectoryRefiner
from src.schemas.action import SemanticAction
from src.utils.viewers import start_viewers


def run_rollout(env, agent, controller, *, max_steps=300, history_length=4,
                max_invalid_decisions=3, output_root='outputs', episode=0, task_id=0,
                seed=0, dry_run=False, metadata=None, save_video=True, category='rollout',
                stop_file=None, grasp_config=None, viewer_config=None):
    """max_steps limits underlying env.step calls, not model decisions. No action replay."""
    if max_steps<=0 or history_length<0 or max_invalid_decisions<=0:raise ValueError('Invalid rollout limits')
    stop_file=Path(stop_file) if stop_file else Path(output_root)/"STOP_ROLLOUT"
    grasp=GraspFeedback(grasp_config)
    agent.controller_context={'semantic_magnitudes':controller.config,'runtime':controller.spec,
        'meaning':'Each physical tick has bounded target increments. Translation/rotation repeat independently; gripper only first tick. Latest observation after batch drives next decision.'}
    task=env.task_instruction
    meta=dict(metadata or {})
    meta.update(task_id=task_id,task_instruction=task,environment_seed=seed,
        controller_configuration=controller.config,action_spec=controller.spec,model_name=agent.model,
        prompt_version=agent.prompt_hash,wire_api=agent.wire_api,reasoning_effort=agent.reasoning_effort,
        source=getattr(env,'source_metadata',{}),dry_run=dry_run,grasp_configuration=grasp.config,
        observation_policy='bounded repeats; all substep images saved; feedback after every batch and immediately after CLOSE',
        max_steps_units='underlying environment steps')
    logger=TrajectoryLogger(output_root,f'task_{task_id}',episode,meta,fps=controller.spec['control_freq'],
        video=save_video,category='dry_run' if dry_run else category)
    history=deque(maxlen=history_length)
    success=False;termination='max_steps';executed=0;decisions=0;invalid_run=0;pending=None
    start_calls,start_invalid=agent.calls,agent.invalid_outputs
    def live(status, **extra):
        logger.live_status(episode_id=logger.dir.name,task=task,episode_status=status,step=executed,
            decision_id=decisions,phase=grasp.state,**grasp.context(),
            astra_calls=agent.calls-start_calls,success=success,**extra)
    try:
        logger.save_json('setup_steps.json',getattr(env,'setup_records',[]))
        obs=env.get_observation();paths=logger.save_observation(obs,0)
        logger.save_json('initial_observation.json',{'robot_state':obs.robot_state.model_dump(),'images':paths,'metadata':obs.metadata})
        live('ready')
        # Direct API users/tests default to headless; CLI explicitly supplies enabled config.
        start_viewers(logger,viewer_config or {'enabled':False})
        success=env.check_success()
        if success:termination='already_successful_at_reset'
        while executed<max_steps and not success:
            if Path(stop_file).exists():termination='operator_stop_for_review';break
            if obs.step!=executed:raise RuntimeError('Stale/out-of-order observation')
            decision_id=decisions;initial_position=np.asarray(obs.robot_state.position);path_length=0.
            pending={'step':executed,'decision_id':decision_id,'task':task,
                'timestamp':datetime.now(timezone.utc).isoformat(),'observation_step':obs.step,
                'robot_state':env.get_robot_state().model_dump(),'images':paths,
                'execution_valid':False,'invalid_model_output':False,'success':False,'reward':None}
            agent.grasp_context=grasp.context()
            agent.request_sink=lambda attempt,payload:logger.save_json(f'requests/step_{decision_id:05d}_attempt_{attempt}.json',payload)
            live('waiting_for_astra',robot_state=pending['robot_state'])
            agent.event_sink=lambda event:live('waiting_for_astra',http_event=event)
            began=time.perf_counter();fallback=False
            try:proposed=agent.act(task,obs,env.get_robot_state(),list(history))
            except InvalidModelOutput:proposed=SemanticAction.noop();fallback=True
            latency=time.perf_counter()-began;decisions+=1
            invalid_run=invalid_run+1 if fallback else 0
            pending.update(proposed_semantic_action=proposed.model_dump(mode='json'),attempts=agent.last_attempts,
                astra_latency=latency,invalid_model_output=any(a.get('valid') is False for a in agent.last_attempts),fallback_noop=fallback)
            if Path(stop_file).exists():
                pending['operator_stop']=True;logger.log_step(**pending);pending=None
                termination='operator_stop_for_review';break
            action=grasp.select(proposed,env.get_robot_state()) if not fallback else SemanticAction.noop()
            # CLOSE always leads directly to another model verification, with no intervening arm batch.
            force_single=fallback or grasp.single or action.gripper.value=='CLOSE'
            counts=controller.repeat_counts(action,force_single=force_single)
            initial_record=dict(pending)
            batch_rows=[];batch_done=False
            for repeat_index in range(counts['repeat_count']):
                if executed>=max_steps or Path(stop_file).exists():
                    termination='operator_stop_for_review' if Path(stop_file).exists() else 'max_steps'
                    batch_done=True;break
                state=env.get_robot_state()
                sub=controller.subaction(action,repeat_index,force_single=force_single)
                detail=controller.describe(sub,state,translation_cap=grasp.cap)
                pending={**initial_record,'step':executed,'substep_index':repeat_index,
                    'observation_step':obs.step,'robot_state':state.model_dump(),'images':paths,
                    'semantic_action':action.model_dump(mode='json'),'executed_semantic_action':sub.model_dump(mode='json'),
                    'attempts':agent.last_attempts if repeat_index==0 else [],
                    'astra_latency':latency if repeat_index==0 else 0.,
                    **counts,**detail,**grasp.context(),'policy_override':grasp.override,
                    'control_phase':action.phase.value,'eef_position':state.position,'gripper_state':state.gripper}
                live('executing',last_action=pending)
                if dry_run:
                    logger.log_step(**pending);print(json.dumps(pending,indent=2),flush=True)
                    pending=None;termination='dry_run_no_action_executed';batch_done=True;break
                next_obs,reward,done,info=env.step(np.array(detail['clipped_action']))
                executed+=1
                post=env.get_robot_state();success=env.check_success()
                displacement=np.asarray(post.position)-np.asarray(state.position)
                path_length+=float(np.linalg.norm(displacement))
                if repeat_index==0:grasp.after_execution(action,post)
                pending.update(execution_valid=True,reward=float(reward),success=success,info=info,
                    next_robot_state=post.model_dump(),next_observation_step=next_obs.step,
                    eef_displacement=displacement.tolist(),
                    cumulative_eef_displacement=(np.asarray(post.position)-initial_position).tolist(),
                    cumulative_eef_path_length_m=path_length,repeat_executed_count=repeat_index+1,
                    grasp_attempt_id=grasp.attempt_id,grasp_retry_count=grasp.retries,
                    recovery_state=grasp.state,grasp_status=grasp.status)
                next_paths=logger.save_observation(next_obs,executed)
                pending['next_images']=next_paths;logger.log_step(**pending);batch_rows.append(pending)
                print(f'decision={decision_id} step={executed-1} repeat={repeat_index+1}/{counts["repeat_count"]} precision={action.precision.value} gripper={sub.gripper.value} phase={grasp.state} success={success}',flush=True)
                committed=pending
                pending=None;obs,paths=next_obs,next_paths
                live('observed',last_action=committed)
                if success or done:
                    termination='success' if success else ('horizon' if info.get('truncated') else 'environment_done')
                    batch_done=True;break
            if batch_rows:
                last=batch_rows[-1]
                summary={'decision_id':decision_id,**counts,'repeat_executed_count':len(batch_rows),
                    'cumulative_eef_displacement':last['cumulative_eef_displacement'],
                    'cumulative_eef_path_length_m':path_length,'final_observation_step':executed,
                    'batch_completed':len(batch_rows)==counts['repeat_count']}
                logger.save_json(f'decision_{decision_id:05d}.json',summary)
                history.append({'step':decision_id,'semantic_action':action.model_dump(mode='json'),
                    'execution_valid':True,'reward':last['reward'],**grasp.context(),
                    'state_summary':{'position':obs.robot_state.position,'gripper':obs.robot_state.gripper},**summary})
            if batch_done:break
            if invalid_run>=max_invalid_decisions:termination='repeated_invalid_model_outputs';break
    except BaseException as exc:
        termination='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error'
        reason=getattr(exc,'failure_reason',termination)
        logger.metadata.update(failure_reason=reason,error={'type':type(exc).__name__,'message':str(exc)})
        if pending:
            pending.update(error_type=type(exc).__name__,attempts=pending.get("attempts",agent.last_attempts),failure_reason=reason)
            logger.log_step(**pending)
        raise
    finally:
        agent.request_sink=None
        agent.event_sink=None
        live('finished',termination=termination,failure_reason=logger.metadata.get('failure_reason'))
        directory=logger.finalize(success=success,termination=termination,astra_calls=agent.calls-start_calls,
            invalid_outputs=agent.invalid_outputs-start_invalid,executed_steps=executed,
            semantic_decisions=decisions,**grasp.summary())
        TrajectoryRefiner().write(directory)
        print('Saved episode:',directory,flush=True)
    return {'success':success,'steps':executed,'decisions':decisions,'termination':termination,'episode_dir':str(directory)}

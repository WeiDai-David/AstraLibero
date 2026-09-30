import copy
import json
from pathlib import Path
import httpx
import numpy as np
import pytest
from src.agents.astra_agent import AstraAgent
from src.agents.grasp_feedback import GraspFeedback, GraspRecoveryError
from src.controllers.semantic_controller import ModeBasedSemanticController
from src.schemas.action import SemanticAction
from src.rollout import run_rollout
from src.schemas.observation import Observation


def action(**kw):
    d=SemanticAction.noop().model_dump(mode='json');d.update(kw)
    return SemanticAction.model_validate(d)

@pytest.mark.parametrize('mode,t,r', [('COARSE',3,2),('NORMAL',2,1),('FINE',1,1)])
def test_independent_repeat_and_gripper(config,spec,state,mode,t,r):
    c=ModeBasedSemanticController(config['controller'],spec)
    a=action(precision=mode,x='POSITIVE',yaw='POSITIVE',gripper='OPEN')
    counts=c.repeat_counts(a)
    assert counts=={'translation_repeat':t,'rotation_repeat':r,'repeat_count':max(t,r)}
    motors=[c.convert(c.subaction(a,i),state) for i in range(max(t,r))]
    assert [v[6] for v in motors]==[-1]+[0]*(max(t,r)-1)
    assert sum(abs(v[0])>0 for v in motors)==t
    assert sum(abs(v[5])>0 for v in motors)==r
    assert all(np.all(np.abs(v)<=1) for v in motors)
    assert c.describe(a,state)['action_clipped'] is False

def test_invalid_repeats_and_excessive_steps(config,spec):
    cfg=copy.deepcopy(config['controller']);cfg['translation_repeat']['COARSE']=0
    with pytest.raises(ValueError):ModeBasedSemanticController(cfg,spec)
    cfg=copy.deepcopy(config['controller']);cfg['translation_magnitudes']['COARSE']=.1
    with pytest.raises(ValueError,match='OSC'):ModeBasedSemanticController(cfg,spec)

def test_grasp_failure_opens_before_alignment_and_counts(state):
    f=GraspFeedback({'max_grasp_retries':1})
    close=f.select(action(gripper='CLOSE'),state);f.after_execution(close,state)
    assert f.state=='VERIFY_GRASP' and not f.verified and f.attempt_id==1
    failed=f.select(action(grasp_status='FAILED',phase='ALIGN',x='POSITIVE',gripper='CLOSE'),state)
    assert failed.gripper.value=='OPEN' and failed.x.value=='ZERO' and f.retries==1
    f.after_execution(failed,state)
    assert f.state=='RECOVER'
    opened=state.model_copy(update={'gripper':[.04,-.04]})
    f.after_execution(f.select(action(x='POSITIVE',phase='ALIGN'),opened),opened)
    assert f.state=='RECOVER_ALIGN'
    close=f.select(action(gripper='CLOSE'),opened);f.after_execution(close,opened)
    assert f.attempt_id==2
    with pytest.raises(GraspRecoveryError) as exc:f.select(action(grasp_status='FAILED'),opened)
    assert exc.value.failure_reason=='grasp_retry_exceeded'

def test_uncertain_lift_once_and_verified_transport(state):
    f=GraspFeedback();f.after_execution(action(gripper='CLOSE'),state)
    a=f.select(action(phase='TRANSPORT',x='POSITIVE',grasp_status='UNCERTAIN'),state)
    assert a.x.value=='ZERO'
    a=f.select(action(phase='OPTIONAL_LIFT_TEST',grasp_status='UNCERTAIN'),state)
    assert f.cap==.001 and f.single and a.z.value=='POSITIVE'
    f.after_execution(a,state)
    a=f.select(action(phase='OPTIONAL_LIFT_TEST',grasp_status='UNCERTAIN'),state)
    assert a.z.value=='ZERO' and f.cap is None
    a=f.select(action(phase='TRANSPORT',x='POSITIVE',grasp_status='SUCCESS'),state)
    assert a.x.value=='POSITIVE' and f.verified and f.successful_attempt==1

def test_success_cannot_be_invented_before_close(state):
    f=GraspFeedback()
    a=f.select(action(phase='TRANSPORT',x='POSITIVE',grasp_status='SUCCESS'),state)
    assert not f.verified and a.x.value=='ZERO'

class Env:
    task_instruction='test fixture'
    setup_records=[]
    def __init__(self,state,done_at=100):self.state=state;self.steps=0;self.actions=[];self.done_at=done_at
    def get_robot_state(self):return self.state
    def get_observation(self):return Observation(rgb=np.full((16,16,3),self.steps,dtype=np.uint8),robot_state=self.state,step=self.steps)
    def step(self,motor):
        self.actions.append(motor);self.steps+=1
        self.state=self.state.model_copy(update={'position':(np.array(self.state.position)+motor[:3]*.01).tolist()})
        return self.get_observation(),0.,self.steps>=self.done_at,{}
    def check_success(self):return False

def make_agent(monkeypatch,handler):
    monkeypatch.setenv('ASTRA_API_KEY','fixture');monkeypatch.setenv('ASTRA_MODEL','fixture');monkeypatch.setenv('ASTRA_BASE_URL','https://fixture.invalid')
    return AstraAgent(client=httpx.Client(transport=httpx.MockTransport(handler)),request_interval_seconds=0)

@pytest.mark.parametrize('budget,done_at,expected',[ (4,100,4),(9,2,2) ])
def test_batches_fresh_feedback_budget_and_displacement(tmp_path,config,spec,state,monkeypatch,budget,done_at,expected):
    seen=[]
    def respond(req):
        data=json.loads(req.content);seen.append(json.loads(data['messages'][1]['content'][0]['text'])['observation_step'])
        return httpx.Response(200,json={'choices':[{'message':{'content':action(precision='COARSE',x='POSITIVE',yaw='POSITIVE',gripper='OPEN').model_dump_json()}}]})
    a=make_agent(monkeypatch,respond);env=Env(state,done_at)
    try:
        result=run_rollout(env,a,ModeBasedSemanticController(config['controller'],spec),max_steps=budget,output_root=tmp_path,save_video=False)
        assert env.steps==expected
        assert seen==([0,3] if expected==4 else [0])
        rows=[json.loads(l) for l in (Path(result['episode_dir'])/'trajectory.jsonl').read_text().splitlines()]
        assert [m[6] for m in env.actions]==([-1,0,0,-1] if expected==4 else [-1,0])
        assert rows[0]['repeat_count']==3
        assert rows[min(2,expected-1)]['cumulative_eef_displacement'][0]==pytest.approx(.01*min(3,expected))
        assert rows[-1]['next_observation_step']==expected
        assert len({r['step'] for r in rows})==expected
    finally:a.close()

def test_close_forces_immediate_reobserve_and_recovery(tmp_path,config,spec,state,monkeypatch):
    seen=[]
    commands=[action(precision='COARSE',gripper='CLOSE',phase='CLOSE'),action(phase='TRANSPORT',grasp_status='FAILED',x='POSITIVE')]
    def respond(req):
        data=json.loads(req.content);seen.append(json.loads(data['messages'][1]['content'][0]['text']))
        return httpx.Response(200,json={'choices':[{'message':{'content':commands[len(seen)-1].model_dump_json()}}]})
    a=make_agent(monkeypatch,respond);env=Env(state)
    try:
        run_rollout(env,a,ModeBasedSemanticController(config['controller'],spec),max_steps=2,output_root=tmp_path,save_video=False)
        assert seen[1]['observation_step']==1
        assert seen[1]['grasp_feedback']['must_verify_now']
        assert [m[6] for m in env.actions]==[1,-1]
        assert np.allclose(env.actions[1][:6],0)
    finally:a.close()

def test_stop_mid_batch_prevents_remaining_ticks(tmp_path,config,spec,state,monkeypatch):
    env=Env(state);step=env.step
    def stop_after_one(m):
        result=step(m);(tmp_path/'STOP_ROLLOUT').write_text('stop');return result
    env.step=stop_after_one
    a=make_agent(monkeypatch,lambda req:httpx.Response(200,json={'choices':[{'message':{'content':action(precision='COARSE',x='POSITIVE').model_dump_json()}}]}))
    try:
        result=run_rollout(env,a,ModeBasedSemanticController(config['controller'],spec),max_steps=6,output_root=tmp_path,save_video=False)
        assert env.steps==1 and result['termination']=='operator_stop_for_review'
    finally:a.close()


def test_repeated_failed_after_clearance_counts_once(state):
    f=GraspFeedback({'max_grasp_retries':1,'max_open_steps':4})
    opened=state.model_copy(update={'gripper':[.036,-.036]})
    f.after_execution(action(gripper='CLOSE'),opened)
    f.lift_used=True
    first=f.select(action(grasp_status='FAILED',gripper='OPEN'),opened)
    f.after_execution(first,opened)
    assert f.state=='RECOVER_ALIGN' and f.retries==1 and f.open_steps==1
    repeated=f.select(action(grasp_status='FAILED',gripper='CLOSE',x='POSITIVE'),opened)
    assert repeated.gripper.value=='OPEN' and repeated.x.value=='ZERO'
    f.after_execution(repeated,opened)
    assert f.retries==1 and f.open_steps==2 and f.lift_used
    assert f.context()['current_attempt_failure_recorded']
    assert f.summary()['failed_grasp_attempt_ids']==[1]
    new_close=f.select(action(gripper='CLOSE',phase='CLOSE'),opened)
    f.after_execution(new_close,opened)
    assert f.attempt_id==2 and not f.lift_used
    with pytest.raises(GraspRecoveryError,match='grasp_retry_exceeded'):
        f.select(action(grasp_status='FAILED'),opened)


def test_repeated_failure_cannot_reset_open_timeout(state):
    f=GraspFeedback({'max_open_steps':2})
    opened=state.model_copy(update={'gripper':[.04,-.04]})
    f.after_execution(action(gripper='CLOSE'),opened)
    for _ in range(2):
        command=f.select(action(grasp_status='FAILED'),opened)
        f.after_execution(command,opened)
    assert f.retries==1
    with pytest.raises(GraspRecoveryError,match='grasp_open_timeout'):
        f.select(action(grasp_status='FAILED'),opened)


def test_multiple_close_decisions_do_not_create_multiple_grasp_attempts(state):
    f=GraspFeedback()
    f.after_execution(f.select(action(gripper='CLOSE'),state),state)
    for _ in range(3):
        command=f.select(action(gripper='CLOSE',grasp_status='UNCERTAIN',x='POSITIVE'),state)
        assert command.gripper.value=='CLOSE' and command.x.value=='ZERO' and f.single
        f.after_execution(command,state)
    assert f.attempt_id==1 and f.retries==0 and not f.verified
    command=f.select(action(gripper='HOLD',grasp_status='UNCERTAIN'),state)
    assert command.gripper.value=='HOLD'  # Gate never silently converts HOLD to CLOSE.


def test_actual_failure_sequence_regression(tmp_path,config,spec,state,monkeypatch):
    env=Env(state.model_copy(update={'gripper':[.036,-.036]}))
    commands=[action(gripper='CLOSE'),action(grasp_status='FAILED',gripper='OPEN'),
              action(grasp_status='FAILED',gripper='OPEN'),action(phase='RECOVER_ALIGN',x='POSITIVE'),
              action(gripper='CLOSE'),action(grasp_status='UNCERTAIN',gripper='CLOSE')]
    received=[]
    def respond(req):
        current=json.loads(json.loads(req.content)['messages'][1]['content'][0]['text'])
        received.append(current)
        return httpx.Response(200,json={'choices':[{'message':{'content':commands[len(received)-1].model_dump_json()}}]})
    a=make_agent(monkeypatch,respond)
    try:
        result=run_rollout(env,a,ModeBasedSemanticController(config['controller'],spec),max_steps=6,
            output_root=tmp_path,save_video=False,grasp_config={'max_grasp_retries':1})
        directory=Path(result['episode_dir'])
        rows=[json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
        assert [r['grasp_retry_count'] for r in rows]==[0,1,1,1,1,1]
        assert [r['grasp_attempt_id'] for r in rows]==[1,1,1,1,2,2]
        assert [float(m[6]) for m in env.actions]==[1,-1,-1,0,1,1]
        assert [r['observation_step'] for r in received]==list(range(6))
        assert received[3]['grasp_feedback']['failed_attempt_ids']==[1]
        assert 'HOLD preserves current target' in received[-1]['grasp_feedback']['gripper_semantics']
        meta=json.loads((directory/'metadata.json').read_text())
        assert meta['grasp_recovery_count']==1 and meta['total_grasp_attempts']==2
    finally:a.close()

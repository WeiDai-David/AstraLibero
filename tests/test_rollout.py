import json
from pathlib import Path
import httpx
import numpy as np
import pytest
from src.agents.astra_agent import AstraAgent, AstraTransportError
from src.controllers.semantic_controller import ModeBasedSemanticController
from src.rollout import run_rollout
from src.schemas.action import SemanticAction
from src.schemas.observation import Observation

class TestEnv:
    __test__=False
    task_instruction="test-only fixture, not a LIBERO success"
    setup_records=[]
    def __init__(self,state): self.state=state; self.steps=0; self.actions=[]
    def get_robot_state(self): return self.state
    def get_observation(self): return Observation(rgb=np.full((16,16,3),self.steps,dtype=np.uint8),robot_state=self.state,step=self.steps)
    def step(self,action):
        self.actions.append(action); self.steps+=1
        return self.get_observation(),float(self.steps==3),self.steps==3,{}
    def check_success(self): return self.steps==3
    def close(self): pass

def make_agent(monkeypatch,handler):
    monkeypatch.setenv("ASTRA_API_KEY","test-only-not-a-secret")
    monkeypatch.setenv("ASTRA_BASE_URL","https://test.invalid/v1")
    monkeypatch.setenv("ASTRA_MODEL","test-fixture")
    return AstraAgent(client=httpx.Client(transport=httpx.MockTransport(handler)))

def test_fresh_images_bounded_history_and_saved_alignment(tmp_path,config,spec,state,monkeypatch):
    seen=[]
    def respond(request):
        payload=json.loads(request.content); content=payload["messages"][1]["content"]
        data=json.loads(content[0]["text"])
        seen.append((data,content[2]["image_url"]["url"]))
        a=SemanticAction.noop().model_dump(mode="json")
        a["x"]="POSITIVE" if data["observation_step"]%2==0 else "NEGATIVE"
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(a)}}]})
    agent=make_agent(monkeypatch,respond); env=TestEnv(state)
    result=run_rollout(env,agent,ModeBasedSemanticController(config["controller"],spec),max_steps=5,history_length=1,output_root=tmp_path)
    assert result["success"] and env.steps==3 and agent.calls==3
    assert [v[0]["observation_step"] for v in seen]==[0,1,2]
    assert [len(v[0]["history"]) for v in seen]==[0,1,1]
    assert len({v[1] for v in seen})==3
    directory=Path(result["episode_dir"])
    rows=[json.loads(x) for x in (directory/"trajectory.jsonl").read_text().splitlines()]
    assert len(rows)==3
    assert [np.sign(a[0]) for a in env.actions]==[1,-1,1]
    for i,row in enumerate(rows):
        assert row["observation_step"]==i and row["next_observation_step"]==i+1
        assert (directory/row["images"]["primary"]).exists()
        assert (directory/row["next_images"]["primary"]).exists()
    assert (directory/"simulation.mp4").stat().st_size>0
    assert (directory/"annotated_trajectory.jsonl").exists()
    assert (directory/"requests/step_00000_attempt_0.json").exists()
    assert "test-only-not-a-secret" not in (directory/"requests/step_00000_attempt_0.json").read_text()
    agent.close()

def test_two_invalid_outputs_safe_noop_and_no_history(tmp_path,config,spec,state,monkeypatch):
    requests=[]
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={"choices":[{"message":{"content":"{}"}}]})
    agent=make_agent(monkeypatch,respond); env=TestEnv(state)
    result=run_rollout(env,agent,ModeBasedSemanticController(config["controller"],spec),max_steps=2,
        history_length=0,max_invalid_decisions=1,output_root=tmp_path,save_video=False)
    assert len(requests)==2 and env.steps==1 and np.allclose(env.actions[0],0)
    assert result["termination"]=="repeated_invalid_model_outputs"
    assert agent.invalid_outputs==2
    agent.close()

def test_transport_failure_does_not_move_and_preserves_partial(tmp_path,config,spec,state,monkeypatch):
    agent=make_agent(monkeypatch,lambda r:httpx.Response(401)); env=TestEnv(state)
    with pytest.raises(AstraTransportError):
        run_rollout(env,agent,ModeBasedSemanticController(config["controller"],spec),output_root=tmp_path,save_video=False)
    assert env.steps==0
    meta=json.loads(next(tmp_path.glob("failure/**/metadata.json")).read_text())
    assert meta["termination"]=="error" and meta["executed_steps"]==0
    agent.close()

def test_dry_run_executes_no_action(tmp_path,config,spec,state,monkeypatch):
    agent=make_agent(monkeypatch,lambda r:httpx.Response(200,json={"choices":[{"message":{"content":SemanticAction.noop().model_dump_json()}}]}))
    env=TestEnv(state)
    result=run_rollout(env,agent,ModeBasedSemanticController(config["controller"],spec),output_root=tmp_path,dry_run=True,save_video=False)
    assert env.steps==0 and result["steps"]==0 and agent.calls==1
    agent.close()


def test_rate_limit_exhaustion_preserves_completed_actions_and_video(tmp_path,config,spec,state,monkeypatch):
    import src.agents.astra_agent as agent_module
    from src.agents.request_policy import RequestGate
    from types import SimpleNamespace
    import imageio.v2 as imageio
    now=[0.0]
    fake=SimpleNamespace(monotonic=lambda:now[0],perf_counter=lambda:now[0],sleep=lambda delay:now.__setitem__(0,now[0]+delay))
    monkeypatch.setattr(agent_module,"time",fake)
    # Disable pacing in this test; retry tests separately verify virtual elapsed time.
    monkeypatch.setattr(agent_module,"REQUEST_GATE",RequestGate())
    received=[]
    env=TestEnv(state)
    def respond(request):
        received.append(json.loads(request.content))
        if len(received)==1:
            return httpx.Response(200,json={"choices":[{"message":{"content":SemanticAction.noop().model_dump_json()}}]})
        assert env.steps==1, "Retries must not execute another motor action"
        return httpx.Response(429)
    agent=make_agent(monkeypatch,respond)
    from src.agents.request_policy import RequestPolicy
    agent.request_policy=RequestPolicy(max_retries=2,request_interval_seconds=0)
    try:
        with pytest.raises(AstraTransportError) as exc:
            run_rollout(env,agent,ModeBasedSemanticController(config["controller"],spec),output_root=tmp_path)
        assert exc.value.failure_reason=="astra_rate_limit"
        assert env.steps==1 and len(env.actions)==1 and agent.calls==4
        meta_path=next(tmp_path.glob("failure/**/metadata.json"))
        meta=json.loads(meta_path.read_text())
        assert meta["failure_reason"]=="astra_rate_limit" and meta["executed_steps"]==1
        directory=meta_path.parent
        rows=[json.loads(line) for line in (directory/"trajectory.jsonl").read_text().splitlines()]
        assert len(rows)==2 and rows[0]["execution_valid"] and not rows[1]["execution_valid"]
        assert rows[1]["failure_reason"]=="astra_rate_limit" and len(rows[1]["attempts"])==3
        assert len(list((directory/"requests").glob("*.json")))==4
        assert (directory/"raw_trajectory.jsonl").read_bytes()==(directory/"trajectory.jsonl").read_bytes()
        reader=imageio.get_reader(str(directory/"simulation.mp4"))
        try:assert reader.count_frames()==2
        finally:reader.close()
        assert (directory/"annotated_trajectory.jsonl").exists()
    finally:agent.close()

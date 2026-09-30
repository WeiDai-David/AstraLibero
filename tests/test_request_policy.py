import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace
import httpx
import numpy as np
import pytest
import src.agents.astra_agent as module
import src.agents.request_policy as policy_module
from src.agents.astra_agent import AstraAgent, AstraTransportError
from src.agents.request_policy import RequestGate, RequestPolicy, retry_after_seconds
from src.schemas.action import SemanticAction
from src.schemas.observation import Observation

@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now=0.0
        sleeps=[]
        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now+=seconds
    c=Clock()
    fake=SimpleNamespace(monotonic=lambda:c.now,perf_counter=lambda:c.now,time=lambda:1700000000+c.now,sleep=c.sleep)
    monkeypatch.setattr(module,"time",fake)
    monkeypatch.setattr(policy_module,"time",fake)
    monkeypatch.setattr(module,"REQUEST_GATE",RequestGate())
    return c

def agent(monkeypatch,handler,wire="responses",**kwargs):
    monkeypatch.setenv("OPENAI_API_KEY","test-secret")
    return AstraAgent(model="test",base_url="https://fixture.invalid",wire_api=wire,
        env_key="OPENAI_API_KEY",client=httpx.Client(transport=httpx.MockTransport(handler)),**kwargs)

def ok(wire):
    text=SemanticAction.noop().model_dump_json()
    if wire=="chat": return httpx.Response(200,json={"choices":[{"message":{"content":text}}]})
    data={"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":text}]}]}}
    return httpx.Response(200,headers={"content-type":"text/event-stream"},text="data: "+json.dumps(data)+"\n\n")

def act(a,state):
    return a.act("test",Observation(rgb=np.zeros((16,16,3),dtype=np.uint8),robot_state=state),state)

@pytest.mark.parametrize("wire",["responses","chat"])
def test_backoff_then_success(monkeypatch,clock,state,wire):
    received=[]; saved=[]
    def handle(request):
        received.append(json.loads(request.content))
        return httpx.Response(429) if len(received)<=5 else ok(wire)
    a=agent(monkeypatch,handle,wire)
    a.request_sink=lambda i,p:saved.append(i)
    try:
        assert act(a,state)==SemanticAction.noop()
        assert clock.sleeps==[2,4,8,16,32]
        assert a.calls==6 and a.invalid_outputs==0
        assert saved==list(range(6))
        assert all(p==received[0] for p in received)
    finally:a.close()

@pytest.mark.parametrize("header,delay",[("45",45),("0",1),("invalid",2),
    (format_datetime(datetime.fromtimestamp(1700000010,tz=timezone.utc),usegmt=True),10)])
def test_retry_after_priority(monkeypatch,clock,state,header,delay):
    count=0
    def handle(request):
        nonlocal count
        count+=1
        return httpx.Response(429,headers={"Retry-After":header}) if count==1 else ok("responses")
    a=agent(monkeypatch,handle)
    try:
        act(a,state)
        assert clock.now==delay
    finally:a.close()

@pytest.mark.parametrize("status",[400,401,403,404])
def test_permanent_errors_not_retried(monkeypatch,clock,state,status):
    a=agent(monkeypatch,lambda r:httpx.Response(status))
    try:
        with pytest.raises(AstraTransportError) as e:act(a,state)
        assert a.calls==1 and clock.sleeps==[]
        assert e.value.status_code==status
    finally:a.close()

@pytest.mark.parametrize("failure",[500,502,503,504,"timeout"])
def test_transient_errors(monkeypatch,clock,state,failure):
    count=0
    def handle(request):
        nonlocal count
        count+=1
        if count==1:
            if failure=="timeout":raise httpx.ReadTimeout("test",request=request)
            return httpx.Response(failure)
        return ok("responses")
    a=agent(monkeypatch,handle)
    try:
        act(a,state)
        assert a.calls==2 and clock.sleeps==[2]
    finally:a.close()

def test_retry_cap_and_rate_reason(monkeypatch,clock,state):
    a=agent(monkeypatch,lambda r:httpx.Response(429),max_retries=4,max_backoff_seconds=4)
    try:
        with pytest.raises(AstraTransportError) as e:act(a,state)
        assert e.value.failure_reason=="astra_rate_limit"
        assert a.calls==5 and clock.sleeps==[2,4,4,4]
    finally:a.close()

def test_interval_across_agents(monkeypatch,clock,state):
    starts=[]
    def handle(request):
        starts.append(clock.now)
        return ok("responses")
    a=agent(monkeypatch,handle);b=agent(monkeypatch,handle)
    try:
        act(a,state);act(b,state);act(a,state)
        assert starts==[0,1,2]
    finally:a.close();b.close()

def test_single_flight_across_agent_instances(monkeypatch,state):
    monkeypatch.setattr(module,"REQUEST_GATE",RequestGate())
    guard=threading.Lock();active=0;peak=0;barrier=threading.Barrier(2)
    def handle(request):
        nonlocal active,peak
        with guard:active+=1;peak=max(peak,active)
        time.sleep(.03)
        with guard:active-=1
        return ok("responses")
    a=agent(monkeypatch,handle,request_interval_seconds=0)
    b=agent(monkeypatch,handle,request_interval_seconds=0)
    def worker(client):barrier.wait(timeout=2);return act(client,state)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,c) for c in (a,b)]
            for f in futures:assert f.result(timeout=3)==SemanticAction.noop()
        assert peak==1
    finally:a.close();b.close()

@pytest.mark.parametrize("kwargs",[{"max_concurrency":2},{"max_concurrency":0},{"max_retries":-1},
    {"max_retries":True},{"request_interval_seconds":-1},{"max_backoff_seconds":1},
    {"initial_backoff_seconds":0},{"request_interval_seconds":float("nan")}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):RequestPolicy(**kwargs)

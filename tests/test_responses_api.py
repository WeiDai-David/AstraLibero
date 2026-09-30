import json
import httpx
import numpy as np
import pytest
from src.agents.astra_agent import AstraAgent, AstraTransportError
from src.schemas.action import SemanticAction
from src.schemas.observation import Observation


def test_responses_sse_and_repair(monkeypatch,state):
    monkeypatch.setenv("OPENAI_API_KEY","test-secret")
    calls=[]
    def handler(request):
        body=json.loads(request.content)
        calls.append(body)
        assert str(request.url)=="https://fixture.invalid/responses"
        assert body["reasoning"]=={"effort":"medium"}
        assert body["store"] is False
        assert body["text"]["format"]["type"]=="json_schema"
        assert any(item["type"]=="input_image" for item in body["input"][0]["content"])
        text="{}" if len(calls)==1 else SemanticAction.noop().model_dump_json()
        event={"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":text}]}]}}
        return httpx.Response(200,headers={"content-type":"text/event-stream"},text="data: "+json.dumps(event)+"\n\n")
    agent=AstraAgent(model="fixture",base_url="https://fixture.invalid",wire_api="responses",env_key="OPENAI_API_KEY",reasoning_effort="medium",client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        action=agent.act("test",Observation(rgb=np.zeros((16,16,3),dtype=np.uint8),robot_state=state),state)
        assert action==SemanticAction.noop()
        assert agent.calls==2 and agent.invalid_outputs==1
        assert calls[1]["input"][-2]["content"][0]["type"]=="output_text"
        assert "test-secret" not in json.dumps(calls)
    finally: agent.close()


def test_truncated_stream_is_not_an_action(monkeypatch,state):
    monkeypatch.setenv("OPENAI_API_KEY","test-secret")
    agent=AstraAgent(model="fixture",base_url="https://fixture.invalid",wire_api="responses",env_key="OPENAI_API_KEY",
        client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,headers={"content-type":"text/event-stream"},text='data: {"type":"response.output_text.delta","delta":"{}"}\n\n'))))
    try:
        with pytest.raises(AstraTransportError):
            agent.act("test",Observation(rgb=np.zeros((16,16,3),dtype=np.uint8)),state)
    finally: agent.close()

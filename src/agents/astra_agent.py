"""Explicit OpenAI-compatible multimodal HTTP transport; never a fake policy."""
import base64
import hashlib
import io
import json
import os
import time
import threading
import logging
from .request_policy import RequestPolicy, REQUEST_GATE, retry_after_seconds
from pathlib import Path
import httpx
from dotenv import load_dotenv
from PIL import Image
from src.schemas.action import SemanticAction
from .base_agent import BaseAgent

class AstraConfigurationError(RuntimeError):
    pass

class AstraTransportError(RuntimeError):
    def __init__(self, message, *, failure_reason="astra_transport_error", status_code=None):
        super().__init__(message)
        self.failure_reason = failure_reason
        self.status_code = status_code


class InvalidModelOutput(ValueError):
    pass

class AstraAgent(BaseAgent):
    def __init__(self, model=None, prompt_path="prompts/astra_semantic_policy.txt", timeout=120, client=None, base_url=None, wire_api="chat", env_key="ASTRA_API_KEY", reasoning_effort=None, max_retries=5, request_interval_seconds=1.0, initial_backoff_seconds=2.0, max_backoff_seconds=32.0, max_concurrency=1):
        self.request_policy = RequestPolicy(max_retries, request_interval_seconds,
            initial_backoff_seconds, max_backoff_seconds, max_concurrency)
        self._decision_lock = threading.Lock()
        load_dotenv()
        self.model = model or os.getenv("ASTRA_MODEL")
        self.base_url = (base_url or os.getenv("ASTRA_BASE_URL", "")).rstrip("/")
        self.wire_api = wire_api
        self.reasoning_effort = reasoning_effort
        if wire_api not in ("chat", "responses"):
            raise AstraConfigurationError("Unsupported wire_api")
        self.key = os.getenv(env_key)
        if not all((self.model, self.base_url, self.key)):
            raise AstraConfigurationError(f"Configure model, base_url and environment variable {env_key}")
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()
        self.client = client or httpx.Client(timeout=timeout)
        self.calls = 0
        self.invalid_outputs = 0
        self.last_attempts = []
        self.last_request = None
        self.request_sink = None
        self.controller_context = {}
        self.grasp_context = {}
        self.event_sink = None
        self.rate_limit_count = 0

    @staticmethod
    def _image(rgb):
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, format="PNG")
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}}

    def act(self, task_instruction, observation, robot_state, history=None):
        # Keep mutable per-decision logs separate even if callers share one agent.
        with self._decision_lock:
            return self._act(task_instruction, observation, robot_state, history)

    def _act(self, task_instruction, observation, robot_state, history=None):
        content = [{"type": "text", "text": json.dumps({"task": task_instruction, "robot_state": robot_state.model_dump(), "observation_step": observation.step, "history": history or [], "camera_metadata": observation.metadata, "controller": self.controller_context, "grasp_feedback": self.grasp_context})}, {"type":"text", "text":"Current primary camera:"}, self._image(observation.rgb)]
        if observation.wrist_rgb is not None:
            content.extend([{"type":"text", "text":"Current wrist camera:"}, self._image(observation.wrist_rgb)])
        messages = [{"role":"system", "content":self.prompt}, {"role":"user", "content":content}]
        schema = SemanticAction.model_json_schema()
        schema["properties"].pop("mode",None)
        schema["required"] = list(schema["properties"])
        for field in schema["properties"].values():
            field.pop("default", None)
        self.last_attempts = []
        for attempt in range(2):
            payload = {"model": self.model, "messages": messages, "response_format": {"type":"json_schema", "json_schema":{"name":"semantic_action", "strict":True, "schema":schema}}}
            if self.wire_api == "responses":
                inputs = []
                for message in messages[1:]:
                    content_in = message["content"]
                    if isinstance(content_in, str):
                        converted = [{"type":"output_text" if message["role"]=="assistant" else "input_text", "text":content_in}]
                    else:
                        converted = [{"type":"input_text","text":item["text"]} if item["type"]=="text" else
                            {"type":"input_image","image_url":item["image_url"]["url"]} for item in content_in]
                    inputs.append({"role":message["role"],"content":converted})
                payload = {"model":self.model,"instructions":self.prompt,"input":inputs,
                    "text":{"format":{"type":"json_schema","name":"semantic_action","strict":True,"schema":schema}},
                    "store":False,"stream":True}
                if self.reasoning_effort: payload["reasoning"]={"effort":self.reasoning_effort}
            body, record = self._request_with_retries(payload, structured_attempt=attempt)
            text = None
            try:
                if self.wire_api == "responses":
                    text = "".join(c["text"] for item in body.get("output",[]) if item.get("type")=="message"
                        for c in item.get("content",[]) if c.get("type")=="output_text")
                else:
                    text = body["choices"][0]["message"]["content"]
                action = SemanticAction.from_model_json(text)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                self.invalid_outputs += 1
                record.update(valid=False, response=text, error=str(exc))
                if attempt == 1:
                    raise InvalidModelOutput("Two invalid structured outputs; execute logged NO-OP") from exc
                messages = messages + [{"role":"assistant", "content":text if isinstance(text,str) else "Invalid response"}, {"role":"user", "content":"Repair the last response. Return exactly one complete JSON object matching the schema. No missing fields; reason may be null."}]
                continue
            record.update(valid=True, response=text, usage=body.get("usage"))
            return action

    def _request_with_retries(self, payload, structured_attempt):
        policy = self.request_policy
        backoff = policy.initial_backoff_seconds
        saw_rate_limit = False
        # One gate for Chat Completions and Responses, across agent instances.
        with REQUEST_GATE.lock:
            for retry in range(policy.max_retries + 1):
                self.last_request = payload
                request_index = len(self.last_attempts)
                if self.request_sink is not None:
                    self.request_sink(request_index, payload)
                REQUEST_GATE.wait_for_start(policy.request_interval_seconds)
                started = time.perf_counter()
                self.calls += 1
                if self.event_sink: self.event_sink({"status":"http_request","http_attempt_count":self.calls,"rate_limit_count":self.rate_limit_count})
                record = {"attempt":request_index, "structured_attempt":structured_attempt,
                          "transport_retry":retry}
                try:
                    if self.wire_api == "responses":
                        body = self._responses(payload)
                    else:
                        response = self.client.post(self.base_url + "/chat/completions",
                            headers={"Authorization":"Bearer " + self.key}, json=payload)
                        response.raise_for_status()
                        body = response.json()
                except (httpx.HTTPError, AstraTransportError) as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    saw_rate_limit = saw_rate_limit or status == 429
                    if status == 429: self.rate_limit_count += 1
                    retryable = status == 429 or (status is not None and 500 <= status <= 599) or isinstance(exc, httpx.TransportError)
                    record.update(latency=time.perf_counter()-started, error_type=type(exc).__name__,
                                  status_code=status, retryable=retryable)
                    self.last_attempts.append(record)
                    if not retryable or retry == policy.max_retries:
                        reason = "astra_rate_limit" if retryable and saw_rate_limit else "astra_transport_error"
                        record.update(retries_exhausted=retryable, failure_reason=reason)
                        raise AstraTransportError(
                            f"Astra request stopped (HTTP {status}, retries={retry}); see recorded attempts",
                            failure_reason=reason, status_code=status) from exc
                    headers = getattr(getattr(exc, "response", None), "headers", {})
                    after = retry_after_seconds(headers.get("Retry-After"))
                    delay = after if after is not None else backoff
                    record.update(retry_delay_seconds=delay,
                                  delay_source="retry_after" if after is not None else "exponential_backoff")
                    logging.getLogger(__name__).warning(
                        "Astra HTTP %s / %s; retry %d/%d in %.2fs (simulation unchanged)",
                        status, type(exc).__name__, retry+1, policy.max_retries, delay)
                    if self.event_sink: self.event_sink({"status":"api_backoff","http_attempt_count":self.calls,"rate_limit_count":self.rate_limit_count,"http_status":status,"retry_delay_seconds":delay,"retry_index":retry+1})
                    time.sleep(delay)
                    backoff = min(backoff * 2, policy.max_backoff_seconds)
                    continue
                record.update(latency=time.perf_counter()-started)
                self.last_attempts.append(record)
                return body, record

    def _responses(self, payload):
        """Read SSE until response.completed; partial output never becomes a motor action."""
        with self.client.stream("POST", self.base_url + "/responses",
                headers={"Authorization":"Bearer " + self.key}, json=payload) as response:
            response.raise_for_status()
            if "text/event-stream" not in response.headers.get("content-type", ""):
                response.read()
                body=response.json()
                if body.get("status") != "completed":
                    raise AstraTransportError("Responses API did not return a completed response")
                return body
            event_lines=[]
            for line in response.iter_lines():
                if line:
                    if line.startswith("data:"): event_lines.append(line[5:].strip())
                    continue
                if not event_lines: continue
                data="\n".join(event_lines); event_lines=[]
                if data=="[DONE]": break
                event=json.loads(data)
                if event.get("type")=="response.completed": return event["response"]
                if event.get("type") in ("error","response.failed","response.incomplete"):
                    raise AstraTransportError("Responses stream reported failure or incomplete output")
        raise AstraTransportError("Responses stream ended without response.completed")

    def close(self):
        self.client.close()

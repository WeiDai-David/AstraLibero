# Astra / LIBERO semantic teacher rollouts

This repository generates teacher trajectories only. It does not implement student distillation.
A multimodal model chooses one semantic correction from **current** RGB images, robot state,
and at most four previous text summaries. A deterministic controller encodes a short
mode-dependent batch of legal `env.step` actions; the latest observation after that batch
is the next model input. Every underlying step is recorded and checked for termination.

```text
Task + current primary/wrist RGB + measured robot state + bounded history
             -> Astra structured semantic action
             -> deterministic world-frame OSC encoding
             -> bounded repeat of legal LIBERO env.step actions
             -> new RGB + state -> repeat
```

Feedback does not guarantee task completion. Success comes only from LIBERO's BDDL
success predicate, never from model text. Stop conditions include success, horizon,
step budget, three consecutive invalid decisions, runtime/API errors, or Ctrl+C.
Completed and partial episodes are retained. Ctrl+C does not discard completed actions.

## Verified local status

- Windows x64, Python 3.10.11; project environment is `.venv`.
- LIBERO source commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- All 779 downloaded runtime files were verified using upstream Git blob hashes.
- robosuite 1.4.0, MuJoCo 2.3.7, BDDL 1.0.1 and NumPy 1.26.4.
- Real task 0 resets, both RGB cameras render, and real simulator stepping works.
- 40 repeat calibration cases passed: baseline, three modes across twelve signed
  translation/rotation directions, and OPEN/CLOSE/HOLD. These cases use one initial pose;
  they do not establish grasp/contact reliability or all-pose safety.
- A real simulator / local HTTP fixture integration run passed three motor steps,
  observation-action alignment checks and a four-frame video decode check.
- No successful manipulation is claimed until the environment success predicate passes. A real provider Responses dry-run has passed; the updated short episode completed 2 real decisions / 6 motor steps / 7 video frames, ending at its test budget without task success.

## Installation / reproduction

Run commands from the repository root. Keep `.tools/python` while using the existing
`.venv`, because it contains the environment's base interpreter. Both are Git-ignored.

```powershell
# Existing environment: no activation required
.\.venv\Scripts\python.exe -m pytest -q

# Fresh installation with an already installed Python 3.10
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe scripts/fetch_libero.py
.\.venv\Scripts\python.exe -m pip install --use-pep517 --no-build-isolation --no-deps -e .tools/vendor/LIBERO-runtime
.\.venv\Scripts\python.exe scripts/setup_local_runtime.py
```

The downloader fetches runtime assets, not demonstrations or model weights. The setup
script registers upstream's outer namespace directory and applies **Windows-only**
compatibility changes to robosuite logging, DLL lookup, and rendering backend.
It does not change physics, gains, action scaling, or task success logic.
`requirements-sim.txt` lists direct simulation pins; `requirements-lock.txt` captures
all currently installed versions. Neither includes downloaded source as a wheel.

## Astra connection

The configured provider is `jzzzi`, model `gpt-6-astra`, Responses wire protocol,
base URL `https://sub.jzzzi.top`, and reasoning effort `medium`. These non-secret
settings live in `configs/libero_spatial.yaml`. The adapter reads the key from the
process environment variable `OPENAI_API_KEY`. No key is written into YAML, JSON
request records, or Git. Keep the calling terminal's environment available when
running the scripts.

Responses requests use current primary/wrist `input_image` data URLs, robot state,
camera transforms, action magnitudes, and bounded text history. Streaming responses
must reach `response.completed`; incomplete streams never produce motor actions.
`text.format` requests strict JSON Schema. One repair is attempted for malformed
structured output; two malformed responses result in one logged NO-OP. HTTP 429, 5xx and network transport errors are retried with bounded backoff.
Other HTTP errors (including 400/401/403) stop immediately and preserve partial records.

A real image-based Responses dry-run has succeeded with this provider. A real
Astra rollout is now being evaluated; see saved episode metadata for final status.
This verifies the configured protocol rather than claiming support for every Astra
service. The older Chat Completions adapter remains available via `wire_api: chat`.

The prompt is editable in `prompts/astra_semantic_policy.txt`. Full request bodies
and action responses are saved for traceability, excluding authorization headers.
A file named `outputs/STOP_ROLLOUT` requests a graceful stop before the next motor
step (including when a model call is already in flight). Remove the flag only after
review, before starting another episode.

## Controller: per-mode step size and independent repeat counts

Large scaling has been removed. Configuration is deterministic and selected by Astra's
`precision`. The legacy `mode` field is retained for old code/records; when `precision`
is supplied it is authoritative. New model output need not contain `mode`.

| Precision | Translation per active axis/tick | Translation repeat | Rotation per active axis/tick | Rotation repeat |
|---|---:|---:|---:|---:|
| COARSE | 0.05 m | 3 | 12 degrees | 2 |
| NORMAL | 0.02 m | 2 | 5 degrees | 1 |
| FINE | 0.005 m | 1 | 2 degrees | 1 |

Translation and rotation run concurrently for `max(translation_repeat, rotation_repeat, 1)`
ticks. Each component becomes zero when its own count ends. OPEN/CLOSE occurs only on
substep zero; subsequent substeps send HOLD. A gripper-only command lasts one tick.
CLOSE always shortens the batch to one tick to obtain immediate grasp verification.
Every tick respects runtime OSC/action bounds, workspace limits and optional norm limits.
Oversized configured magnitudes are rejected instead of relying on repeated clipping.
The OSC source, gains, physical output limits (+/-0.05 m, +/-0.5 rad per axis here) and
normalization are unchanged. Repeats do not promise that target increments accumulate
as actual motion; measured displacement is logged separately.

`requested_action` and `clipped_action` use normalized motor units. `base_action` and
`requested_physical_delta` explicitly use meters/radians; `physical_target_delta` is
the decoded action actually sent. Each row logs decision_id, substep_index, both repeat
counts, cumulative EEF displacement vector and path length. A decision summary file
records requested versus executed repeat count if success/stop/budget truncates a batch.
`max_steps` counts motor steps, not decisions. The default is 300.

Each API request follows the latest completed batch. Images/states are saved after EVERY
motor step. Success, horizon and operator stop are checked inside the repeat loop.
No simulation time advances while the API waits or retries.

## Grasp verification and recovery

The separate `GraspFeedback` gate accepts only Astra's visual grasp judgement; neither
controller nor gate detects objects, computes targets, or plans grasps. CLOSE triggers
VERIFY_GRASP and a fresh model call. Joint closure never implies grasp success.
Unverified lateral transport is blocked. SUCCESS permits LIFT/TRANSPORT/PLACE.
FAILED forces stationary OPEN before RECOVER_ALIGN. Recovery continues one observed
step at a time until configured finger clearance, with an opening timeout.
UNCERTAIN may request a single tiny lift test (1 mm target cap, once per attempt),
followed by another visual judgement. No automatic success is inferred from the test.
The default allows three recoveries after the initial attempt; a further failure ends
with `failure_reason=grasp_retry_exceeded`. Uncertain verification and opening are also
bounded. Proposed versus actually executed semantic actions and gate overrides are
stored separately, preserving auditability.

```json
{"precision":"FINE","x":"POSITIVE","y":"ZERO","z":"NEGATIVE",
 "roll":"ZERO","pitch":"ZERO","yaw":"ZERO","gripper":"OPEN",
 "phase":"ALIGN","grasp_status":"NOT_APPLICABLE","reason":"Align before descending."}
```

Grasp status: NOT_APPLICABLE / SUCCESS / FAILED / UNCERTAIN. Phase includes APPROACH,
ALIGN, CLOSE, VERIFY_GRASP, OPTIONAL_LIFT_TEST, RECOVER, RECOVER_ALIGN, LIFT, TRANSPORT,
PLACE. New live outputs require precision/phase/grasp_status; legacy records remain readable.

## Automatic viewers

`run_episode.py` opens one status window and one dual-camera window per episode.
The independent viewers refresh from files and never call/reset the simulator.
Closing either with Q/Esc or the window close button does not stop control.
Missing wrist imagery renders `WRIST CAMERA NOT AVAILABLE`. Headless/GUI errors do not
abort the rollout; telemetry sharing errors on Windows are best-effort and nonfatal.
Viewer process logs live outside the episode directory so they cannot block its move.
Use `--no-viewer` for servers; `viewer.refresh_every_steps` defaults to 1.
The status window shows actions, physical/normalized values, independent repeats, EEF
state, grasp feedback, latency, HTTP attempts, 429 retries and the last eight motor steps.

## Commands

```powershell
# Inspect actual motor scaling without stepping
.\.venv\Scripts\python.exe scripts/inspect_action_space.py

# Real reset + both cameras + conversion; zero env.step and zero API calls
.\.venv\Scripts\python.exe scripts/run_episode.py --dry-run

# Measured calibration with per-step videos, required before model rollouts
.\.venv\Scripts\python.exe scripts/calibrate_controller.py --rotations --all-modes --output outputs/controller_repeat_calibration.json

# Real terminal control: fine x+, normal z-, fine yaw+, gripper close, noop, quit
.\.venv\Scripts\python.exe scripts/manual_semantic_control.py

# Real camera + real model call, but execute zero actions
.\.venv\Scripts\python.exe scripts/run_episode.py --dry-run --call-astra

# One real closed-loop model episode; default task 0, seed 0
.\.venv\Scripts\python.exe scripts/run_episode.py --episode 5 --max-steps 300 --clear-stop

# Five episodes using independent seeds
.\.venv\Scripts\python.exe scripts/run_benchmark.py --episodes 5

# Summary includes only actual rollout success/failure categories
.\.venv\Scripts\python.exe scripts/analyze_rollouts.py --root outputs --output results.csv

# Integration diagnostic: real physics + labelled LOCAL HTTP fixture, not Astra
.\.venv\Scripts\python.exe scripts/validate_closed_loop.py
```

Changing controller scaling/configuration invalidates the stored calibration gate.
The default 300-motor-step budget can be overridden; larger budgets can be expensive
because every semantic batch requires a fresh multimodal call. Simulator time
is paused during the request, preserving synchronous closed-loop semantics.

## Saved data

```text
outputs/
  calibration/<case>/episode_.../   # real directional probes; not model rollouts
  integration_test/task_0/.../     # real simulator with labelled local HTTP fixture
  dry_run/task_0/.../              # no executed motor actions
  in_progress/task_0/.../          # incremental records if process is killed
  success/task_0/episode_.../
  failure/task_0/episode_.../
    metadata.json
    setup_steps.json
    initial_observation.json
    trajectory.jsonl
    raw_trajectory.jsonl
    annotated_trajectory.jsonl
    refinement_stats.json
    simulation.mp4
    images/step_00000_primary.png
    images/step_00000_wrist.png
    requests/step_00000_attempt_0.json
```

Each policy transition records pre/post images, pre/post robot state, semantic action,
continuous motor action, target delta, reward, success, execution validity, timestamps,
model attempts and latency. JSONL is flushed and synced after each step. The video
contains both cameras side-by-side, initial frame plus every post-step frame, at the
simulator control rate (API waiting time is excluded). Lossless PNGs are the training
source, not compressed video frames. Before policy start, ten settling NO-OPs are
listed separately in `setup_steps.json`; their images are not part of the policy video.
Dry-run disables settling entirely.

Refinement marks reversals, oscillations, repeated actions, candidate redundant NO-OPs
and prolonged alternating corrections. It never changes recorded observation-action
pairs or claims that dropping feedback steps is safe. Raw episodes are retained.

## Limits and interpretation

- Initialization currently uses seeded procedural reset. Official benchmark init-state
  files are not loaded; generated results must not be labelled standard benchmark scores.
- Upstream `use_object_obs=False` fails initialization in this commit. The simulator
  computes those extra fields internally, but the wrapper exposes **only RGB, camera
  transforms and robot proprioception**. The policy/controller never receives object poses.
- Direction calibration passes only at the tested initial pose and magnitude. Contact,
  camera-grounding, eventual task success and real Astra protocol support need evaluation.
- Unit tests with fixture responses are not scientific success evidence. Calibration and
  integration categories are explicitly excluded from success-rate aggregation.
- The Windows renderer needs a working OpenGL driver. Local compatibility patches are
  replayable through `scripts/setup_local_runtime.py`.


## Astra request throttling and retries

All outbound Astra calls are centralized in `src/agents/astra_agent.py`: both
`/responses` (including the entire SSE stream) and `/chat/completions` use the same
request policy. Episode and benchmark CLIs use the `agent` configuration directly.
The benchmark executes episodes sequentially.

```yaml
agent:
  max_retries: 5
  request_interval_seconds: 1.0
  initial_backoff_seconds: 2.0
  max_backoff_seconds: 32.0
  max_concurrency: 1
```

`max_retries` counts retries **after** the initial HTTP request, so the default
permits six HTTP attempts per structured request. The default backoff is
2, 4, 8, 16, 32 seconds. A valid `Retry-After` value (seconds or HTTP date) takes
priority, even when it exceeds `max_backoff_seconds`; that cap applies only to
computed exponential backoff. Invalid headers fall back to exponential backoff.
The one-second minimum between request starts still applies, including repair
requests, retries, and subsequent decisions.

A shared process-wide gate serializes agent instances and holds the slot until
a streamed response finishes. Only `max_concurrency: 1` is accepted for this
implementation. This is process-local coordination: do not start multiple rollout
processes against the same account expecting a shared cross-process rate limiter.

HTTP 429, HTTP 5xx and network transport errors (including timeouts) can retry.
HTTP 400/401/403 and other non-retryable statuses fail immediately. Each HTTP
attempt has a separate request record, retry index, latency, status code and delay
source; authorization headers are never recorded. Network retries reuse the same
observation and payload. They do not call `env.step`, reset LIBERO, or replay any
previous motor action. The separate one-repair rule for malformed semantic JSON
is unchanged.

When rate-limit retries are exhausted, the episode is finalized under
`outputs/failure/` with `failure_reason: astra_rate_limit` in metadata and the
unexecuted step record. Previously completed transitions, raw/annotated JSONL,
images and the simulation video are retained. Pure server/network failures without
any 429 are labelled `astra_transport_error`, avoiding misclassification as rate
limits. No partial or failed API output is converted into an extra motor action.

Retry-policy tests use mocked HTTP and virtual time; no paid provider calls or
real robot actions are performed by this test suite.


## New YAML sections

Controller adds `translation_repeat` and `rotation_repeat`, and updates existing
`translation_magnitudes` and `rotation_degrees`. There are no scaling multipliers.
`grasp` includes enable_lift_test, max_grasp_retries, lift_test_translation_m,
open_width_threshold_m, max_open_steps and max_verification_steps.
`viewer` includes enabled, show_status_window, show_camera_window and refresh_every_steps.
Existing API retry and request-spacing parameters remain unchanged.

The repeated-action calibration report is `outputs/controller_repeat_calibration.json`.
Old one-tick reports are retained and are not accepted for changed controller settings.
`--clear-stop` explicitly clears a previous STOP_ROLLOUT flag for a NEW episode; it does
not resume or replay a prior episode. The actual short validation is at
`outputs/failure/task_0/episode_0004_16d79e41`, termination=max_steps, not task success.
Grasp recovery is unit-tested; the six-step real test only reached approach. It does
not validate physical recovery or eventual task success. Repeat batches intentionally
reduce visual-feedback frequency and should stay FINE/one tick near contact.


## Recovery-count correction after episode 0005

A failed `grasp_attempt_id` consumes recovery budget at most once. Repeated FAILED
judgements during RECOVER/RECOVER_ALIGN keep opening safely without consuming another
retry or resetting the opening timeout. Only a subsequent executed CLOSE outside
VERIFY_GRASP starts a new attempt. Consecutive CLOSE decisions in VERIFY_GRASP remain
the same attempt and each require a fresh observation. Context exposes failed_attempt_ids;
metadata records failed_grasp_attempt_ids.

Panda CLOSE is incremental; HOLD preserves the previous target, even when only partly
closed. The prompt explicitly asks Astra to choose CLOSE again on a later decision
when more closure is appropriate, while reporting UNCERTAIN until visual evidence
supports SUCCESS. The gate never silently converts HOLD into CLOSE. The 1 mm lift-test
cap, OSC limits and per-mode repeat configuration are unchanged.

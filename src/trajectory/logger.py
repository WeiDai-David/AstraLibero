"""Crash-resilient JSONL plus lossless observation images and a simulation-time video."""
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image

class TrajectoryLogger:
    def __init__(self, root="outputs", task="task_0", episode=0, metadata=None, fps=20, video=True, category="rollout"):
        self.root = Path(root).resolve()
        self.category = category
        name = f"episode_{episode:04d}_{uuid.uuid4().hex[:8]}"
        self.dir = self.root / "in_progress" / task / name
        self.dir.mkdir(parents=True, exist_ok=False)
        for sub in ("images", "requests"):
            (self.dir/sub).mkdir()
        self.metadata = dict(metadata or {})
        self.metadata.update(status="in_progress", started_at=datetime.now(timezone.utc).isoformat(), video_fps=fps,
            video_timebase="simulation time; model latency excluded", category=category)
        self.steps = []
        self.start = time.perf_counter()
        self.stream = (self.dir/"trajectory.jsonl").open("a",encoding="utf-8")
        self.video = imageio.get_writer(str(self.dir/"simulation.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=1) if video else None
        self.frames = 0
        self._write_metadata()

    def _write_metadata(self):
        tmp = self.dir/"metadata.tmp"
        tmp.write_text(json.dumps(self.metadata,indent=2,allow_nan=False),encoding="utf-8")
        for attempt in range(10):
            try:
                tmp.replace(self.dir/"metadata.json")
                break
            except PermissionError:
                if attempt==9:
                    (self.dir/"metadata.json").write_text(tmp.read_text(encoding="utf-8"),encoding="utf-8")
                else: time.sleep(.05)

    def live_status(self, **data):
        # GUI telemetry is best effort. A reader/Windows sharing violation must
        # never abort a motor sequence or prevent trajectory/video finalization.
        try:
            tmp=self.dir/"live_status.tmp"
            text=json.dumps(data,allow_nan=False)
            tmp.write_text(text,encoding="utf-8")
            try: tmp.replace(self.dir/"live_status.json")
            except OSError:
                (self.dir/"live_status.json").write_text(text,encoding="utf-8")
        except (OSError,ValueError):
            self.metadata['viewer_status_write_failures']=self.metadata.get('viewer_status_write_failures',0)+1

    def save_json(self, name, data):
        (self.dir/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")

    def save_observation(self, observation, step):
        paths = {}
        images = []
        for name, array in (("primary",observation.rgb),("wrist",observation.wrist_rgb)):
            if array is None: continue
            a = np.asarray(array)
            if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
                raise ValueError("Expected RGB uint8 camera frame")
            rel = f"images/step_{step:05d}_{name}.png"
            Image.fromarray(a).save(self.dir/rel)
            paths[name] = rel
            images.append(a)
        if not images: raise ValueError("No observation image supplied")
        if self.video:
            self.video.append_data(np.concatenate(images,axis=1))
        self.frames += 1
        return paths

    def log_step(self, **data):
        line = json.dumps(data,allow_nan=False)
        self.stream.write(line+"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.steps.append(data)
        self.metadata.update(last_executed_step=data.get("step"),last_decision_id=data.get("decision_id"))


    def finalize(self, success=False, termination="unknown", **extra):
        self.stream.close()
        video_error = None
        if self.video:
            try: self.video.close()
            except Exception as exc: video_error = str(exc)
        self.metadata.update(success=bool(success),termination=termination,status="complete",
            total_steps=sum(bool(r.get("execution_valid")) for r in self.steps),decisions=extra.get("semantic_decisions",len(self.steps)),trajectory_records=len(self.steps),
            rollout_duration=time.perf_counter()-self.start,video_frames=self.frames,
            video_error=video_error,**extra)
        self._write_metadata()
        raw = self.dir/"raw_trajectory.jsonl"
        raw.write_bytes((self.dir/"trajectory.jsonl").read_bytes())
        bucket = self.category if self.category != "rollout" else ("success" if success else "failure")
        target = self.root/bucket/self.dir.parent.name/self.dir.name
        target.parent.mkdir(parents=True,exist_ok=True)
        # Both paths are constructed under the checked output root; no recursive move/deletion.
        if self.root not in target.resolve().parents: raise ValueError("Invalid destination")
        for attempt in range(20):
            try:
                self.dir.rename(target)
                break
            except PermissionError:
                if attempt==19: raise
                time.sleep(.05)
        self.dir = target
        if video_error: raise RuntimeError(f"Video finalization failed: {video_error}")
        return self.dir

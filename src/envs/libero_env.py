"""All LIBERO/robosuite-specific code, including runtime action introspection."""
import importlib.util
import json
import os
import random
import runpy
from pathlib import Path
import numpy as np
import yaml
from src.schemas.action import RobotState
from src.schemas.observation import Observation

class LiberoEnvWrapper:
    def __init__(self, config=None):
        self.config = config or {}
        self.env = None
        self._obs = None
        self.steps = 0
        self.task_instruction = ""
        self.action_spec = None

    def reset(self, task_id=0, seed=0):
        self.close()
        spec = importlib.util.find_spec("libero")
        if spec is None:
            raise RuntimeError("Install LIBERO using the documented pinned runtime snapshot")
        root = Path(next(iter(spec.submodule_search_locations))) / "libero"
        cfgdir = Path(self.config.get("libero_config_path", ".tools/libero-config")).resolve()
        cfgdir.mkdir(parents=True, exist_ok=True)
        os.environ["LIBERO_CONFIG_PATH"] = str(cfgdir)
        paths = {"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"), "init_states": str(root / "init_files"), "assets": str(root / "assets"), "datasets": str(root.parent / "datasets")}
        (cfgdir / "config.yaml").write_text(yaml.safe_dump(paths), encoding="utf-8")
        # Loading the upstream static task map avoids importing training-only torch.
        task_map = runpy.run_path(str(root / "benchmark/libero_suite_task_map.py"))["libero_task_map"]
        suite = self.config.get("suite", "libero_spatial")
        if suite not in task_map or not 0 <= task_id < len(task_map[suite]):
            raise ValueError("Invalid suite or task ID (official task order index 0)")
        name = task_map[suite][task_id]
        bddl = root / "bddl_files" / suite / (name + ".bddl")
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.errors import RandomizationError
        random.seed(seed)
        np.random.seed(seed)
        cameras = [self.config.get("primary_camera", "agentview")]
        wrist = self.config.get("wrist_camera", "robot0_eye_in_hand")
        if wrist: cameras.append(wrist)
        # Upstream has an initialization bug with use_object_obs=False. Its
        # extra observations stay private; only RGB and robot proprioception leave this adapter.
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=cameras,
            camera_heights=self.config.get("image_size", 256), camera_widths=self.config.get("image_size", 256),
            control_freq=self.config.get("control_freq", 20), horizon=self.config.get("horizon", 2000),
            ignore_done=False, use_object_obs=True)
        self.env.seed(seed)
        # The upstream wrapper reset retries forever; bound only RandomizationError here.
        for attempt in range(10):
            try:
                self._obs = self.env.env.reset()
                break
            except RandomizationError:
                if attempt == 9: raise
        self.steps = 0
        self.task_instruction = self.env.language_instruction
        self.action_spec = self.inspect_action_space()
        self.source_metadata = {"task_name": name, "suite": suite, "task_order_index": 0,
            "initialization": "seeded procedural reset (not benchmark init-state evaluation)",
            "bddl_file": str(bddl), "source_manifest": str(root.parent.parent / "source_manifest.json")}
        manifest = root.parent.parent / "source_manifest.json"
        self.source_metadata["commit"] = json.loads(manifest.read_text())["commit"] if manifest.exists() else None
        self.setup_records = []
        for i in range(self.config.get("settle_steps", 10)):
            before = self.get_robot_state().model_dump()
            obs, reward, done, info = self.env.step(np.zeros(7))
            self._obs = obs
            self.setup_records.append({"step": i, "action": [0.0]*7, "before": before, "after": self.get_robot_state().model_dump(), "reward": float(reward)})
            if done: break
        return self.get_observation()

    def inspect_action_space(self):
        robot = self.env.robots[0]
        c = robot.controller
        if c.name != "OSC_POSE" or c.impedance_mode != "fixed" or not c.use_delta:
            raise RuntimeError("Unsupported runtime controller; calibration required")
        low, high = self.env.env.action_spec
        if len(low) != 7: raise RuntimeError("Expected exactly seven motor commands")
        return {"controller": c.name, "use_delta": bool(c.use_delta), "impedance_mode": c.impedance_mode,
            **{key: np.asarray(getattr(c,key)).tolist() for key in ("input_min", "input_max", "output_min", "output_max")},
            "action_low": np.asarray(low).tolist(), "action_high": np.asarray(high).tolist(),
            "gripper": type(robot.gripper).__name__, "gripper_speed": float(robot.gripper.speed),
            "frame": "world", "rotation": "axis-angle components in radians", "control_freq": self.config.get("control_freq",20)}

    def get_robot_state(self):
        if self._obs is None: raise RuntimeError("Environment has not been reset")
        return RobotState(position=np.asarray(self._obs["robot0_eef_pos"]).tolist(),
            orientation=np.asarray(self._obs["robot0_eef_quat"]).tolist(),
            gripper=np.asarray(self._obs["robot0_gripper_qpos"]).tolist(),
            gripper_target=np.asarray(self.env.robots[0].gripper.current_action).tolist())

    def get_observation(self):
        from robosuite import macros
        # robosuite 1.4 returns OpenGL bottom-up images by default. Convert only once.
        def camera(name):
            a = np.asarray(self._obs[name + "_image"], dtype=np.uint8)
            return np.ascontiguousarray(a[::-1] if macros.IMAGE_CONVENTION == "opengl" else a)
        primary = self.config.get("primary_camera", "agentview")
        wrist = self.config.get("wrist_camera", "robot0_eye_in_hand")
        if wrist and wrist + "_image" not in self._obs: wrist = None
        camera_info = {}
        for name in (primary, wrist):
            if not name: continue
            idx = self.env.sim.model.camera_name2id(name)
            camera_info[name] = {"position_world":self.env.sim.data.cam_xpos[idx].tolist(),
                "rotation_camera_to_world":self.env.sim.data.cam_xmat[idx].reshape(3,3).tolist(),
                "fovy_degrees":float(self.env.sim.model.cam_fovy[idx])}
        return Observation(rgb=camera(primary), wrist_rgb=camera(wrist) if wrist else None,
            robot_state=self.get_robot_state(), step=self.steps,
            metadata={"cameras":camera_info, "image_origin":"top-left", "coordinate_frame":"world"})

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all(): raise ValueError("Invalid continuous action")
        if np.any(action < np.array(self.action_spec["action_low"])) or np.any(action > np.array(self.action_spec["action_high"])):
            raise ValueError("Action outside runtime limits")
        self._obs, reward, done, info = self.env.step(action)
        self.steps += 1
        # LIBERO overrides done with success; explicitly respect configured horizon.
        truncated = self.steps + len(self.setup_records) >= self.config.get("horizon",2000)
        return self.get_observation(), float(reward), bool(done or truncated), {**info, "truncated": truncated}

    def check_success(self):
        return bool(self.env.check_success())

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

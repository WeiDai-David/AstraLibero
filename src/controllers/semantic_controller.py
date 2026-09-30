"""Pure encoding of world-frame OSC deltas, with no perception or planning."""
import numpy as np
from .base_controller import BaseSemanticController
from src.schemas.action import SemanticAction, RobotState

class ModeBasedSemanticController(BaseSemanticController):
    def __init__(self, config: dict, action_spec: dict):
        if config.get("granularity", 3) != 3:
            raise ValueError("Only direction+mode granularity=3 is implemented")
        self.config = config
        self.spec = action_spec
        if action_spec.get("controller") != "OSC_POSE" or not action_spec.get("use_delta"):
            raise ValueError("Requires verified fixed-impedance OSC_POSE delta controller")
        self.input_min = self._vector(action_spec["input_min"], 6)
        self.input_max = self._vector(action_spec["input_max"], 6)
        self.output_min = self._vector(action_spec["output_min"], 6)
        self.output_max = self._vector(action_spec["output_max"], 6)
        self.low = self._vector(action_spec["action_low"], 7)
        self.high = self._vector(action_spec["action_high"], 7)
        if np.any(self.input_max <= self.input_min) or np.any(self.output_max <= self.output_min):
            raise ValueError("Invalid controller scaling range")
        self.translation = self._modes(config["translation_magnitudes"])
        self.rotation = {k: np.deg2rad(v) for k, v in self._modes(config["rotation_degrees"]).items()}
        if "translation_scale" in config or "rotation_scale" in config:
            raise ValueError("Scaling is superseded by per-mode magnitudes and repeat counts")
        self.translation_repeat = self._repeats(config.get("translation_repeat", dict.fromkeys(self.translation,1)))
        self.rotation_repeat = self._repeats(config.get("rotation_repeat", dict.fromkeys(self.rotation,1)))
        for sizes, lo, hi in ((self.translation,self.output_min[:3],self.output_max[:3]),
                              (self.rotation,self.output_min[3:],self.output_max[3:])):
            if any(v > min(float(min(hi)),float(min(-lo))) + 1e-12 for v in sizes.values()):
                raise ValueError("Configured per-tick magnitude exceeds inspected OSC limits")
        self.gripper = config["gripper"]
        if set(self.gripper) != {"OPEN", "HOLD", "CLOSE"}:
            raise ValueError("Specify OPEN, HOLD and CLOSE")
        if self.gripper != {"OPEN": -1.0, "HOLD": 0.0, "CLOSE": 1.0} or action_spec.get("gripper") != "PandaGripper":
            raise ValueError("Only inspected PandaGripper incremental target mapping is supported")
        self.workspace = config.get("workspace")
        if self.workspace:
            lo = self._vector(self.workspace["min"], 3)
            hi = self._vector(self.workspace["max"], 3)
            if np.any(lo >= hi):
                raise ValueError("Invalid workspace bounds")
        for key in ("max_translation_norm", "max_rotation_norm"):
            if config.get(key) is not None and (not np.isfinite(config[key]) or config[key] <= 0):
                raise ValueError(f"{key} must be positive")

    @staticmethod
    def _vector(value, size):
        a = np.asarray(value, dtype=float)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f"Expected finite vector of length {size}")
        return a

    @staticmethod
    def _modes(values):
        if set(values) != {"COARSE", "NORMAL", "FINE"}:
            raise ValueError("All three modes must be configured")
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Magnitudes must be finite and positive")
        return {k: float(v) for k, v in values.items()}

    @staticmethod
    def _repeats(values):
        if set(values) != {"COARSE","NORMAL","FINE"} or any(type(v) is not int or v < 1 for v in values.values()):
            raise ValueError("Repeat counts must be positive integers for all modes")
        return dict(values)

    def repeat_counts(self, action, *, force_single=False):
        moving = any(v.value != "ZERO" for v in (action.x,action.y,action.z))
        rotating = any(v.value != "ZERO" for v in (action.roll,action.pitch,action.yaw))
        t = self.translation_repeat[action.precision.value] if moving else 0
        r = self.rotation_repeat[action.precision.value] if rotating else 0
        if force_single: t,r = min(t,1),min(r,1)
        return {"translation_repeat":t,"rotation_repeat":r,"repeat_count":max(t,r,1)}

    def subaction(self, action, index, *, force_single=False):
        counts = self.repeat_counts(action,force_single=force_single)
        if not 0 <= index < counts["repeat_count"]: raise ValueError("Substep out of range")
        data = action.model_dump(mode="json")
        if index >= counts["translation_repeat"]:
            for axis in ("x","y","z"): data[axis]="ZERO"
        if index >= counts["rotation_repeat"]:
            for axis in ("roll","pitch","yaw"): data[axis]="ZERO"
        if index > 0: data["gripper"]="HOLD"
        return SemanticAction.model_validate(data)

    def base_delta(self, action: SemanticAction) -> np.ndarray:
        sign = {"NEGATIVE": -1, "ZERO": 0, "POSITIVE": 1}
        axes = (action.x, action.y, action.z, action.roll, action.pitch, action.yaw)
        delta = np.array([sign[v.value] for v in axes], dtype=float)
        delta[:3] *= self.translation[action.precision.value]
        delta[3:] *= self.rotation[action.precision.value]
        return delta

    def physical_delta(self, action: SemanticAction, state: RobotState, *, translation_cap=None) -> np.ndarray:
        delta = self.base_delta(action)
        if translation_cap is not None:
            if not np.isfinite(translation_cap) or translation_cap <= 0:
                raise ValueError("translation_cap must be positive")
            norm = np.linalg.norm(delta[:3])
            if norm > translation_cap: delta[:3] *= translation_cap / norm
        for block, key in ((slice(0, 3), "max_translation_norm"), (slice(3, 6), "max_rotation_norm")):
            maximum = self.config.get(key)
            norm = np.linalg.norm(delta[block])
            if maximum and norm > maximum:
                delta[block] *= maximum / norm
        if self.workspace:
            position = np.asarray(state.position)
            lo, hi = np.array(self.workspace["min"]), np.array(self.workspace["max"])
            if np.any(position < lo) or np.any(position > hi):
                raise ValueError("Current end effector is outside configured workspace; stop for review")
            delta[:3] = np.clip(position + delta[:3], lo, hi) - position
        return np.clip(delta, self.output_min, self.output_max)

    def convert(self, semantic_action: SemanticAction, robot_state: RobotState, *, translation_cap=None) -> np.ndarray:
        physical = self.physical_delta(semantic_action, robot_state, translation_cap=translation_cap)
        # Inverse of robosuite Controller.scale_action, including asymmetric bounds.
        encoded = self.input_min + (physical - self.output_min) * (self.input_max - self.input_min) / (self.output_max - self.output_min)
        result = np.r_[encoded, self.gripper[semantic_action.gripper.value]]
        return np.clip(result, self.low, self.high)

    def describe(self, action, state, *, translation_cap=None):
        requested = self.base_delta(action)
        grip = self.gripper[action.gripper.value]
        unbounded = self.input_min + (requested-self.output_min)*(self.input_max-self.input_min)/(self.output_max-self.output_min)
        motor = self.convert(action,state,translation_cap=translation_cap)
        physical = self.output_min + (motor[:6]-self.input_min)*(self.output_max-self.output_min)/(self.input_max-self.input_min)
        return {"precision":action.precision.value,"base_action":np.r_[requested,grip].tolist(),
            "base_action_units":"meters, axis-angle radians, gripper command",
            "requested_physical_delta":requested.tolist(),"physical_target_delta":physical.tolist(),
            "requested_action":np.r_[unbounded,grip].tolist(),"clipped_action":motor.tolist(),
            "continuous_action":motor.tolist(),"final_continuous_action":motor.tolist(),
            "continuous_action_units":"runtime-normalized LIBERO action",
            "action_clipped":not np.allclose(requested,physical,atol=1e-12,rtol=0),
            "translation_cap_m":translation_cap}

SemanticController = ModeBasedSemanticController

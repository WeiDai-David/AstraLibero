"""Strict semantic commands; robot values must come from actual sensors."""
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Direction(str, Enum):
    NEGATIVE = "NEGATIVE"
    ZERO = "ZERO"
    POSITIVE = "POSITIVE"

class Mode(str, Enum):
    COARSE = "COARSE"
    NORMAL = "NORMAL"
    FINE = "FINE"

class Gripper(str, Enum):
    OPEN = "OPEN"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

class Phase(str, Enum):
    APPROACH = "APPROACH"
    ALIGN = "ALIGN"
    CLOSE = "CLOSE"
    VERIFY_GRASP = "VERIFY_GRASP"
    OPTIONAL_LIFT_TEST = "OPTIONAL_LIFT_TEST"
    RECOVER = "RECOVER"
    RECOVER_ALIGN = "RECOVER_ALIGN"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"

class GraspStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"

class SemanticAction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Mode = Mode.FINE
    precision: Mode = Mode.FINE
    phase: Phase = Phase.APPROACH
    grasp_status: GraspStatus = GraspStatus.NOT_APPLICABLE

    @model_validator(mode="before")
    @classmethod
    def legacy_mode(cls, data):
        # Legacy scripts/recordings retain mode. Live model output is checked separately.
        if isinstance(data, dict):
            data = dict(data)
            if "precision" not in data and "mode" in data: data["precision"] = data["mode"]
            if "mode" not in data and "precision" in data: data["mode"] = data["precision"]
            if "precision" in data: data["mode"] = data["precision"]
        return data

    @classmethod
    def from_model_json(cls, text):
        import json
        data = json.loads(text)
        required = {"precision","phase","grasp_status","x","y","z","roll","pitch","yaw","gripper"}
        if not isinstance(data, dict) or not required.issubset(data):
            raise ValueError("Live Astra output must include precision, phase, grasp_status and all action fields")
        return cls.model_validate(data)
    x: Direction
    y: Direction
    z: Direction
    roll: Direction
    pitch: Direction
    yaw: Direction
    gripper: Gripper
    reason: str | None = None

    @classmethod
    def noop(cls):
        return cls(mode="FINE", **{k: "ZERO" for k in ("x", "y", "z", "roll", "pitch", "yaw")}, gripper="HOLD")

class RobotState(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    position: list[float] = Field(min_length=3, max_length=3)
    orientation: list[float] = Field(min_length=4, max_length=4, description="Quaternion xyzw, world frame")
    gripper: list[float] = Field(min_length=1, description="Measured gripper joint positions")
    gripper_target: list[float] = Field(default_factory=list, description="Internal actuator target; not an object sensor")

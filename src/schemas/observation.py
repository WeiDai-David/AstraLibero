from typing import Any, Optional
from pydantic import BaseModel, Field
from .action import RobotState
class Observation(BaseModel):
    rgb: Any = None
    wrist_rgb: Optional[Any] = None
    robot_state: Optional[RobotState] = None
    step: int = 0
    metadata: dict = Field(default_factory=dict)

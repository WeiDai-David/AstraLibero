from pathlib import Path
import yaml
from src.agents.request_policy import RequestPolicy

def load_config(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict): raise ValueError("Configuration must be a YAML mapping")
    for name in ("environment", "controller", "agent", "rollout"):
        if not isinstance(data.get(name), dict): raise ValueError(f"Missing {name} configuration")
    for key in ("max_steps", "max_invalid_decisions"):
        if not isinstance(data["rollout"].get(key), int) or data["rollout"][key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if data["rollout"].get("history_length", -1) < 0: raise ValueError("History length cannot be negative")
    if data["environment"].get("control_freq", 0) <= 0: raise ValueError("Invalid control frequency")
    names = RequestPolicy.__dataclass_fields__
    RequestPolicy(**{k: v for k, v in data["agent"].items() if k in names})
    from src.agents.grasp_feedback import GraspFeedback
    GraspFeedback(data.get("grasp"))
    viewer=data.get("viewer",{})
    for key in ("enabled","show_status_window","show_camera_window"):
        if key in viewer and type(viewer[key]) is not bool: raise ValueError(key)
    n=viewer.get("refresh_every_steps",1)
    if type(n) is not int or n<1:raise ValueError("refresh_every_steps")
    return data

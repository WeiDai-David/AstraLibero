import pytest
from src.schemas.action import RobotState
from src.utils.config import load_config

@pytest.fixture
def config(): return load_config("configs/libero_spatial.yaml")

@pytest.fixture
def state(): return RobotState(position=[0,0,1],orientation=[0,0,0,1],gripper=[.02,-.02],gripper_target=[0,0])

@pytest.fixture
def spec():
    # Explicit test fixture matching inspected OSC defaults; production reads runtime.
    return {"controller":"OSC_POSE","use_delta":True,"impedance_mode":"fixed",
        "input_min":[-1.]*6,"input_max":[1.]*6,"output_min":[-.05]*3+[-.5]*3,
        "output_max":[.05]*3+[.5]*3,"action_low":[-1.]*7,"action_high":[1.]*7,
        "gripper":"PandaGripper","control_freq":20}

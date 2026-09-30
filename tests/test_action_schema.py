import pytest
from pydantic import ValidationError
from src.schemas.action import RobotState, SemanticAction

def test_noop(): assert SemanticAction.noop().gripper.value=="HOLD"

def test_missing_direction_rejected():
    a=SemanticAction.noop().model_dump(mode="json"); del a["z"]
    with pytest.raises(ValidationError): SemanticAction.model_validate(a)

def test_extra_and_unknown_rejected():
    a=SemanticAction.noop().model_dump(mode="json"); a["x"]="RIGHT"
    with pytest.raises(ValidationError): SemanticAction.model_validate(a)
    a=SemanticAction.noop().model_dump(mode="json"); a["guess"]=1
    with pytest.raises(ValidationError): SemanticAction.model_validate(a)

def test_no_invented_robot_state():
    with pytest.raises(ValidationError): RobotState()

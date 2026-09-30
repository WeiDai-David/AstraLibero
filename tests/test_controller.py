import copy
import numpy as np
import pytest
from src.controllers.semantic_controller import ModeBasedSemanticController
from src.schemas.action import SemanticAction

@pytest.mark.parametrize("mode,axis,sign,expected",[
    ("FINE","x","POSITIVE",.1),("NORMAL","z","NEGATIVE",-.4),
    ("COARSE","yaw","POSITIVE",np.deg2rad(12)/.5)])
def test_inverse_scaling(config,spec,state,mode,axis,sign,expected):
    ctrl=ModeBasedSemanticController(config["controller"],spec)
    a=SemanticAction.noop().model_dump(mode="json"); a.update(mode=mode,precision=mode); a[axis]=sign
    result=ctrl.convert(SemanticAction.model_validate(a),state)
    idx=("x","y","z","roll","pitch","yaw").index(axis)
    assert result[idx]==pytest.approx(expected)
    decoded=np.array(spec["output_min"])+(result[:6]+1)/2*(np.array(spec["output_max"])-spec["output_min"])
    assert np.allclose(decoded,ctrl.physical_delta(SemanticAction.model_validate(a),state))

def test_noop(config,spec,state):
    assert np.allclose(ModeBasedSemanticController(config["controller"],spec).convert(SemanticAction.noop(),state),0)

@pytest.mark.parametrize("command,value",[("OPEN",-1),("HOLD",0),("CLOSE",1)])
def test_gripper(config,spec,state,command,value):
    a=SemanticAction.noop().model_dump(mode="json"); a["gripper"]=command
    assert ModeBasedSemanticController(config["controller"],spec).convert(SemanticAction.model_validate(a),state)[6]==value

def test_workspace_and_clipping(config,spec,state):
    c=copy.deepcopy(config["controller"]); c["workspace"]={"min":[-.001,-.001,.999],"max":[.001,.001,1.001]}
    a=SemanticAction.noop().model_dump(mode="json"); a["x"]="POSITIVE"
    assert ModeBasedSemanticController(c,spec).convert(SemanticAction.model_validate(a),state)[0]==pytest.approx(.02)

def test_reject_unsupported(config,spec):
    spec["use_delta"]=False
    with pytest.raises(ValueError): ModeBasedSemanticController(config["controller"],spec)

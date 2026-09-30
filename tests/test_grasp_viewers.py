import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.agents.grasp_feedback import GraspFeedback,GraspRecoveryError
from src.schemas.action import SemanticAction
from src.utils.viewers import start_viewers
from scripts.status_viewer import render
from scripts.view_rollout import build_frame


def act(**fields):
    d=SemanticAction.noop().model_dump(mode='json');d.update(fields);return SemanticAction.model_validate(d)


def test_legacy_schema_and_strict_model():
    old=SemanticAction.noop().model_dump(mode='json')
    for name in ('precision','phase','grasp_status'):old.pop(name)
    assert SemanticAction.model_validate(old).precision.value=='FINE'
    with pytest.raises(ValueError):SemanticAction.from_model_json(json.dumps(old))
    new=dict(old,precision='COARSE',phase='APPROACH',grasp_status='NOT_APPLICABLE');new.pop('mode')
    assert SemanticAction.from_model_json(json.dumps(new)).mode.value=='COARSE'


def test_failed_without_formal_close_still_opens(state):
    f=GraspFeedback()
    a=f.select(act(grasp_status='FAILED',x='POSITIVE',gripper='CLOSE'),state)
    assert a.gripper.value=='OPEN' and a.x.value=='ZERO'


def test_release_placement_can_continue(state):
    f=GraspFeedback();f.after_execution(act(gripper='CLOSE'),state)
    a=f.select(act(grasp_status='SUCCESS',phase='PLACE',gripper='OPEN'),state)
    f.after_execution(a,state)
    a=f.select(act(phase='PLACE',gripper='OPEN'),state)
    assert a.gripper.value=='OPEN' and not f.verified


def test_unknown_verification_is_bounded(state):
    f=GraspFeedback({'max_verification_steps':1,'max_grasp_retries':0})
    f.after_execution(act(gripper='CLOSE'),state)
    f.select(act(grasp_status='UNCERTAIN'),state)
    with pytest.raises(GraspRecoveryError):f.select(act(grasp_status='UNCERTAIN'),state)


def test_viewer_without_wrist_and_headless_launch(tmp_path,monkeypatch):
    from PIL import Image
    p=tmp_path/'failure/task_0/episode_0000_fixture';(p/'images').mkdir(parents=True)
    (p/'metadata.json').write_text(json.dumps({'status':'complete','task_instruction':'test'}))
    Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(p/'images/primary.png')
    (p/'initial_observation.json').write_text(json.dumps({'images':{'primary':'images/primary.png'}}))
    (p/'trajectory.jsonl').write_text('')
    frame=build_frame(tmp_path,p.name)[0]
    assert frame.shape[1]==960  # Missing wrist has its own labelled panel.
    assert render(tmp_path,p.name).shape[1]==1180
    import src.utils.viewers as viewers
    monkeypatch.setattr(viewers.subprocess,'Popen',lambda *a,**k: (_ for _ in ()).throw(OSError('no desktop')))
    log=SimpleNamespace(root=tmp_path,dir=p,metadata={},_write_metadata=lambda:None)
    assert start_viewers(log,{'enabled':False})==[]
    assert start_viewers(log,{'enabled':True})==[]
    assert len(log.metadata['viewer_warnings'])==2


def test_live_status_permission_error_is_nonfatal(tmp_path,monkeypatch):
    from src.trajectory.logger import TrajectoryLogger
    log=TrajectoryLogger(tmp_path,video=False)
    original=Path.replace
    def deny_live(self,target):
        if self.name=='live_status.tmp':raise PermissionError('reader lock')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',deny_live)
    log.live_status(step=1)
    assert json.loads((log.dir/'live_status.json').read_text())['step']==1
    log.finalize(termination='test')

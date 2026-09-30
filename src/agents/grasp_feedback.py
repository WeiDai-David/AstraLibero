"""Feedback gate, not a planner: all grasp judgements originate in Astra."""
import math
from src.schemas.action import SemanticAction

class GraspRecoveryError(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.failure_reason=reason

class GraspFeedback:
    def __init__(self, config=None):
        self.config={"enable_lift_test":True,"max_grasp_retries":3,"lift_test_translation_m":.001,
            "open_width_threshold_m":.07,"max_open_steps":30,"max_verification_steps":20,**(config or {})}
        for k in ("max_grasp_retries","max_open_steps","max_verification_steps"):
            if type(self.config[k]) is not int or self.config[k] < (0 if k=="max_grasp_retries" else 1): raise ValueError(k)
        for k in ("lift_test_translation_m","open_width_threshold_m"):
            if not math.isfinite(self.config[k]) or self.config[k]<=0: raise ValueError(k)
        if type(self.config['enable_lift_test']) is not bool: raise ValueError('enable_lift_test')
        self.state="APPROACH"; self.attempt_id=0;self.retries=0;self.successful_attempt=None
        self.verified=False;self.lift_used=False;self.verify_steps=0;self.open_steps=0
        self.status="NOT_APPLICABLE";self.override=None;self.cap=None;self.single=False
        self.failed_attempt_ids=set()

    def context(self):
        return {"recovery_state":self.state,"grasp_attempt_id":self.attempt_id,
            "grasp_retry_count":self.retries,"grasp_status":self.status,
            "verified":self.verified,"max_grasp_retries":self.config['max_grasp_retries'],
            "must_verify_now":self.state=="VERIFY_GRASP","lift_test_already_used":self.lift_used,
            "failed_attempt_ids":sorted(self.failed_attempt_ids),
            "current_attempt_failure_recorded":self.attempt_id in self.failed_attempt_ids,
            "gripper_semantics":"CLOSE advances target for ONE tick. HOLD preserves current target, including a partially closed target; it does not continue closing toward full closure. If still aligned and more closure is needed, choose CLOSE again after inspecting this observation. Never infer success from target or joint width.",
            "instruction":"Judge only the CURRENT images. CLOSE is not success. Failed grasp must reopen before alignment."}

    def summary(self):
        return {"total_grasp_attempts":self.attempt_id,"successful_grasp_attempt":self.successful_attempt,
            "grasp_recovery_count":self.retries,"failed_grasp_attempt_ids":sorted(self.failed_attempt_ids)}

    def safe_action(self, gripper="HOLD", phase="VERIFY_GRASP", z="ZERO"):
        d=SemanticAction.noop().model_dump(mode='json')
        d.update(precision='FINE',mode='FINE',phase=phase,grasp_status=self.status,gripper=gripper,z=z)
        return SemanticAction.model_validate(d)

    def fail(self):
        self.verified=False
        first_failure = self.attempt_id not in self.failed_attempt_ids
        if first_failure:
            if self.retries >= self.config['max_grasp_retries']:
                raise GraspRecoveryError('grasp_retry_exceeded')
            self.failed_attempt_ids.add(self.attempt_id)
            self.retries+=1
            self.open_steps=0
            self.override='failed_grasp_open_before_realign'
        else:
            # A new judgement about the SAME failed attempt is not a new retry.
            # Keep opening/verification budgets and lift-test history intact.
            if self.open_steps>=self.config['max_open_steps']:
                raise GraspRecoveryError('grasp_open_timeout')
            self.override='same_failed_attempt_reopen_without_new_retry'
        self.state='RECOVER';self.single=True
        return self.safe_action('OPEN','RECOVER')

    def select(self, proposed, robot_state):
        self.override=None;self.cap=None;self.single=False
        self.status=proposed.grasp_status.value
        if proposed.gripper.value=='CLOSE': self.single=True
        if self.state=='RECOVER':
            self.override='reopen_until_finger_clearance';self.single=True
            if self.open_steps>=self.config['max_open_steps']: raise GraspRecoveryError('grasp_open_timeout')
            return self.safe_action('OPEN','RECOVER')
        if self.state=='VERIFY_GRASP':
            self.verify_steps+=1
            if self.status=='FAILED': return self.fail()
            if self.status=='SUCCESS':
                self.verified=True;self.successful_attempt=self.attempt_id;self.verify_steps=0
            else:
                self.status='UNCERTAIN'
                if self.verify_steps>self.config['max_verification_steps']:return self.fail()
                self.single=True
                # Allow incremental finger closure, but no lateral movement without verification.
                if proposed.gripper.value=='CLOSE':
                    self.override='unverified_close_without_translation'
                    return self.safe_action('CLOSE')
                if proposed.phase.value=='OPTIONAL_LIFT_TEST' and self.config['enable_lift_test'] and not self.lift_used:
                    self.override='bounded_optional_lift_test';self.cap=self.config['lift_test_translation_m']
                    return self.safe_action('HOLD','OPTIONAL_LIFT_TEST','POSITIVE')
                if proposed.gripper.value=='OPEN':return self.fail()
                self.override='await_grasp_verification';return self.safe_action()
        elif self.status=='FAILED':return self.fail()
        if proposed.phase.value in ('LIFT','TRANSPORT','PLACE') and not self.verified and not (proposed.phase.value=='PLACE' and self.state=='PLACE' and self.successful_attempt is not None):
            self.single=True;self.override='transport_blocked_without_verified_grasp'
            return self.safe_action(phase='ALIGN')
        return proposed

    def after_execution(self, action, state):
        # Called once after the semantic batch, never infer a grasp from joint width.
        if self.state=='RECOVER':
            self.open_steps+=1
            if sum(abs(v) for v in state.gripper)>=self.config['open_width_threshold_m']:
                self.state='RECOVER_ALIGN'
            return
        if action.gripper.value=='CLOSE':
            if self.state!='VERIFY_GRASP':
                self.attempt_id+=1;self.verify_steps=0;self.lift_used=False
            self.state='VERIFY_GRASP';self.verified=False
        elif action.phase.value=='OPTIONAL_LIFT_TEST':
            self.lift_used=True;self.state='VERIFY_GRASP'
        elif self.state=='VERIFY_GRASP' and not self.verified:
            pass
        elif self.verified:
            if action.gripper.value=='OPEN': self.verified=False;self.state='PLACE'
            else:self.state='RECOVER_ALIGN' if self.state=='RECOVER_ALIGN' else action.phase.value
        else:self.state='RECOVER_ALIGN' if self.state=='RECOVER_ALIGN' else action.phase.value

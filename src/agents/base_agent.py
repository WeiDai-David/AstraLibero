from abc import ABC, abstractmethod
from src.schemas.action import SemanticAction, RobotState
class BaseAgent(ABC):
 @abstractmethod
 def act(self, task_instruction, observation, robot_state, history=None)->SemanticAction: ...

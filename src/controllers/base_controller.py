from abc import ABC, abstractmethod
import numpy as np
from src.schemas.action import SemanticAction, RobotState
class BaseSemanticController(ABC):
 @abstractmethod
 def convert(self, semantic_action: SemanticAction, robot_state: RobotState) -> np.ndarray: ...

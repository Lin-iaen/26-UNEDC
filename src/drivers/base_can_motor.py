from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MotorFeedback:
    position_deg: float
    speed_rpm: float
    current_a: float


class BaseCANMotor(ABC):

    @abstractmethod
    def connect(self, channel: str, node_id: int) -> None:
        """Connect to the motor over CAN bus.

        Args:
            channel: CAN interface name (e.g. "can0", "vcan0").
            node_id: CAN node ID of the target motor (1-127).
        """

    @abstractmethod
    def send_position_cmd(self, target_deg: float, velocity_limit: float = 0.0) -> None:
        """Send a position command to the motor.

        Args:
            target_deg: Target position in degrees.
            velocity_limit: Max velocity in rpm. 0 means driver default.
        """

    @abstractmethod
    def get_feedback(self) -> MotorFeedback:
        """Read and return the latest motor feedback.

        Returns:
            MotorFeedback: Current position, speed, and current draw.
        """

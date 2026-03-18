"""
gait.py
x2_ultra 双足机器人步态生成器
"""

from typing import Dict, Tuple
import jax.numpy as jp

from . import x2_constants as constants


class GaitGenerator:
    """双足步态生成器，适配 X2 Ultra 人形机器人"""

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or constants.X2UltraConstants.GAIT_CONFIG
        self.constants = constants.X2UltraConstants()

        # 步态参数
        self.gait_type = self.config["gait_type"]
        self.stride_length = self.config.get("stride_length", 0.3)
        self.step_height = self.config.get("step_height", 0.1)
        self.cadence = self.config.get("cadence", 1.5)  # 步频（步/秒）
        self.duty_factor = self.config.get("duty_factor", 0.6)
        self.double_support_ratio = self.config.get("double_support_ratio", 0.2)
        self.phase_offset = self.config.get("phase_offset", 0.0)

        self.gait_period = 1.0 / self.cadence
        self.phase_speed = 2 * jp.pi / self.gait_period

        self.phase = 0.0
        self.phase_offsets = {
            "walk": jp.array([0.0, 0.5]),  # 左腿与右腿相位差 π
            "run": jp.array([0.0, 0.0]),   # 同相
            "jump": jp.array([0.0, 0.0]),  # 同相
            "stand": jp.array([0.0, 0.5]), # 站立时反相保持平衡
        }

    def update(self, dt: float, command: jp.ndarray) -> Tuple[float, jp.ndarray, jp.ndarray]:
        """更新步态相位并返回每条腿相位和接触状态"""
        forward_speed = float(command[0])
        speed_factor = 1.0 + 0.3 * forward_speed
        cadence = self.cadence * speed_factor
        self.gait_period = 1.0 / cadence
        self.phase_speed = 2 * jp.pi / self.gait_period

        self.phase = (self.phase + self.phase_speed * dt) % (2 * jp.pi)
        offsets = self.phase_offsets.get(self.gait_type, self.phase_offsets["walk"])
        leg_phases = (self.phase + offsets * 2 * jp.pi + self.phase_offset) % (2 * jp.pi)

        contact_state = jp.zeros(2)
        for idx in range(2):
            phase = leg_phases[idx]
            if self.gait_type == "walk":
                # 分别处理单支撑、双支撑、摆动
                stance_end = jp.pi * self.duty_factor
                double_support_end = jp.pi * (self.duty_factor + self.double_support_ratio)
                contact = jp.where(
                    phase < stance_end,
                    1.0,
                    jp.where(phase < double_support_end, 0.5, 0.0),
                )
            else:
                # run/jump 同相，phase<π 为支撑
                contact = jp.where(phase < jp.pi, 1.0, 0.0)
            contact_state = contact_state.at[idx].set(contact)

        return self.phase, leg_phases, contact_state

    def get_swing_trajectory(
        self,
        leg_idx: int,
        phase: float,
        step_length: float = None,
        step_height: float = None,
    ) -> Tuple[jp.ndarray, jp.ndarray]:
        """生成双足摆动期轨迹"""
        if leg_idx not in (0, 1):
            raise ValueError("双足机器人只有两条腿，leg_idx 必须是 0（左）或 1（右）")

        step_length = self.stride_length if step_length is None else step_length
        step_height = self.step_height if step_height is None else step_height

        norm_phase = (phase % (2 * jp.pi)) / (2 * jp.pi)

        if norm_phase >= 0.5:
            return jp.zeros(3), jp.zeros(3)

        t = norm_phase * 2  # 映射到 [0,1]
        x = step_length * (t - 0.5)
        z = 4 * step_height * t * (1 - t)
        y_offset = 0.05 if leg_idx == 0 else -0.05

        position = jp.array([x, y_offset, z])
        dx = step_length
        dz = 4 * step_height * (1 - 2 * t)
        velocity = jp.array([dx, 0.0, dz])

        return position, velocity

    def get_stance_trajectory(
        self,
        leg_idx: int,
        phase: float,
        body_velocity: jp.ndarray,
    ) -> Tuple[jp.ndarray, jp.ndarray]:
        """生成双足支撑期轨迹"""
        if leg_idx not in (0, 1):
            raise ValueError("双足机器人只有两条腿，leg_idx 必须是 0（左）或 1（右）")

        y_offset = 0.07135 if leg_idx == 0 else -0.07135
        position = jp.array([0.0, y_offset, -0.68])  # 默认脚在身体下方
        velocity = -body_velocity * 0.1  # 缓和补偿

        return position, velocity
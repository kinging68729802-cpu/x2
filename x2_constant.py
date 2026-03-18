"""
x2_constants.py
x2_ultra机器人常量定义
"""

import jax.numpy as jp
from flax import struct


@struct.dataclass
class X2UltraConstants:
    """x2_ultra机器人常量"""
    
    # 机器人名称
    ROBOT_NAME: str = "x2_ultra"
    
    # 时间参数
    CONTROL_FREQUENCY: float = 50.0  # Hz
    SIMULATION_FREQUENCY: float = 1000.0  # Hz
    CONTROL_SUBSTEPS: int = 20
    DT: float = 0.001  # 模拟时间步长
    
    # 身体部件ID（根据XML文件）
    BODY_IDS = struct.field(
        default_factory=lambda: {
            "pelvis": 0,
            "torso": 1,
            "left_hip_pitch": 2,
            "left_hip_roll": 3,
            "left_hip_yaw": 4,
            "left_knee": 5,
            "left_ankle_pitch": 6,
            "left_ankle_roll": 7,
            "right_hip_pitch": 8,
            "right_hip_roll": 9,
            "right_hip_yaw": 10,
            "right_knee": 11,
            "right_ankle_pitch": 12,
            "right_ankle_roll": 13,
        }
    )
    
    # 关节ID映射
    JOINT_IDS = struct.field(
        default_factory=lambda: {
            "left_hip_pitch": 0,
            "left_hip_roll": 1,
            "left_hip_yaw": 2,
            "left_knee": 3,
            "left_ankle_pitch": 4,
            "left_ankle_roll": 5,
            "right_hip_pitch": 6,
            "right_hip_roll": 7,
            "right_hip_yaw": 8,
            "right_knee": 9,
            "right_ankle_pitch": 10,
            "right_ankle_roll": 11,
        }
    )
    
    # 腿部配置
    LEG_CONFIG = struct.field(
        default_factory=lambda: {
            "left_leg": [0, 1, 2, 3, 4, 5],  # 左腿关节索引
            "right_leg": [6, 7, 8, 9, 10, 11],  # 右腿关节索引
            "num_legs": 2,
            "joints_per_leg": 6,
        }
    )
    
    # 默认关节位置（站立姿态）
    DEFAULT_JOINT_POSITIONS = struct.field(
        default_factory=lambda: jp.array([
            0.0,   # left_hip_pitch
            0.0,   # left_hip_roll
            0.0,   # left_hip_yaw
            0.2,   # left_knee (轻微弯曲)
            0.0,   # left_ankle_pitch
            0.0,   # left_ankle_roll
            0.0,   # right_hip_pitch
            0.0,   # right_hip_roll
            0.0,   # right_hip_yaw
            0.2,   # right_knee (轻微弯曲)
            0.0,   # right_ankle_pitch
            0.0,   # right_ankle_roll
        ])
    )
    
    # 命令限制
    COMMAND_LIMITS = struct.field(
        default_factory=lambda: {
            "forward_speed": (0.0, 1.0),      # m/s
            "lateral_speed": (-0.5, 0.5),     # m/s
            "turning_rate": (-1.0, 1.0),      # rad/s
            "body_height": (0.6, 0.9),        # m
        }
    )
    
    # 奖励权重
    REWARD_WEIGHTS = struct.field(
        default_factory=lambda: {
            "forward_velocity": 1.5,
            "lateral_velocity": 0.5,
            "heading_tracking": 0.5,
            "upright": 1.0,
            "energy_efficiency": -0.005,
            "smoothness": -0.001,
            "joint_limits": -0.1,
            "foot_clearance": 0.05,
            "foot_slip": -0.02,
            "survival": 0.1,
        }
    )
    
    # 观测配置
    OBSERVATION_CONFIG = struct.field(
        default_factory=lambda: {
            "include_joint_pos": True,
            "include_joint_vel": True,
            "include_body_lin_vel": True,
            "include_body_ang_vel": True,
            "include_body_height": True,
            "include_gravity_vector": True,
            "include_command": True,
            "include_phase": True,
            "include_last_action": True,
            "history_length": 3,
        }
    )
    
    # 动作配置
    ACTION_CONFIG = struct.field(
        default_factory=lambda: {
            "action_type": "torque",  # torque, position, pd_target
            "kp": 100.0,
            "kd": 10.0,
            "clip_range": (-1.0, 1.0),
            "action_scale": 100.0, 
        }
    )
    
    # 步态配置
    GAIT_CONFIG = struct.field(
        default_factory=lambda: {
            "gait_type": "walk",
            "stride_length": 0.3,
            "step_height": 0.08,
            "cadence": 1.5, # 步频 (步/秒)
            "duty_factor": 0.6,
            "phase_offset": 0.0,
        }
    )
    
    # 域随机化配置
    RANDOMIZATION_CONFIG = struct.field(
        default_factory=lambda: {
            "enabled": True,
            "friction_range": (0.5, 1.5),
            "motor_strength_range": (0.8, 1.2),
            "body_mass_range": (0.9, 1.1),
            "ground_height_range": (-0.05, 0.05),
            "joint_damping_range": (0.8, 1.2),
            "sensor_noise_range": (0.0, 0.01),
        }
    )
    
    # 终止条件
    TERMINATION_CONFIG = struct.field(
        default_factory=lambda: {
            "max_episode_length": 1000,
            "min_body_height": 0.3,
            "max_body_tilt": 0.8,  # rad
            "max_fall_velocity": 2.0,  # m/s
            "max_joint_limit_violation": 5,
        }
    )
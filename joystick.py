from typing import Dict, Tuple
import jax
import jax.numpy as jp
from flax import struct

from . import x2_constants as constants


@struct.dataclass
class JoystickConfig:
    """Joystick配置"""
    
    # 命令平滑参数
    smoothing_alpha: float = 0.5
    
    # 奖励权重
    reward_weights: Dict[str, float] = struct.field(
        default_factory=lambda: constants.X2UltraConstants.REWARD_WEIGHTS
    )
    
    # 命令更新频率
    command_update_frequency: float = 10.0  # Hz


class JoystickController:
    """Joystick控制器"""
    
    def __init__(self, config: JoystickConfig = None):
        self.config = config or JoystickConfig()
        self.constants = constants.X2UltraConstants()
        
        # 命令滤波器状态
        self.last_filtered_command = jp.zeros(4)
    
    def generate_command(self, rng: jp.ndarray, current_command: jp.ndarray) -> jp.ndarray:
        """生成随机命令"""
        rngs = jax.random.split(rng, 4)
        
        limits = self.constants.COMMAND_LIMITS
        
        # 生成随机命令
        raw_command = jp.array([
            jax.random.uniform(rngs[0], minval=limits["forward_speed"][0], maxval=limits["forward_speed"][1]),
            jax.random.uniform(rngs[1], minval=limits["lateral_speed"][0], maxval=limits["lateral_speed"][1]),
            jax.random.uniform(rngs[2], minval=limits["turning_rate"][0], maxval=limits["turning_rate"][1]),
            jax.random.uniform(rngs[3], minval=limits["body_height"][0], maxval=limits["body_height"][1]),
        ])
        
        # 应用低通滤波
        filtered_command = (
            self.config.smoothing_alpha * raw_command + 
            (1 - self.config.smoothing_alpha) * self.last_filtered_command
        )
        
        self.last_filtered_command = filtered_command
        
        return filtered_command
    
    def compute_rewards(self, state, action, physics) -> Dict[str, float]:
        """计算所有奖励组件"""
        rewards = {}
        
        # 1. 速度跟踪奖励
        velocity_rewards = self._compute_velocity_rewards(state, physics)
        rewards.update(velocity_rewards)
        
        # 2. 直立奖励
        upright_reward = self._compute_upright_reward(physics)
        rewards["upright"] = upright_reward
        
        # 3. 能量效率惩罚
        energy_reward = self._compute_energy_reward(action, physics)
        rewards["energy_efficiency"] = energy_reward
        
        # 4. 动作平滑度惩罚
        if hasattr(state, 'action_history') and state.action_history.shape[0] > 1:
            smoothness_reward = self._compute_smoothness_reward(action, state.action_history[1])
            rewards["smoothness"] = smoothness_reward
        
        # 5. 关节限位惩罚
        joint_limit_reward = self._compute_joint_limit_reward(physics)
        rewards["joint_limits"] = joint_limit_reward
        
        # 6. 生存奖励
        rewards["survival"] = self.config.reward_weights["survival"]
        
        return rewards
    
    def _compute_velocity_rewards(self, state, physics) -> Dict[str, float]:
        """计算速度相关奖励"""
        rewards = {}
        
        # 期望速度
        desired_forward = state.command[0]
        desired_lateral = state.command[1]
        desired_turning = state.command[2]
        
        # 实际速度
        body_lin_vel = physics.cvel[1, 3:] # 线速度
        body_ang_vel = physics.cvel[1, :3] # 角速度
        
        # 前进速度跟踪
        forward_error = jp.abs(body_lin_vel[0] - desired_forward)
        forward_reward = self.config.reward_weights["forward_velocity"] * -forward_error
        rewards["forward_velocity"] = forward_reward
        
        # 侧向速度跟踪
        lateral_error = jp.abs(body_lin_vel[1] - desired_lateral)
        lateral_reward = self.config.reward_weights["lateral_velocity"] * -lateral_error
        rewards["lateral_velocity"] = lateral_reward
        
        # 转向跟踪
        turning_error = jp.abs(body_ang_vel[2] - desired_turning)
        turning_reward = self.config.reward_weights["heading_tracking"] * -turning_error
        rewards["heading_tracking"] = turning_reward
        
        return rewards
    
    def _compute_upright_reward(self, physics) -> float:
        """计算直立奖励"""
        body_quat = physics.xquat[1]
        upright_reward = self.config.reward_weights["upright"] * jp.abs(body_quat[0])
        return float(upright_reward)
    
    def _compute_energy_reward(self, action, physics) -> float:
        """计算能量效率惩罚"""
        joint_vel = physics.qvel[6:]  # 排除自由关节速度
        energy = jp.sum(jp.abs(action * joint_vel))
        energy_reward = self.config.reward_weights["energy_efficiency"] * energy
        return float(energy_reward)
    
    def _compute_smoothness_reward(self, current_action, last_action) -> float:
        """计算动作平滑度惩罚"""
        action_diff = jp.sum(jp.abs(current_action - last_action))
        smoothness_reward = self.config.reward_weights["smoothness"] * -action_diff
        return float(smoothness_reward)
    
    def _compute_joint_limit_reward(self, physics) -> float:
        """计算关节限位惩罚"""
        # 简化实现
        joint_pos = physics.qpos[7:]
        joint_limit_penalty = 0.0
        
        # 检查关节是否接近限位
        for i in range(len(joint_pos)):
            if joint_pos[i] < -1.5 or joint_pos[i] > 1.5:  # 简化限位检查
                joint_limit_penalty += 1.0
        
        joint_limit_reward = self.config.reward_weights["joint_limits"] * -joint_limit_penalty
        return float(joint_limit_reward)
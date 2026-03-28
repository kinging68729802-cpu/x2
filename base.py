"""
base_functional.py
函数式 X2 Ultra 环境，兼容 JAX JIT/VMAP
"""

from typing import Any, Dict, Tuple
from functools import partial
import numpy as np
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jp
from flax import struct
# 修复导入
try:
    from mujoco.mjx._src import math as mjx_math
except ImportError:
    # 备用方案
    mjx_math = mjx._src.math
import x2_constant as constants
@struct.dataclass
class X2UltraState:
    """环境状态 - 与 __init__.py 导入名匹配"""
    physics_data: Any  # mjx.Data
    time: jp.ndarray
    steps: jp.ndarray
    command: jp.ndarray      # [4]
    last_action: jp.ndarray  # [action_dim]
class X2UltraBaseEnv:
    """
    与 __init__.py 导入名匹配的类
    完全函数式设计，支持 JIT/VMAP
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.const = constants.X2UltraConstants()
        
        # 加载模型
        xml_path = self.config.get('model_path', 'x2_ultra.xml')
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._model_jx = mjx.put_model(self._model)
        
        # 维度
        self.action_dim = self._model.nu  # 29个执行器
        self.observation_dim = self._compute_obs_dim()
        
        print(f"[X2Ultra] obs_dim={self.observation_dim}, action_dim={self.action_dim}")
        
        # 预编译 JIT 函数（关键优化）
        self._reset_jit = jax.jit(self._reset_impl, static_argnums=(0,))
        self._step_jit = jax.jit(self._step_impl, static_argnums=(0,))
    
    def _compute_obs_dim(self) -> int:
        """计算观测维度"""
        cfg = self.const.OBSERVATION_CONFIG
        n_joints = self.action_dim  # 使用执行器数作为关节数
        
        dim = 0
        if cfg["include_joint_pos"]:
            dim += n_joints  # 实际控制关节位置
        if cfg["include_joint_vel"]:
            dim += n_joints
        
        # IMU
        if cfg["include_body_lin_vel"]:
            dim += 3
        if cfg["include_body_ang_vel"]:
            dim += 3
        if cfg["include_body_height"]:
            dim += 1
        if cfg["include_gravity_vector"]:
            dim += 3
        
        # 命令和动作
        if cfg["include_command"]:
            dim += 4
        if cfg["include_phase"]:
            dim += 1
        if cfg["include_last_action"]:
            dim += self.action_dim
        
        return dim
    
    # ========================================================================
    # 公开 API（非 JIT，用于 Python 调用）
    # ========================================================================
    
    def reset(self, rng: jp.ndarray) -> Tuple[X2UltraState, jp.ndarray]:
        return self._reset_jit(rng)
    
    def step(self, state: X2UltraState, action: jp.ndarray, rng: jp.ndarray):
        return self._step_jit(state, action, rng)
    
    # ========================================================================
    # JIT 实现（内部使用）
    # ========================================================================
    
    def _reset_impl(self, rng: jp.ndarray):
        """JIT实现"""
        data = mjx.make_data(self._model_jx)
        
        # 设置初始高度
        qpos = data.qpos.at[2].set(0.68)
        
        # [关键修复] 扩展默认位置到29个关节
        default_12 = self.const.DEFAULT_JOINT_POSITIONS  # 12个值
        # 扩展到29个：腿部12个 + 其他置0
        full_default = jp.zeros(self.action_dim)
        full_default = full_default.at[:12].set(default_12)
        # 设置腰部和头部为合理值
        if self.action_dim > 12:
            full_default = full_default.at[12:15].set(jp.array([0.0, 0.0, 0.0]))  # 腰部
        if self.action_dim > 15:
            full_default = full_default.at[15:17].set(jp.array([0.0, 0.0]))  # 头部
        # 手臂自然下垂
        if self.action_dim > 17:
            full_default = full_default.at[17:29].set(jp.zeros(12))
        
        # 设置关节位置
        qpos = qpos.at[7:7+self.action_dim].set(full_default)
        data = data.replace(qpos=qpos)
        
        # 随机命令
        rng, cmd_rng = jax.random.split(rng)
        cmd = self._sample_command(cmd_rng)
        
        state = X2UltraState(
            physics_data=data,
            time=jp.array(0.0),
            steps=jp.array(0),
            command=cmd,
            last_action=jp.zeros(self.action_dim)
        )
        
        return state, self._get_obs(state)
    
    def _step_impl(
        self,
        state: X2UltraState,
        action: jp.ndarray,
        rng: jp.ndarray
    ):
        """JIT 兼容的 step"""
        # 动作限幅 [-1, 1] 然后缩放
        action = jp.clip(action, -1.0, 1.0)
        ctrl = action * self.const.ACTION_CONFIG["action_scale"]
        
        # 模拟
        data = state.physics_data
        
        def sim_step(d, _):
            d = d.replace(ctrl=ctrl)
            return mjx.step(self._model_jx, d), None
        
        data, _ = jax.lax.scan(sim_step, data, None, 
                               length=self.const.CONTROL_SUBSTEPS)
        
        # 更新
        dt = self.const.CONTROL_SUBSTEPS * self._model.opt.timestep
        new_time = state.time + dt
        new_steps = state.steps + 1
        
        # 命令更新（每 50 步 = 1秒）
        rng, cmd_rng = jax.random.split(rng)
        new_cmd = jax.lax.cond(
            new_steps % 50 == 0,
            lambda _: self._sample_command(cmd_rng),
            lambda _: state.command,
            None
        )
        
        new_state = X2UltraState(
            physics_data=data,
            time=new_time,
            steps=new_steps,
            command=new_cmd,
            last_action=action
        )
        
        obs = self._get_obs(new_state)
        reward = self._compute_reward(new_state, action)
        done = self._check_termination(new_state)
        truncated = new_steps >= self.const.TERMINATION_CONFIG["max_episode_length"]
        
        info = {"height": data.xpos[1, 2]}
        
        return new_state, obs, reward, done, truncated, info
    
    # ========================================================================
    # 辅助函数
    # ========================================================================
    
    def _sample_command(self, rng: jp.ndarray) -> jp.ndarray:
        """随机命令"""
        lim = self.const.COMMAND_LIMITS
        rngs = jax.random.split(rng, 4)
        return jp.array([
            jax.random.uniform(rngs[0], lim["forward_speed"][0], lim["forward_speed"][1]),
            jax.random.uniform(rngs[1], lim["lateral_speed"][0], lim["lateral_speed"][1]),
            jax.random.uniform(rngs[2], lim["turning_rate"][0], lim["turning_rate"][1]),
            jax.random.uniform(rngs[3], lim["body_height"][0], lim["body_height"][1]),
        ])
    
    def _get_obs(self, state: X2UltraState) -> jp.ndarray:
        """构建观测"""
        parts = []
        d = state.physics_data
        cfg = self.const.OBSERVATION_CONFIG
        
        # [关键修复] 正确提取29个关节的状态
        if cfg["include_joint_pos"]:
            # qpos: [7自由关节 + n_joints]，取后29个
            joint_pos = d.qpos[7:7+self.action_dim]
            parts.append(joint_pos)
        
        if cfg["include_joint_vel"]:
            # qvel: [6自由关节速度 + n_joints速度]
            joint_vel = d.qvel[6:6+self.action_dim]
            parts.append(joint_vel)
        
        # IMU数据
        if cfg["include_body_lin_vel"]:
            parts.append(d.cvel[1, 3:6])
        if cfg["include_body_ang_vel"]:
            parts.append(d.cvel[1, 0:3])
        if cfg["include_body_height"]:
            parts.append(jp.array([d.xpos[1, 2]]))
        if cfg["include_gravity_vector"]:
            # [关键修复] 手动实现四元数旋转
            quat = d.xquat[1]
            gravity = self._quat_rotate(quat, jp.array([0., 0., -1.]))
            parts.append(gravity)
        
        # 命令和相位
        if cfg["include_command"]:
            parts.append(state.command)
        if cfg["include_phase"]:
            phase = jp.sin(2 * jp.pi * state.time * 1.5)
            parts.append(jp.array([phase]))
        if cfg["include_last_action"]:
            parts.append(state.last_action)
        
        return jp.concatenate(parts)
    
    def _quat_rotate(self, q: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
        """手动实现四元数旋转避免私有 API"""
       # q = [w, x, y, z]
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # 计算 q * v * q^-1
        # 纯虚四元数
        qv = jp.array([0., v[0], v[1], v[2]])
        
        # 四元数乘法
        t = jp.array([
            -x*qv[1] - y*qv[2] - z*qv[3],
            w*qv[1] + y*qv[3] - z*qv[2],
            w*qv[2] - x*qv[3] + z*qv[1],
            w*qv[3] + x*qv[2] - y*qv[1]
        ])
        
        result = jp.array([
            t[1]*w - t[0]*x - t[3]*y + t[2]*z,
            t[2]*w + t[3]*x - t[0]*y - t[1]*z,
            t[3]*w - t[2]*x + t[1]*y - t[0]*z
        ])
        
        return result
    
    def _compute_reward(self, state: X2UltraState, action: jp.ndarray) -> jp.ndarray:
        """丰富奖励函数"""
        d = state.physics_data
        w = self.const.REWARD_WEIGHTS
        
        # 速度跟踪（主要奖励）
        target_fwd = state.command[0]
        actual_fwd = d.cvel[1, 3]  # x 速度
        vel_reward = w["forward_velocity"] * jp.exp(-2.0 * (actual_fwd - target_fwd)**2)
        
        # 侧向速度惩罚（保持直线）
        lat_vel = d.cvel[1, 4]
        lat_penalty = -0.5 * lat_vel**2
        
        # 直立
        quat = d.xquat[1]
        upright = quat[0]  # w
        upright_reward = w["upright"] * (upright ** 2)
        
        # 高度
        height = d.xpos[1, 2]
        target_h = state.command[3]
        height_reward = -1.0 * jp.abs(height - target_h)
        
        # 能量
        joint_vel = d.qvel[6:6+self.action_dim]
        energy = jp.sum(jp.abs(action * joint_vel))
        energy_penalty = w["energy_efficiency"] * energy
        
        # 动作平滑（需要历史，这里简化）
        smooth_penalty = -0.001 * jp.sum(action ** 2)
        
        # 生存
        survival = w["survival"]
        
        total = (vel_reward + lat_penalty + upright_reward + 
                height_reward + energy_penalty + smooth_penalty + survival)
        
        return jp.squeeze(total)
    
    def _check_termination(self, state: X2UltraState) -> jp.ndarray:
        """终止检查"""
        d = state.physics_data
        cfg = self.const.TERMINATION_CONFIG
        
        height = d.xpos[1, 2]
        too_low = height < cfg["min_body_height"]
        
        quat = d.xquat[1]
        # 简化倾斜检查
        tilt_penalty = 1.0 - quat[0]**2  # 偏离直立的程度
        too_tilted = tilt_penalty > 0.5  # 约 60 度
        
        return jp.logical_or(too_low, too_tilted)
     
     
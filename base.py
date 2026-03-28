"""
base_functional.py
函数式 X2 Ultra 环境，兼容 JAX JIT/VMAP
"""

import dataclasses
from typing import Any, Dict, Tuple, Callable
import numpy as np
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jp
from flax import struct
from functools import partial
import x2_constant as constants


@struct.dataclass
class X2UltraEnvState:
    """纯函数式环境状态（可被 JAX 处理）"""
    # 核心物理状态
    physics_data: Any  # mjx.Data
    
    # 时间信息
    time: jp.ndarray  # 标量数组，便于 JIT
    steps: jp.ndarray
    
    # 命令
    command: jp.ndarray  # [4]
    
    # 历史
    last_action: jp.ndarray  # [action_dim]
    
    # 随机数状态
    rng: jp.ndarray


@struct.dataclass
class X2UltraConfig:
    """环境配置（静态）"""
    model_path: str = "x2_ultra.xml"
    control_frequency: float = 50.0
    simulation_frequency: float = 1000.0
    
    @property
    def control_substeps(self) -> int:
        return int(self.simulation_frequency / self.control_frequency)


class X2UltraFunctionalEnv:
    """
    函数式环境：不保存状态，所有方法为纯函数
    通过 jax.jit 编译 step/reset
    """
    
    def __init__(self, config: Dict = None):
        self.config = X2UltraConfig(**(config or {}))
        self.const = constants.X2UltraConstants()
        
        # 加载模型（只读，可被 JIT 捕获）
        self._model = self._load_model()
        self._model_jx = mjx.put_model(self._model)
        
        # 预计算维度
        self.obs_dim = self._compute_obs_dim()
        self.action_dim = self._model.nu
        
        print(f"[X2Ultra] 初始化完成: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
    
    def _load_model(self) -> mujoco.MjModel:
        xml_path = self.config.model_path
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
            print(f"[X2Ultra] 加载模型: {xml_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"无法加载模型 {xml_path}: {e}")
    
    def _compute_obs_dim(self) -> int:
        """计算观测维度"""
        cfg = self.const.OBSERVATION_CONFIG
        dim = 0
        
        # 关节位置 (nq - 7 自由关节)
        if cfg["include_joint_pos"]:
            dim += self._model.nq - 7
        
        # 关节速度 (nv - 6 自由关节)
        if cfg["include_joint_vel"]:
            dim += self._model.nv - 6
        
        # IMU 数据
        if cfg["include_body_lin_vel"]:
            dim += 3
        if cfg["include_body_ang_vel"]:
            dim += 3
        if cfg["include_body_height"]:
            dim += 1
        if cfg["include_gravity_vector"]:
            dim += 3
        
        # 命令和相位
        if cfg["include_command"]:
            dim += 4
        if cfg["include_phase"]:
            dim += 1
        if cfg["include_last_action"]:
            dim += self._model.nu  # 动作维度
        
        return dim
    
    # =========================================================================
    # 纯函数：可被 jax.jit
    # =========================================================================
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jp.ndarray) -> Tuple[X2UltraEnvState, jp.ndarray]:
        """纯函数式重置"""
        # 创建初始物理状态
        data = mjx.make_data(self._model_jx)
        
        # 设置初始位置
        qpos = data.qpos.at[2].set(0.68)  # 站立高度
        
        # 设置默认关节角度
        default_pos = self.const.DEFAULT_JOINT_POSITIONS
        for i, pos in enumerate(default_pos):
            if 7 + i < self._model.nq:
                qpos = qpos.at[7 + i].set(pos)
        
        data = data.replace(qpos=qpos)
        
        # 生成随机命令
        rng, cmd_rng = jax.random.split(rng)
        command = self._sample_command(cmd_rng)
        
        # 创建状态
        state = X2UltraEnvState(
            physics_data=data,
            time=jp.array(0.0),
            steps=jp.array(0),
            command=command,
            last_action=jp.zeros(self.action_dim),
            rng=rng
        )
        
        # 获取观测
        obs = self._get_obs(state)
        
        return state, obs
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: X2UltraEnvState,
        action: jp.ndarray,
        rng: jp.ndarray
    ) -> Tuple[X2UltraEnvState, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, Dict]:
        """纯函数式 step，完全可 JIT"""
        
        # 处理动作（PD 控制）
        ctrl = self._action_to_ctrl(state, action)
        
        # 模拟多个子步
        data = state.physics_data
        
        def sim_step(carry, _):
            d = carry
            d = d.replace(ctrl=ctrl)
            d = mjx.step(self._model_jx, d)
            return d, None
        
        data, _ = jax.lax.scan(
            sim_step,
            data,
            None,
            length=self.config.control_substeps
        )
        
        # 更新时间
        dt = self.config.control_substeps * self._model.opt.timestep
        new_time = state.time + dt
        new_steps = state.steps + 1
        
        # 更新命令（定期）
        rng, cmd_rng = jax.random.split(rng)
        new_command = jax.lax.cond(
            new_steps % 10 == 0,  # 10Hz 命令更新
            lambda _: self._sample_command(cmd_rng),
            lambda _: state.command,
            None
        )
        
        # 创建新状态
        new_state = X2UltraEnvState(
            physics_data=data,
            time=new_time,
            steps=new_steps,
            command=new_command,
            last_action=action,
            rng=rng
        )
        
        # 获取观测
        obs = self._get_obs(new_state)
        
        # 计算奖励
        reward = self._compute_reward(new_state, action)
        
        # 检查终止
        done = self._check_termination(new_state)
        truncated = new_steps >= self.const.TERMINATION_CONFIG["max_episode_length"]
        
        # 信息字典（JAX 兼容）
        info = {
            "reward": reward,
            "height": data.xpos[1, 2],  # 身体高度
        }
        
        return new_state, obs, reward, done, truncated, info
    
    # =========================================================================
    # 辅助函数
    # =========================================================================
    
    def _sample_command(self, rng: jp.ndarray) -> jp.ndarray:
        """采样随机命令"""
        limits = self.const.COMMAND_LIMITS
        rngs = jax.random.split(rng, 4)
        
        return jp.array([
            jax.random.uniform(rngs[0], minval=limits["forward_speed"][0], maxval=limits["forward_speed"][1]),
            jax.random.uniform(rngs[1], minval=limits["lateral_speed"][0], maxval=limits["lateral_speed"][1]),
            jax.random.uniform(rngs[2], minval=limits["turning_rate"][0], maxval=limits["turning_rate"][1]),
            jax.random.uniform(rngs[3], minval=limits["body_height"][0], maxval=limits["body_height"][1]),
        ])
    
    def _action_to_ctrl(self, state: X2UltraEnvState, action: jp.ndarray) -> jp.ndarray:
        """将动作转换为控制信号"""
        cfg = self.const.ACTION_CONFIG
        
        if cfg["action_type"] == "torque":
            return action * cfg["action_scale"]
        elif cfg["action_type"] == "position":
            # PD 控制
            current_pos = state.physics_data.qpos[7:]
            current_vel = state.physics_data.qvel[6:]
            target_pos = action
            
            kp = cfg["kp"]
            kd = cfg["kd"]
            
            ctrl = kp * (target_pos - current_pos) - kd * current_vel
            return jp.clip(ctrl, *cfg["clip_range"])
        else:
            raise ValueError(f"未知动作类型: {cfg['action_type']}")
    
    def _get_obs(self, state: X2UltraEnvState) -> jp.ndarray:
        """获取观测（纯函数）"""
        obs_parts = []
        data = state.physics_data
        cfg = self.const.OBSERVATION_CONFIG
        
        # 关节位置
        if cfg["include_joint_pos"]:
            obs_parts.append(data.qpos[7:])
        
        # 关节速度
        if cfg["include_joint_vel"]:
            obs_parts.append(data.qvel[6:])
        
        # 身体速度（从 cvel）
        if cfg["include_body_lin_vel"]:
            # 注意：mjx 中 cvel 的索引可能需要调整
            obs_parts.append(data.cvel[1, 3:6])
        
        if cfg["include_body_ang_vel"]:
            obs_parts.append(data.cvel[1, 0:3])
        
        # 身体高度
        if cfg["include_body_height"]:
            obs_parts.append(jp.array([data.xpos[1, 2]]))
        
        # 重力向量
        if cfg["include_gravity_vector"]:
            # 从四元数计算
            quat = data.xquat[1]
            gravity = mjx._src.math.quat_rotate(quat, jp.array([0, 0, -1]))
            obs_parts.append(gravity)
        
        # 命令
        if cfg["include_command"]:
            obs_parts.append(state.command)
        
        # 相位（简化）
        if cfg["include_phase"]:
            phase = jp.sin(2 * jp.pi * state.time * 1.5)  # 1.5Hz 步态
            obs_parts.append(jp.array([phase]))
        
        # 上次动作
        if cfg["include_last_action"]:
            obs_parts.append(state.last_action)
        
        return jp.concatenate(obs_parts)
    
    def _compute_reward(self, state: X2UltraEnvState, action: jp.ndarray) -> jp.ndarray:
        """计算奖励（需要丰富的奖励函数才能学习行走）"""
        data = state.physics_data
        weights = self.const.REWARD_WEIGHTS
        
        # 目标速度跟踪
        target_vel = state.command[0]  # 前进速度
        actual_vel = data.cvel[1, 3]  # x方向速度
        vel_error = jp.abs(actual_vel - target_vel)
        vel_reward = weights["forward_velocity"] * (1.0 - vel_error)
        
        # 直立奖励
        quat = data.xquat[1]
        uprightness = quat[0]  # w 分量，接近1表示直立
        upright_reward = weights["upright"] * uprightness
        
        # 能量惩罚
        joint_vel = data.qvel[6:]
        energy = jp.sum(jp.abs(action * joint_vel))
        energy_penalty = weights["energy_efficiency"] * energy
        
        # 高度奖励
        height = data.xpos[1, 2]
        target_height = state.command[3]
        height_reward = -0.5 * jp.abs(height - target_height)
        
        # 生存奖励
        survival = weights["survival"]
        
        total = vel_reward + upright_reward + energy_penalty + height_reward + survival
        
        # 确保是标量
        return jp.squeeze(total)
    
    def _check_termination(self, state: X2UltraEnvState) -> jp.ndarray:
        """检查是否终止"""
        data = state.physics_data
        cfg = self.const.TERMINATION_CONFIG
        
        # 高度检查
        height = data.xpos[1, 2]
        too_low = height < cfg["min_body_height"]
        
        # 倾斜检查
        quat = data.xquat[1]
        tilt = jp.arccos(jp.clip(jp.abs(quat[0]), -1.0, 1.0)) * 2
        too_tilted = tilt > cfg["max_body_tilt"]
        
        return jp.logical_or(too_low, too_tilted)
     
     
import dataclasses
from typing import Any, Dict, Optional, Tuple
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
import jax
import jax.numpy as jp
from flax import struct
import x2_constant as constants

@struct.dataclass
class X2UltraState:    
    """x2_ultra机器人状态"""        
# 物理状态
    physics_state: Any  # mjx.Data
    time: float = 0.0
    steps: int = 0        
# 命令状态
    command: jp.ndarray = jp.zeros(4)  # [forward, lateral, turning, height]
    last_command: jp.ndarray = jp.zeros(4)
    command_time: float = 0.0        
# 步态状态
    phase: float = 0.0
    gait_phase: jp.ndarray = jp.zeros(2)  # 两条腿的相位
    foot_contact_state: jp.ndarray = jp.zeros(2)  # 脚部接触状态        
# 观测历史
    observation_history: jp.ndarray = jp.zeros((3, 100))  # 简化        
# 动作历史
    action_history: jp.ndarray = jp.zeros((2, 12))  # 12个关节        
# 奖励累积
    total_reward: float = 0.0
    episode_reward: float = 0.0
    reward_components: Dict[str, float] = struct.field(default_factory=dict)        
# 终止标志
    terminated: bool = False
    truncated: bool = False
    termination_info: Dict[str, Any] = struct.field(default_factory=dict)

class X2UltraBaseEnv:
     def __init__(self, config: Optional[Dict[str, Any]] = None):        
        """初始化环境"""        
        self.config = config or {}        
        self.constants = constants.X2UltraConstants()                
        # 加载模型        
        self.model = self._load_model()        
        self.model_jx = mjx.put_model(self.model)                
        # 初始化状态        
        self.state = self._initialize_state()                
        # 计算维度        
        self.observation_dim = self._compute_observation_dim()        
        self.action_dim = self.model.nu                
        # 初始化随机数生成器        
        self._rng = jax.random.PRNGKey(42)                
        # 提取机器人信息        
        self._extract_robot_info()
        # 提取动作缩放因子 
        self.action_scale = self.constants.ACTION_CONFIG.get("action_scale", 1.0)                
        
        print(f"{self.constants.ROBOT_NAME}环境初始化完成:")        
        print(f"  观测维度: {self.observation_dim}")        
        print(f"  动作维度: {self.action_dim}")        
        print(f"  关节数量: {self.model.nq}")        
        print(f"  执行器数量: {self.model.nu}")

     def _load_model(self) -> mujoco.MjModel:
        """加载MuJoCo模型"""
        xml_path = self.config.get("model_path", "x2_ultra.xml")
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)            
            print(f"成功加载模型: {xml_path}")        
        except Exception as e:            
            print(f"无法从文件加载模型: {e}")
                              
        return model
        
     def _extract_robot_info(self):        
        """提取机器人信息"""        
        self.joint_info = {}        
        self.body_info = {}        
        self.actuator_info = {}
    
        # 关节信息
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_info[name] = {
                    "type": self.model.jnt_type[i],
                    "range": self.model.jnt_range[i] if self.model.jnt_range[i, 0] != self.model.jnt_range[i, 1] else None,
                    "damping": self.model.dof_damping[i] if i < self.model.nv else 0.0,}
                
            # 身体信息
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self.body_info[name] = {
                    "mass": self.model.body_mass[i],
                    "inertia": self.model.body_inertia[i],}
                                    
            # 执行器信息
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuator_info[name] = {
                    "gain": self.model.actuator_gainprm[i, 0],
                    "bias": self.model.actuator_biasprm[i, 0],
                    "ctrlrange": self.model.actuator_ctrlrange[i],}
    
     def _initialize_state(self) -> X2UltraState:
        """初始化状态"""
        data = mjx.make_data(self.model_jx)                
        
        # 设置初始位置
        qpos = data.qpos.copy()
        qpos = qpos.at[2].set(1.0)  # z位置

       # 设置关节初始角度
        if self.model.nq > 7:
            # 使用默认关节位置
            default_pos = self.constants.DEFAULT_JOINT_POSITIONS
            for i, pos in enumerate(default_pos):
                if 7 + i < self.model.nq:
                    qpos = qpos.at[7 + i].set(pos)
        
        data = data.replace(qpos=qpos)

        # 初始命令
        initial_command = jp.array([0.5, 0.0, 0.0, 0.8])

        return X2UltraState(
            physics_state=data,
            command=initial_command,
            observation_history=jp.zeros((self.constants.OBSERVATION_CONFIG["history_length"], 
                                         self.observation_dim)),
            action_history=jp.zeros((2, self.action_dim)),)
     
     def _compute_observation_dim(self) -> int:
        """计算观测维度"""
        obs_config = self.constants.OBSERVATION_CONFIG
        dim = 0
        
        # 关节信息
        if obs_config["include_joint_pos"]:
            dim += self.model.nq - 7
        
        if obs_config["include_joint_vel"]:
            dim += self.model.nv - 6
        
        # 身体信息
        if obs_config["include_body_lin_vel"]:
            dim += 3
        
        if obs_config["include_body_ang_vel"]:
            dim += 3
        
        if obs_config["include_body_height"]:
            dim += 1
        
        if obs_config["include_gravity_vector"]:
            dim += 3
        
        # 命令和相位
        if obs_config["include_command"]:
            dim += 4
        
        if obs_config["include_phase"]:
            dim += 1
        
        if obs_config["include_last_action"]:
            dim += self.action_dim
        
        return dim
     
     def _get_observation(self, state: X2UltraState) -> jp.ndarray:
        """获取观测"""
        obs_parts = []
        physics = state.physics_state
        obs_config = self.constants.OBSERVATION_CONFIG
        
        # 关节位置和速度
        if obs_config["include_joint_pos"]:
            joint_pos = physics.qpos[7:]
            obs_parts.append(joint_pos)
        
        if obs_config["include_joint_vel"]:
            joint_vel = physics.qvel[6:]
            obs_parts.append(joint_vel)
        
        # 身体线速度和角速度
        if obs_config["include_body_lin_vel"]:
            # 线速度在索引 3:6
            body_lin_vel = physics.cvel[1, 3:] 
            obs_parts.append(body_lin_vel)
        
        if obs_config["include_body_ang_vel"]:
            # 角速度在索引 0:3
            body_ang_vel = physics.cvel[1, :3]
            obs_parts.append(body_ang_vel)
        
        # 身体高度
        if obs_config["include_body_height"]:
            body_height = physics.xpos[1, 2]
            obs_parts.append(jp.array([body_height]))
        
        # 重力向量
        if obs_config["include_gravity_vector"]:
            body_quat = physics.xquat[1]
            gravity_world = jp.array([0, 0, -1])
            gravity_body = mjx_math.quat_rotate(body_quat, gravity_world)
            obs_parts.append(gravity_body)
        
        # 命令和相位
        if obs_config["include_command"]:
            obs_parts.append(state.command)
        
        if obs_config["include_phase"]:
            obs_parts.append(jp.array([state.phase]))
        
        # 上次动作
        if obs_config["include_last_action"]:
            last_action = state.action_history[0] if state.action_history.shape[0] > 0 else jp.zeros(self.action_dim)
            obs_parts.append(last_action)
        
        # 组合所有观测
        observation = jp.concatenate(obs_parts)
        
        return observation
     
     def reset(self, rng: Optional[jp.ndarray] = None) -> Tuple[X2UltraState, jp.ndarray]:
        """重置环境"""
        if rng is None:
            rng = self._rng
        
        # 重置物理状态
        self.state = self._initialize_state()
        
        # 应用域随机化
        self.state = self._apply_domain_randomization(self.state, rng)
        
        # 生成初始命令
        command_rng, rng = jax.random.split(rng)
        initial_command = self._generate_random_command(command_rng)
        
        # 更新状态
        self.state = self.state.replace(
            command=initial_command,
            last_command=jp.zeros(4),
            command_time=0.0,
            steps=0,
            time=0.0,
            phase=0.0,
            gait_phase=jp.zeros(2),
            foot_contact_state=jp.zeros(2),
            total_reward=0.0,
            episode_reward=0.0,
            reward_components={},
            terminated=False,
            truncated=False,
            termination_info={},
        )
        
        # 获取初始观测
        observation = self._get_observation(self.state)
        
        # 更新观测历史
        obs_history = jp.zeros((self.constants.OBSERVATION_CONFIG["history_length"], self.observation_dim))
        obs_history = obs_history.at[0].set(observation)
        self.state = self.state.replace(observation_history=obs_history)
        
        self._rng = rng
        
        return self.state, observation
     
     def _generate_random_command(self, rng: jp.ndarray) -> jp.ndarray:
        """生成随机命令"""
        rngs = jax.random.split(rng, 4)
        
        limits = self.constants.COMMAND_LIMITS
        
        raw_command = jp.array([
            jax.random.uniform(rngs[0], minval=limits["forward_speed"][0], maxval=limits["forward_speed"][1]),
            jax.random.uniform(rngs[1], minval=limits["lateral_speed"][0], maxval=limits["lateral_speed"][1]),
            jax.random.uniform(rngs[2], minval=limits["turning_rate"][0], maxval=limits["turning_rate"][1]),
            jax.random.uniform(rngs[3], minval=limits["body_height"][0], maxval=limits["body_height"][1]),
        ])
        
        return raw_command
     
     def _apply_domain_randomization(self, state: X2UltraState, rng: jp.ndarray) -> X2UltraState:
        """应用域随机化"""
        if not self.constants.RANDOMIZATION_CONFIG["enabled"]:
            return state
        
        rngs = jax.random.split(rng, 5)
        physics = state.physics_state
        model_jx = physics.model
        
        # 随机化摩擦系数
        friction = jax.random.uniform(
            rngs[0],
            minval=self.constants.RANDOMIZATION_CONFIG["friction_range"][0],
            maxval=self.constants.RANDOMIZATION_CONFIG["friction_range"][1]
        )
        model_jx = model_jx.replace(
            opt=model_jx.opt.replace(
                friction=model_jx.opt.friction.at[0].set(friction)
            )
        )
        
        # 随机化执行器强度
        motor_strength = jax.random.uniform(
            rngs[1],
            minval=self.constants.RANDOMIZATION_CONFIG["motor_strength_range"][0],
            maxval=self.constants.RANDOMIZATION_CONFIG["motor_strength_range"][1],
            shape=(self.model.nu,)
        )
        actuator_gain = model_jx.actuator_gainprm[:, 0] * motor_strength
        model_jx = model_jx.replace(
            actuator_gainprm=model_jx.actuator_gainprm.at[:, 0].set(actuator_gain)
        )
        
        # 随机化地面高度
        ground_height = jax.random.uniform(
            rngs[2],
            minval=self.constants.RANDOMIZATION_CONFIG["ground_height_range"][0],
            maxval=self.constants.RANDOMIZATION_CONFIG["ground_height_range"][1]
        )
        geom_pos = model_jx.geom_pos.copy()
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            geom_pos = geom_pos.at[floor_id, 2].set(ground_height)
        model_jx = model_jx.replace(geom_pos=geom_pos)
        
        # 更新物理状态
        physics = physics.replace(model=model_jx)
        
        return state.replace(physics_state=physics)
     
     def step(self, action: jp.ndarray, rng: Optional[jp.ndarray] = None) -> Tuple[X2UltraState, jp.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        if rng is None:
            rng = self._rng
        
        # 更新命令
        current_time = self.state.time
        command_interval = 1.0 / 10.0  # 10Hz
        
        if current_time >= self.state.command_time + command_interval:
            command_rng, rng = jax.random.split(rng)
            new_command = self._generate_random_command(command_rng)
            
            self.state = self.state.replace(
                last_command=self.state.command,
                command=new_command,
                command_time=current_time
            )
        
        # 处理动作
        ctrl = self._process_action(action, self.state)
        
        # 执行模拟步
        physics_state = self.state.physics_state
        for _ in range(self.constants.CONTROL_SUBSTEPS):
            physics_state = physics_state.replace(ctrl=ctrl)
            physics_state = mjx.step(self.model_jx, physics_state)
        
        # 更新时间
        dt = self.constants.CONTROL_SUBSTEPS * self.model.opt.timestep
        new_time = current_time + dt
        new_steps = self.state.steps + 1
        
        # 更新物理状态
        self.state = self.state.replace(
            physics_state=physics_state,
            time=new_time,
            steps=new_steps
        )
        
        # 更新动作历史
        action_history = self.state.action_history
        action_history = jp.roll(action_history, shift=1, axis=0)
        action_history = action_history.at[0].set(action)
        self.state = self.state.replace(action_history=action_history)
        
        # 获取观测
        observation = self._get_observation(self.state)
        
        # 更新观测历史
        obs_history = self.state.observation_history
        obs_history = jp.roll(obs_history, shift=1, axis=0)
        obs_history = obs_history.at[0].set(observation)
        self.state = self.state.replace(observation_history=obs_history)
        
        # 计算奖励
        reward, reward_components = self._compute_reward(self.state, action)
        
        # 更新奖励累积
        new_total_reward = self.state.total_reward + reward
        new_episode_reward = self.state.episode_reward + reward
        
        self.state = self.state.replace(
            total_reward=new_total_reward,
            episode_reward=new_episode_reward,
            reward_components=reward_components
        )
        
        # 检查终止条件
        terminated, termination_info = self._check_termination(self.state)
        truncated = self.state.steps >= self.constants.TERMINATION_CONFIG["max_episode_length"]
        
        self.state = self.state.replace(
            terminated=terminated,
            truncated=truncated,
            termination_info=termination_info
        )
        
        # 构建信息字典
        info = {
            "reward_components": reward_components,
            "termination_info": termination_info,
            "steps": self.state.steps,
            "time": self.state.time,
            "command": self.state.command,
        }
        
        self._rng = rng
        
        return self.state, observation, reward, terminated, truncated, info
     
     def _process_action(self, action: jp.ndarray, state: X2UltraState) -> jp.ndarray:
        """处理动作"""
        action_config = self.constants.ACTION_CONFIG
        action_type = action_config["action_type"]
        
        if action_type == "torque":
            ctrl = action * self.action_scale
        elif action_type == "position":
            target_pos = action
            current_pos = state.physics_state.qpos[7:]
            current_vel = state.physics_state.qvel[6:]
            
            kp = action_config["kp"]
            kd = action_config["kd"]
            
            ctrl = kp * (target_pos - current_pos) + kd * (-current_vel)
        else:
            raise ValueError(f"未知的动作类型: {action_type}")
        
        # 限幅
        clip_min, clip_max = action_config["clip_range"]
        ctrl = jp.clip(ctrl, clip_min, clip_max)
        
        return ctrl
     
     def _compute_reward(self, state: X2UltraState, action: jp.ndarray) -> Tuple[float, Dict[str, float]]:
        """计算奖励"""
        # 基础实现，实际应该在joystick.py中实现
        reward_components = {
            "survival": 0.1,
        }
        
        total_reward = sum(reward_components.values())
        
        return float(total_reward), reward_components
     
     def _check_termination(self, state: X2UltraState) -> Tuple[bool, Dict[str, Any]]:
        """检查终止条件"""
        termination_config = self.constants.TERMINATION_CONFIG
        termination_info = {}
        terminated = False
        
        physics = state.physics_state
        
        # 身体高度检查
        body_height = physics.xpos[1, 2]
        if body_height < termination_config["min_body_height"]:
            termination_info["body_height"] = float(body_height)
            terminated = True
        
        # 身体倾斜检查
        body_quat = physics.xquat[1]
        tilt_angle = jp.arccos(jp.abs(body_quat[0])) * 2.0
        if tilt_angle > termination_config["max_body_tilt"]:
            termination_info["body_tilt"] = float(tilt_angle)
            terminated = True
        
        return terminated, termination_info
     
     
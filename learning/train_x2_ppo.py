"""
train_x2_ultra_ppo.py
x2_ultra双足机器人PPO训练脚本（最终优化版）
基于MuJoCo Playground最佳实践
"""

import os
import sys
import time
import functools
from typing import Dict, Tuple, Any
from dataclasses import dataclass
import json

import numpy as np
import jax
import jax.numpy as jp
from jax import random, jit, lax
import flax
from flax import linen as nn
from flax.training import checkpoints
import optax
from functools import partial
import tqdm

# 确保能找到环境模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base import X2UltraBaseEnv, X2UltraState
from joystick import JoystickController
from gait import GaitGenerator


# ============================================================================
# 配置
# ============================================================================

@dataclass
class PPOConfig:
    """PPO训练配置"""
    # 环境参数
    env_name: str = "x2_ultra"
    num_envs: int = 32  # 并行环境数（根据GPU内存调整）
    episode_length: int = 1000
    
    # PPO超参数
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    
    # 训练参数
    num_iterations: int = 2000  # 总迭代次数
    num_epochs: int = 4  # 每次迭代的训练轮数
    minibatch_size: int = 256
    num_minibatches: int = 8
    
    # 网络架构
    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    
    # 优化
    normalize_obs: bool = True
    normalize_adv: bool = True
    learning_rate_decay: float = 0.99  # 学习率衰减系数
    
    # 保存与日志
    log_interval: int = 10
    save_interval: int = 100
    checkpoint_dir: str = "./checkpoints/x2_ultra_ppo"
    seed: int = 42


# ============================================================================
# 网络定义（正交初始化）
# ============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    action_dim: int
    hidden_sizes: Tuple[int, ...]
    
    @nn.compact
    def __call__(self, x):
        # 共享特征提取
        for size in self.hidden_sizes:
            x = nn.Dense(
                size,
                kernel_init=nn.initializers.orthogonal(scale=np.sqrt(2)),
                bias_init=nn.initializers.constant(0.0)
            )(x)
            x = nn.tanh(x)
        
        # Actor头（正交初始化，scale=0.01保证初始策略接近均匀分布）
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.constant(0.0),
            name="actor_mean"
        )(x)
        
        # 独立的学习参数：动作标准差
        actor_logstd = self.param(
            "actor_logstd",
            nn.initializers.constant(-0.5),  # 初始标准差 ≈ 0.6
            (self.action_dim,)
        )
        
        # Critic头（正交初始化，scale=1.0）
        critic = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
            bias_init=nn.initializers.constant(0.0),
            name="critic"
        )(x)
        
        return actor_mean, actor_logstd, critic.squeeze(-1)


# ============================================================================
# 核心训练函数（JIT优化）
# ============================================================================

@partial(jit, static_argnums=(0, 5, 6))
def collect_rollouts(
    env_step_fn: callable,
    carry_state: Tuple,
    network_params: Dict,
    apply_fn: callable,
    rng: jax.random.PRNGKey,
    episode_length: int,
    num_envs: int
) -> Tuple[Dict, Tuple]:
    """使用lax.scan高效收集rollouts"""
    
    def rollout_step(carry, rng):
        env_state, obs = carry
        
        # 1. 获取动作和价值
        actor_mean, actor_logstd, value = apply_fn(network_params, obs)
        action_rng, env_rng = jax.random.split(rng)
        
        # 2. 重参数化采样
        std = jp.exp(actor_logstd)
        noise = jax.random.normal(action_rng, actor_mean.shape)
        action = actor_mean + noise * std
        
        # 3. 计算log概率
        log_prob = -0.5 * ((action - actor_mean) / std) ** 2 \
                   - 0.5 * jp.log(2 * jp.pi) - actor_logstd
        log_prob = log_prob.sum(axis=-1)
        
        # 4. 环境交互
        next_state, next_obs, reward, done, trunc, info = env_step_fn(env_state, action, env_rng)
        
        #统一处理截断/终止，以保证优势计算正确
        terminated = jp.logical_or(done, trunc)
        
        transition = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': terminated,
            'log_prob': log_prob,
            'value': value
        }
        
        return (next_state, next_obs), transition
    
    # 使用scan加速
    rngs = jax.random.split(rng, episode_length)
    (final_state, final_obs), transitions = lax.scan(
        rollout_step,
        carry_state,
        rngs,
        length=episode_length
    )
    
    return transitions, (final_state, final_obs)


@partial(jit, static_argnums=(0,))
def compute_gae(
    rewards: jp.ndarray,
    values: jp.ndarray,
    dones: jp.ndarray,
    last_values: jp.ndarray,  # 增加末状态价值，用于正确的bootstrap
    gamma: float,
    gae_lambda: float
) -> Tuple[jp.ndarray, jp.ndarray]:
    """计算GAE和回报（使用scan反向计算"""
    
    # 拼接末状态价值，保证优势/回报引导完整
    next_values = jp.concatenate(
        [values[1:], jp.expand_dims(last_values, axis=0)],
        axis=0
    )
    
    def gae_step(gae, transition):
        reward, value, next_value, done = transition
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return gae, (gae, gae + value)
    
    _, (advantages, returns) = lax.scan(
        gae_step,
        jp.zeros_like(last_values),
        (rewards, values, next_values, dones),
        reverse=True
    )
    
    return advantages, returns


@jit
def update_step(
    params: Dict,
    apply_fn: callable,
    batch: Tuple,
    optimizer_state: Any,
    optimizer: optax.GradientTransformation,
    config: PPOConfig
) -> Tuple[Dict, Any, Dict]:
    """PPO参数更新步骤"""
    
    obs, actions, old_log_probs, advantages, returns = batch
    
    def loss_fn(params):
        # 前向传播
        actor_mean, actor_logstd, values = apply_fn(params, obs)
        
        # 计算新的log概率
        std = jp.exp(actor_logstd)
        new_log_probs = -0.5 * ((actions - actor_mean) / std) ** 2 \
                        - 0.5 * jp.log(2 * jp.pi) - actor_logstd
        new_log_probs = new_log_probs.sum(axis=-1)
        
        # 重要性采样比率
        ratio = jp.exp(new_log_probs - old_log_probs)
        
        # PPO裁剪损失
        policy_loss1 = ratio * advantages
        policy_loss2 = jp.clip(ratio, 1 - config.clip_epsilon, 
                               1 + config.clip_epsilon) * advantages
        policy_loss = -jp.minimum(policy_loss1, policy_loss2).mean()
        
        # 价值损失
        value_loss = ((returns - values) ** 2).mean()
        
        # 熵奖励
        entropy = -new_log_probs.mean()
        entropy_loss = -config.entropy_coef * entropy
        
        # 总损失
        total_loss = policy_loss + config.value_coef * value_loss + entropy_loss
        
        return total_loss, {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }
    
    # 计算梯度
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # 梯度裁剪由优化器链处理，这里直接调用优化器更新
    updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_optimizer_state, metrics


# ============================================================================
# 主训练循环
# ============================================================================

def train(config: PPOConfig):
    """主训练函数"""
    
    print("=" * 60)
    print(f"开始训练 {config.env_name}")
    print("=" * 60)
    print(f"环境数量: {config.num_envs}")
    print(f"总迭代次数: {config.num_iterations}")
    print(f"学习率: {config.learning_rate}")
    
    # 创建环境
    env = X2UltraBaseEnv({'model_path': 'x2_ultra.xml'})
    
    obs_dim = env.observation_dim
    action_dim = env.action_dim
    
    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    
    # 创建网络和优化器
    network = ActorCritic(action_dim=action_dim, hidden_sizes=config.hidden_sizes)
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    dummy_obs = jp.zeros((obs_dim,))
    params = network.init(init_rng, dummy_obs)
    
    # 学习率调度
    schedule = optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=1000,
        decay_rate=config.learning_rate_decay
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=schedule)
    )
    
    optimizer_state = optimizer.init(params)
    
    # 创建检查点目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(config.checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    # 环境重置函数
    def reset_env(rng):
        state, obs = env.reset(rng)
        return state, obs
    
    # 环境步进函数
    def step_env(state, action, rng):
        return env.step(action, rng)
    
    # 训练循环
    best_reward = -float('inf')
    
    print("\n开始训练...")
    
    for iteration in tqdm.tqdm(range(config.num_iterations)):
        iter_start = time.time()
        
        # 1. 重置环境
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config.num_envs)
        
        # 向量化重置
        reset_fn = jax.vmap(reset_env)
        env_states, obs = reset_fn(reset_rngs)
        
        # 2. 收集rollouts
        rng, rollout_rng = jax.random.split(rng)
        
        # 向量化rollout
        vmap_collect = jax.vmap(
            functools.partial(
                collect_rollouts,
                step_env,
                network_params=params,
                apply_fn=network.apply,
                episode_length=config.episode_length,
                num_envs=config.num_envs
            )
        )
        
        # 为每个环境生成随机数
        rollout_rngs = jax.random.split(rollout_rng, config.num_envs)
        transitions, (final_states, final_obs) = vmap_collect(
            (env_states, obs), rollout_rngs
        )
        # 调整轴顺序，确保时间轴在前、环境轴在后
        transitions = jax.tree_map(lambda x: jp.swapaxes(x, 0, 1), transitions)
        
        # 3. 计算GAE和回报
        # 计算末状态价值作为bootstrap
        _, _, last_values = network.apply(params, final_obs)
        advantages, returns = compute_gae(
            transitions['reward'],
            transitions['value'],
            transitions['done'],
            last_values,
            config.gamma,
            config.gae_lambda
        )
        
        # 4. 标准化优势
        if config.normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 5. 准备训练数据
        obs = transitions['obs'].reshape(-1, obs_dim)
        actions = transitions['action'].reshape(-1, action_dim)
        log_probs = transitions['log_prob'].reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        
        batch_size = obs.shape[0]
        
        # 6. 训练多个epoch
        epoch_metrics = []
        
        for epoch in range(config.num_epochs):
            # 打乱数据
            rng, shuffle_rng = jax.random.split(rng)
            perm = jax.random.permutation(shuffle_rng, batch_size)
            
            obs_shuffled = obs[perm]
            actions_shuffled = actions[perm]
            log_probs_shuffled = log_probs[perm]
            advantages_shuffled = advantages[perm]
            returns_shuffled = returns[perm]
            
            # 分minibatch训练
            num_batches = batch_size // config.minibatch_size
            
            for i in range(num_batches):
                start_idx = i * config.minibatch_size
                end_idx = start_idx + config.minibatch_size
                
                batch = (
                    obs_shuffled[start_idx:end_idx],
                    actions_shuffled[start_idx:end_idx],
                    log_probs_shuffled[start_idx:end_idx],
                    advantages_shuffled[start_idx:end_idx],
                    returns_shuffled[start_idx:end_idx]
                )
                
                # 更新参数
                params, optimizer_state, metrics = update_step(
                    params, network.apply, batch, 
                    optimizer_state, optimizer, config
                )
                
                epoch_metrics.append(metrics)
        
        # 7. 计算统计信息
        mean_reward = float(transitions['reward'].mean())
        elapsed = time.time() - iter_start
        
        # 8. 日志记录
        if iteration % config.log_interval == 0:
            avg_metrics = {k: np.mean([m[k] for m in epoch_metrics]) 
                          for k in epoch_metrics[0].keys()}
            
            print(f"\nIter {iteration}/{config.num_iterations}")
            print(f"  Reward: {mean_reward:.4f}")
            print(f"  Policy Loss: {avg_metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {avg_metrics['value_loss']:.4f}")
            print(f"  Entropy: {avg_metrics['entropy']:.4f}")
            print(f"  Time: {elapsed:.2f}s")
        
        # 9. 保存检查点
        if iteration % config.save_interval == 0:
            checkpoint_data = {
                'params': params,
                'optimizer_state': optimizer_state,
                'iteration': iteration,
                'config': vars(config)
            }
            
            checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f'checkpoint_{iteration}'
            )
            checkpoints.save_checkpoint(
                config.checkpoint_dir,
                checkpoint_data,
                step=iteration,
                keep=3
            )
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_path = os.path.join(config.checkpoint_dir, 'best')
                checkpoints.save_checkpoint(
                    best_path,
                    checkpoint_data,
                    step=iteration
                )
    
    # 保存最终模型
    final_path = os.path.join(config.checkpoint_dir, 'final')
    checkpoints.save_checkpoint(
        final_path,
        {'params': params, 'config': vars(config)},
        step=config.num_iterations
    )
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳奖励: {best_reward:.4f}")
    print(f"模型保存到: {config.checkpoint_dir}")
    print("=" * 60)
    
    return params


# ============================================================================
# 评估函数
# ============================================================================

def evaluate(params: Dict, config: PPOConfig, num_episodes: int = 10):
    """评估训练好的策略"""
    
    print("\n" + "=" * 60)
    print("评估策略")
    print("=" * 60)
    
    env = X2UltraBaseEnv({'model_path': 'x2_ultra.xml'})
    network = ActorCritic(action_dim=env.action_dim, hidden_sizes=config.hidden_sizes)
    
    rng = jax.random.PRNGKey(0)
    rewards = []
    
    for episode in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state, obs = env.reset(reset_rng)
        
        episode_reward = 0.0
        done = False
        
        while not done:
            # 确定性策略（使用均值）
            actor_mean, _, _ = network.apply(params, obs)
            action = actor_mean  # 不添加噪声
            
            rng, step_rng = jax.random.split(rng)
            state, obs, reward, done, trunc, info = env.step(action, step_rng)
            episode_reward += reward
            
            if done or trunc:
                break
        
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"\n平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return rewards


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    # 配置（可根据GPU内存调整num_envs）
    config = PPOConfig(
        num_envs=512,  # GPU内存不足时降低此值
        num_iterations=2000,
        learning_rate=3e-4,
        episode_length=1000,
        hidden_sizes=(256, 256, 128),
    )
    
    # 训练
    params = train(config)
    
    # 评估
    evaluate(params, config, num_episodes=10)


if __name__ == "__main__":
    # 检查GPU
    print(f"JAX设备: {jax.devices()}")
    print(f"JAX版本: {jax.__version__}")
    print()
    
    main()
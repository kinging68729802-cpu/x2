"""
train_x2_ultra_ppo.py
x2_ultra双足机器人PPO训练脚本（最终优化版）
基于MuJoCo Playground最佳实践
"""

import os
import sys
import time
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from functools import partial
import json
import numpy as np
import jax
import jax.numpy as jp
from jax import jit, lax, vmap, random
import flax.linen as nn
from flax.training import checkpoints
import optax
import tqdm
# 确保导入修复后的环境
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import X2UltraBaseEnv, X2UltraState
# ============================================================================
# 配置
# ============================================================================
@dataclass
class PPOConfig:
    """PPO训练配置"""
    # 环境
    num_envs: int = 32              # [修复] 先从小批量开始测试
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
    num_iterations: int = 2000
    num_epochs: int = 4
    minibatch_size: int = 256       # 32*1000/128 = 250，取256
    
    # 网络
    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    
    # 其他
    normalize_adv: bool = True
    checkpoint_dir: str = "./checkpoints/x2_ultra_ppo"
    seed: int = 42
# ============================================================================
# 网络定义（添加动作约束）
# ============================================================================
class ActorCritic(nn.Module):
    """Actor-Critic网络 - 输出约束在[-1,1]的动作"""
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    
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
        
        # Actor头 - [关键修复] 使用tanh约束动作范围
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
        )(x)
        actor_mean = nn.tanh(actor_mean)
        
        actor_logstd = self.param(
            "actor_logstd",
            nn.initializers.constant(-0.5),
            (self.action_dim,)
        )
        
        critic = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
        )(x)
        
        return actor_mean, actor_logstd, critic.squeeze(-1)
# ============================================================================
# 核心训练函数 - 纯JAX实现，完全可JIT
# ============================================================================
@jit
def compute_gae(
    rewards: jp.ndarray,      # [T]
    values: jp.ndarray,       # [T]
    dones: jp.ndarray,        # [T]
    last_value: jp.ndarray,   # 标量
    gamma: float,
    gae_lambda: float
) -> Tuple[jp.ndarray, jp.ndarray]:
    """计算GAE和returns - 纯函数，完全可JIT"""
    
    # 扩展values以包含bootstrap
    next_values = jp.concatenate([values[1:], jp.array([last_value])])
    
    def gae_step(carry, transition):
        gae, next_val = carry
        reward, value, done = transition
        
        # TD误差
        delta = reward + gamma * next_val * (1 - done) - value
        
        # GAE
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        
        # 下一个value用于下次迭代
        return (gae, value), (gae, gae + value)  # advantage, return
    
    # 反向扫描
    _, (advantages, returns) = lax.scan(
        gae_step,
        (jp.array(0.0), jp.array(0.0)),  # 初始gae和next_val
        (rewards, values, dones),
        reverse=True
    )
    
    return advantages, returns
@jit
def ppo_loss(
    params: Dict,
    apply_fn: callable,
    batch: Tuple,
    config: PPOConfig
) -> Tuple[jp.ndarray, Dict]:
    """计算PPO损失 - 纯函数，完全可JIT"""
    
    obs, actions, old_log_probs, advantages, returns = batch
    
    # 前向传播
    mean, logstd, values = apply_fn(params, obs)
    
    # 计算新的log概率
    std = jp.exp(logstd)
    
    # 对数概率（高斯分布）
    log_prob = -0.5 * ((actions - mean) / std) ** 2
    log_prob = log_prob - 0.5 * jp.log(2 * jp.pi) - logstd
    log_prob = log_prob.sum(axis=-1)
    
    # 重要性采样比率
    ratio = jp.exp(log_prob - old_log_probs)
    
    # PPO裁剪目标
    surr1 = ratio * advantages
    surr2 = jp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
    policy_loss = -jp.minimum(surr1, surr2).mean()
    
    # 价值损失
    value_loss = ((returns - values) ** 2).mean()
    
    # 熵奖励（鼓励探索）
    entropy = 0.5 * (jp.log(2 * jp.pi * std ** 2) + 1).sum()
    entropy_loss = -config.entropy_coef * entropy
    
    # 总损失
    total_loss = policy_loss + config.value_coef * value_loss + entropy_loss
    
    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'approx_kl': ((ratio - 1) - jp.log(ratio)).mean(),  # 近似KL散度
    }
    
    return total_loss, metrics
# ============================================================================
# 环境交互 - 关键修复：使用vmap正确并行化
# ============================================================================
def make_env_step(env: X2UltraBaseEnv):
    """
    创建环境步进函数，兼容vmap
    关键：env已经是JIT编译的，这里只是包装
    """
    def step_fn(state, action, rng):
        # 直接调用env的JIT编译方法
        return env.step(state, action, rng)
    return step_fn
def collect_rollouts_vectorized(
    env: X2UltraBaseEnv,
    policy_params: Dict,
    policy_apply: callable,
    rng: jax.random.PRNGKey,
    config: PPOConfig
) -> Dict:
    """
    向量化收集rollouts - 关键修复版本
    
    策略：使用lax.scan进行时间展开，使用vmap进行环境并行
    """
    
    # 创建步进函数
    step_fn = make_env_step(env)
    
    # 为每个环境创建初始状态
    rng, *reset_rngs = random.split(rng, config.num_envs + 1)
    reset_rngs = jp.array(reset_rngs)
    
    # vmap重置所有环境
    def reset_single(rng):
        return env.reset(rng)
    
    # [关键] 使用vmap并行重置
    states, obs = vmap(reset_single)(reset_rngs)
    
    # 定义单步扫描函数
    def scan_step(carry, step_rng):
        states, obs = carry
        
        # 策略推理（批量）
        mean, logstd, values = policy_apply(policy_params, obs)
        
        # 采样动作
        step_rng, action_rng = random.split(step_rng)
        std = jp.exp(logstd)
        noise = random.normal(action_rng, mean.shape)
        actions = mean + noise * std
        # [关键] 再次裁剪确保安全
        actions = jp.clip(actions, -1.0, 1.0)
        
        # 计算log概率
        log_probs = -0.5 * ((actions - mean) / std) ** 2
        log_probs = log_probs - 0.5 * jp.log(2 * jp.pi) - logstd
        log_probs = log_probs.sum(axis=-1)
        
        # 环境步进 - vmap并行
        step_rng, *step_rngs = random.split(step_rng, config.num_envs + 1)
        step_rngs = jp.array(step_rngs)
        
        # [关键] vmap环境步进
        def step_single(state, action, rng):
            return step_fn(state, action, rng)
        
        new_states, new_obs, rewards, dones, truncs, infos = vmap(step_single)(
            states, actions, step_rngs
        )
        
        # 统一终止标志
        terminated = jp.logical_or(dones, truncs)
        
        # 存储transition
        transition = {
            'obs': obs,
            'action': actions,
            'reward': rewards,
            'done': terminated,
            'log_prob': log_probs,
            'value': values,
        }
        
        return (new_states, new_obs), (transition, new_obs)
    
    # 时间展开 - 使用lax.scan
    scan_rngs = random.split(rng, config.episode_length)
    
    (final_states, final_obs), (transitions, all_obs) = lax.scan(
        scan_step,
        (states, obs),
        scan_rngs,
        length=config.episode_length
    )
    
    # transitions形状: [T, N_env, ...] -> 需要调整为 [N_env, T, ...]
    transitions = jax.tree_map(lambda x: jp.swapaxes(x, 0, 1), transitions)
    
    # 计算最终价值用于bootstrap
    _, _, last_values = policy_apply(policy_params, final_obs)
    
    return transitions, final_states, last_values
# ============================================================================
# 训练循环
# ============================================================================
def train(config: PPOConfig):
    """主训练函数"""
    
    print("=" * 70)
    print("X2 Ultra 人形机器人 PPO 训练")
    print("=" * 70)
    print(f"环境数量: {config.num_envs}")
    print(f"回合长度: {config.episode_length}")
    print(f"总迭代: {config.num_iterations}")
    print(f"学习率: {config.learning_rate}")
    print(f"JAX设备: {jax.devices()}")
    print("=" * 70)
    
    # 创建环境（单例，内部使用JIT）
    env = X2UltraBaseEnv({'model_path': 'x2_ultra.xml'})
    
    obs_dim = env.observation_dim
    action_dim = env.action_dim
    print(f"观测维度: {obs_dim}, 动作维度: {action_dim}")
    
    # 创建网络
    network = ActorCritic(action_dim=action_dim, hidden_sizes=config.hidden_sizes)
    
    # 初始化
    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)
    
    dummy_obs = jp.zeros((obs_dim,))
    params = network.init(init_rng, dummy_obs)
    
    # 优化器
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate)
    )
    opt_state = optimizer.init(params)
    
    # 创建检查点目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(config.checkpoint_dir, 'config.json'), 'w') as f:
        json.dump({k: v for k, v in config.__dict__.items() if isinstance(v, (int, float, str, list, tuple))}, f, indent=2)
    
    # 训练循环
    best_reward = -float('inf')
    
    for iteration in tqdm.tqdm(range(config.num_iterations), desc="Training"):
        iter_start = time.time()
        
        # 1. 收集数据
        rng, rollout_rng = random.split(rng)
        
        transitions, final_states, last_values = collect_rollouts_vectorized(
            env, params, network.apply, rollout_rng, config
        )
        
        # 2. 计算GAE（对每个环境分别计算，然后合并）
        all_advantages = []
        all_returns = []
        
        for i in range(config.num_envs):
            adv, ret = compute_gae(
                transitions['reward'][i],
                transitions['value'][i],
                transitions['done'][i],
                last_values[i],
                config.gamma,
                config.gae_lambda
            )
            all_advantages.append(adv)
            all_returns.append(ret)
        
        advantages = jp.stack(all_advantages)  # [N_env, T]
        returns = jp.stack(all_returns)
        
        # 3. 标准化优势
        if config.normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 4. 准备训练数据 - 展平
        obs_flat = transitions['obs'].reshape(-1, obs_dim)
        actions_flat = transitions['action'].reshape(-1, action_dim)
        log_probs_flat = transitions['log_prob'].reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        
        total_samples = obs_flat.shape[0]
        
        # 5. 训练多个epoch
        epoch_metrics = []
        
        for epoch in range(config.num_epochs):
            # 随机打乱
            rng, shuffle_rng = random.split(rng)
            perm = random.permutation(shuffle_rng, total_samples)
            
            # 应用置换
            obs_shuf = obs_flat[perm]
            actions_shuf = actions_flat[perm]
            log_probs_shuf = log_probs_flat[perm]
            adv_shuf = advantages_flat[perm]
            ret_shuf = returns_flat[perm]
            
            # minibatch训练
            num_minibatches = total_samples // config.minibatch_size
            
            for i in range(num_minibatches):
                start = i * config.minibatch_size
                end = start + config.minibatch_size
                
                batch = (
                    obs_shuf[start:end],
                    actions_shuf[start:end],
                    log_probs_shuf[start:end],
                    adv_shuf[start:end],
                    ret_shuf[start:end]
                )
                
                # 计算损失和梯度
                (loss, metrics), grads = jax.value_and_grad(
                    ppo_loss, has_aux=True
                )(params, network.apply, batch, config)
                
                # 更新参数
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                
                epoch_metrics.append(metrics)
        
        # 6. 日志
        mean_reward = float(transitions['reward'].mean())
        elapsed = time.time() - iter_start
        
        if iteration % 10 == 0:
            avg_metrics = {k: float(np.mean([m[k] for m in epoch_metrics])) 
                          for k in epoch_metrics[0].keys()}
            
            print(f"\n[Iter {iteration}] Reward: {mean_reward:.3f}, Time: {elapsed:.1f}s")
            print(f"  Policy Loss: {avg_metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {avg_metrics['value_loss']:.4f}")
            print(f"  Entropy: {avg_metrics['entropy']:.4f}")
            print(f"  Approx KL: {avg_metrics['approx_kl']:.4f}")
        
        # 7. 保存检查点
        if iteration % 100 == 0:
            ckpt = {
                'params': params,
                'opt_state': opt_state,
                'iteration': iteration,
                'config': config
            }
            checkpoints.save_checkpoint(
                config.checkpoint_dir,
                ckpt,
                step=iteration,
                keep=3
            )
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                checkpoints.save_checkpoint(
                    os.path.join(config.checkpoint_dir, 'best'),
                    ckpt,
                    step=iteration
                )
    
    # 最终保存
    final_ckpt = {'params': params, 'config': config}
    checkpoints.save_checkpoint(
        os.path.join(config.checkpoint_dir, 'final'),
        final_ckpt,
        step=config.num_iterations
    )
    
    print(f"\n训练完成！最佳奖励: {best_reward:.3f}")
    return params
# ============================================================================
# 评估
# ============================================================================
def evaluate(env: X2UltraBaseEnv, params: Dict, network: ActorCritic, 
             num_episodes: int = 10, max_steps: int = 1000):
    """评估训练好的策略"""
    
    print("\n" + "=" * 70)
    print("策略评估")
    print("=" * 70)
    
    rng = random.PRNGKey(0)
    episode_rewards = []
    
    for ep in range(num_episodes):
        rng, reset_rng = random.split(rng)
        state, obs = env.reset(reset_rng)
        
        ep_reward = 0.0
        
        for step in range(max_steps):
            # 确定性策略
            mean, _, _ = network.apply(params, obs)
            action = mean  # 无噪声
            
            rng, step_rng = random.split(rng)
            state, obs, reward, done, trunc, info = env.step(state, action, step_rng)
            
            ep_reward += float(reward)
            
            if done or trunc:
                break
        
        episode_rewards.append(ep_reward)
        print(f"Episode {ep+1}: {ep_reward:.2f}")
    
    print(f"\n平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    return episode_rewards


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
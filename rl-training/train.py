import sys

# 屏蔽 TensorFlow，防止 Windows 下与 PyTorch 冲突
sys.modules['tensorflow'] = None

import os
import gymnasium as gym
import numpy as np

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# 引入我们写好的环境
# 确保 gem_env.py 在 game 文件夹下，或者根据你的实际路径调整 import
try:
    from game.gem_env import GemTycoonEnv
except ImportError:
    # 如果都在根目录
    from gem_env import GemTycoonEnv


def make_env():
    """
    创建环境的工厂函数。
    必须包裹 Monitor，否则 VecNormalize 无法统计 Reward。
    """
    env = GemTycoonEnv()
    # 允许Monitor记录每局的各种信息 (info_keywords可以记录自定义数据)
    env = Monitor(env)
    return env


class TensorboardCallback(BaseCallback):
    """
    自定义回调函数，用于把环境里的 'game_stats' 记录到 Tensorboard。
    让我们能看到金币、回合数、最高分的变化曲线。
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 每 1000 步记录一次，节省日志空间
        if self.n_calls % 1000 == 0:
            # 获取当前环境的未归一化信息
            # 注意：因为套了 VecNormalize，直接拿 obs 是归一化过的
            # 我们从 info 或者环境内部取值
            infos = self.locals["infos"][0]
            # 也可以直接访问 env 对象 (需要解包)
            raw_env = self.training_env.envs[0].unwrapped

            self.logger.record("game/money", raw_env.money)
            self.logger.record("game/round", raw_env.round)
            self.logger.record("game/level", raw_env.level)
            self.logger.record("game/score", raw_env.score)
            self.logger.record("game/play_style", raw_env.play_style)
            self.logger.record("game/max_artifacts", raw_env.max_artifacts)
        return True


def train():
    # 创建日志目录
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 1. 创建并向量化环境
    # DummyVecEnv 适合单进程调试，如果你想多核加速可以用 SubprocVecEnv
    env = DummyVecEnv([make_env])

    # 2. 【关键修改】应用 VecNormalize
    # norm_obs=True: 归一化观察空间 (金币、分数等)
    # norm_reward=True: 归一化奖励 (让 PPO 更稳定)
    # clip_obs=10: 防止极端值破坏网络
    print("正在应用 VecNormalize 以处理数值差异...")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    # 3. 配置 MaskablePPO
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./gem_tensorboard/",

        # 核心超参数
        learning_rate=3e-4,  # 学习率
        n_steps=2048,  # 每次更新收集的步数
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # 折扣因子，0.99 关注长远利益

        # 【关键修改】熵系数 (Entropy Coefficient)
        # 提高到 0.01 (默认通常是 0.0 或 0.00x)
        # 强迫 AI 在动作概率分布上保持一定的随机性，防止过早收敛到局部最优
        ent_coef=0.01,

        # 裁剪范围
        clip_range=0.2,
    )

    print(f"开始训练... 目标步数: 1,000,000")
    print(f"Action Masking: 已启用")
    print(f"Entropy Coef: {model.ent_coef}")

    # 4. 设置回调
    # 每 50,000 步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=log_dir,
        name_prefix='gem_model'
    )

    # 合并回调
    callbacks = [checkpoint_callback, TensorboardCallback()]

    # 5. 开始训练
    try:
        model.learn(total_timesteps=1000000, callback=callbacks)
    except KeyboardInterrupt:
        print("训练被用户中断，正在保存当前进度...")

    # 6. 保存最终模型和环境统计数据
    # 注意：必须保存 env 的统计数据 (mean/var)，否则加载模型后无法正确归一化新数据
    model.save("gem_tycoon_final_model")
    env.save("gem_tycoon_vecnormalize.pkl")

    print("训练完成。模型已保存为 'gem_tycoon_final_model.zip'")
    print("环境统计数据已保存为 'gem_tycoon_vecnormalize.pkl'")


if __name__ == "__main__":
    train()
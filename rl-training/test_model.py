import sys

# 1. 屏蔽 TensorFlow
sys.modules['tensorflow'] = None

import time
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 引入环境
try:
    from game.gem_env import GemTycoonEnv, CARD_KEYS, ARTIFACT_IDS
except ImportError:
    from gem_env import GemTycoonEnv, CARD_KEYS, ARTIFACT_IDS


def decode_action(action, env):
    """将数字动作翻译为人类可读的描述"""
    if action == 0: return "🎰 旋转 (SPIN)"
    if 1 <= action <= 4:
        # 获取商店物品名称
        item = env.shop_items[action - 1]
        if item:
            name = item['id']
            price = item['price']
            return f"🛒 购买: {name} ({price}G)"
        else:
            return f"🛒 购买: 空货架 (非法)"
    if action == 5: return "🔄 刷新商店 (Reroll)"
    if action == 6: return "⏩ 下一关 (Next Round)"
    if 7 <= action <= 16: return f"🎯 指定卡牌目标: {CARD_KEYS[action - 7]}"
    if 17 <= action <= 21: return f"🔨 指定神器目标: Slot {action - 17}"
    return "UNKNOWN"


def main():
    print("正在加载模型和环境统计数据...")

    # 1. 创建并加载环境
    # 必须使用 DummyVecEnv，因为模型是在向量化环境中训练的
    env = DummyVecEnv([lambda: GemTycoonEnv()])

    # 【关键】加载训练时的归一化统计数据 (mean/var)
    # 如果没有这一步，AI 会表现得像个傻瓜
    try:
        env = VecNormalize.load("gem_tycoon_vecnormalize.pkl", env)
    except FileNotFoundError:
        print("❌ 错误：未找到 'gem_tycoon_vecnormalize.pkl'。")
        print("请先运行 train.py 进行训练！")
        return

    # 测试时不需要更新统计数据，也不需要归一化奖励（我们要看真实奖励）
    env.training = False
    env.norm_reward = False

    # 2. 加载模型
    try:
        model = MaskablePPO.load("gem_tycoon_final_model", env=env)
    except FileNotFoundError:
        print("❌ 错误：未找到模型文件 'gem_tycoon_final_model.zip'。")
        return

    print("✅ 加载成功！开始演示 AI 操作...")
    print("-" * 50)

    # 3. 开始游戏循环
    obs = env.reset()
    done = False

    # 获取内部环境实例以便读取非 Observation 数据
    raw_env = env.envs[0].unwrapped

    total_steps = 0

    while not done:
        # 获取动作掩码 (Masking)
        # 必须从环境中获取当前的 mask，否则 AI 会尝试非法操作
        action_masks = env.env_method("action_masks")[0]

        # 预测动作 (Deterministic=True 代表不使用随机性，展示最强实力)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

        # 记录动作前的状态用于打印
        current_phase = "🛍️ 商店" if raw_env.phase == 1 else "🎰 旋转"
        action_desc = decode_action(action[0], raw_env)

        # 执行一步
        obs, reward, done, info = env.step(action)

        # --- 打印解说 ---
        print(f"[{current_phase}] 步骤 {total_steps}: {action_desc}")

        # 如果是旋转，打印得分
        if action[0] == 0:
            print(f"   └── 得分: {raw_env.score} / {raw_env.target_score} (剩余次数: {raw_env.spins_left})")

        # 如果是购买魔法进入了 Pending 状态
        if raw_env.pending_state > 0:
            print(f"   └── 🤔 正在思考目标... (Pending State: {raw_env.pending_state})")

        total_steps += 1

        # 为了演示效果，如果是 Pending 状态或者重要操作，稍微停顿
        # time.sleep(0.05)

    # 4. 游戏结束总结
    print("-" * 50)
    print("🏁 游戏结束！")
    print(f"🏆 最终层数: Level {raw_env.level} - Round {raw_env.round}")
    print(f"💰 剩余金币: {raw_env.money}")
    print(f"🃏 最终卡组 ({len(raw_env.deck)}张): {raw_env.deck}")
    print(f"👑 最终神器: {raw_env.artifacts_list}")
    print(f"🎨 流派偏好: {['Normal', 'RockLover', 'GoldLover', 'GemPurist'][int(raw_env.play_style)]}")


if __name__ == "__main__":
    main()
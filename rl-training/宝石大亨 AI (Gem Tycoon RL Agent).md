# 💎 宝石大亨 AI (Gem Tycoon RL Agent)

这是一个基于深度强化学习（Deep Reinforcement Learning）的高级 AI 项目，旨在训练一个智能体精通 Roguelike 构筑游戏《宝石大亨 (Gem Tycoon)》。

本项目构建了一个**100% 逻辑拟真**的游戏环境，并采用了 **Maskable PPO**、**流派引导 (Archetype Bias)**、**动作屏蔽 (Action Masking)** 以及 **状态归一化 (Observation Normalization)** 等先进技术，成功解决了长视距规划、数值爆炸和稀疏奖励等 RL 难题。

## ✨ 核心特性

### 1. 🕹️ 硬核拟真环境 (High-Fidelity Environment)

- **像素级逻辑复刻**: 基于原始游戏源码 (`game.html`) 重构，完美还原了所有神器的运算优先级（如地质锤 vs 抛光布）、成长时序（红运当头）和利息结算点。
- **严格的状态机**: 实现了 `SPINNING` (旋转) 和 `SHOPPING` (购物) 的阶段锁定，并引入 `Pending State` 处理复杂的“购买 -> 选择目标”微操逻辑。
- **反消费主义陷阱**: 移除了简单的“购买即奖励”，强迫 AI 理解“不买”和“删卡”的长远价值。

### 2. 🧠 高级训练策略 (Advanced RL Techniques)

- **Action Masking (动作屏蔽)**: 使用 `sb3-contrib` 的 `MaskablePPO`。在神经网络输出前，强制屏蔽非法动作（如买不起、槽位满、无合法施法目标），将无效探索降至 0。
- **VecNormalize (自动归一化)**: 针对游戏数值跨度巨大（金币~10 vs 分数~1,000,000）的问题，使用 `VecNormalize` 动态调整观察空间和奖励分布，防止梯度爆炸。
- **Archetype Bias (流派引导)**: 在训练初期随机赋予 AI “性格”（如原石流爱好者、纯金币流、纯宝石流），通过特定的奖励引导 AI 探索不同的局部最优解，防止策略过早收敛。
- **Entropy Regularization**: 设置高熵系数 (`ent_coef=0.01`)，保持 AI 的探索欲。

### 3. 🎯 精细的奖励设计 (Reward Shaping)

- **对数得分奖励**: `log(score)` 适应指数级的分数增长。
- **删卡奖励**: 引导 AI 学习 Deck-building 核心的“精简卡组”策略。
- **利息奖励**: 教会 AI 延迟满足，利用“猪猪存钱罐”进行理财。

## 📂 项目结构

Bash

```
GemTycoon-RL/
├── game/
│   ├── __init__.py
│   └── gem_env.py          # 核心环境代码 (Gymnasium 接口, V6.0 最终版)
├── logs/                   # (自动生成) 模型 Checkpoints
├── gem_tensorboard/        # (自动生成) 训练日志
├── train.py                # 训练脚本 (含 MaskablePPO 与 VecNormalize 配置)
├── test_model.py           # 推理脚本 (含可视化解说与归一化加载)
├── requirements.txt        # 依赖列表
└── README.md               # 项目文档
```

## 🚀 快速开始

### 1. 环境准备

建议使用 Conda 创建独立环境以避免冲突：

Bash

```
conda create -n gem_rl python=3.10
conda activate gem_rl
```

### 2. 安装依赖

本项目采用 PyTorch 后端（Windows 友好）：

Bash

```
pip install -r requirements.txt
```

### 3. 开始训练

运行训练脚本。脚本会自动启用 `VecNormalize` 和 `Action Masking`。

Bash

```
python train.py
```

*训练产物 `gem_tycoon_final_model.zip` 和 `gem_tycoon_vecnormalize.pkl` 将保存在根目录。*

### 4. 监控训练

实时查看胜率、奖励曲线和流派分布：

Bash

```
tensorboard --logdir ./gem_tensorboard/
```

### 5. 测试与评估 (Showtime!)

训练完成后，运行测试脚本。它会加载训练好的模型和归一化参数，并以文字流形式“解说” AI 的操作。

Bash

```
python test_model.py
```

**运行效果示例：**

Plaintext

```
[🛍️ 商店] 步骤 45: 🛒 购买: magic_box (20G)
[🛍️ 商店] 步骤 46: ⏩ 下一关 (Next Round)
[🎰 旋转] 步骤 47: 🎰 旋转 (SPIN)
   └── 得分: 2450 / 500 (剩余次数: 3)
[🛍️ 商店] 步骤 52: 🛒 购买: purify (15G)
   └── 🤔 正在思考目标... (Pending State: 1.0)
[🛍️ 商店] 步骤 53: 🎯 指定卡牌目标: rock
```

## 🛠️ 技术细节：观察与动作空间

### 动作空间 (Action Space)

`Discrete(22)` - 移除了“取消”动作，强制 AI 决策：

- **0**: 旋转 (Spin)
- **1-4**: 购买 (Buy)
- **5**: 刷新 (Reroll)
- **6**: 下一关 (Next)
- **7-16**: 指定卡牌目标 (Target Card)
- **17-21**: 指定神器槽位 (Target Artifact)

### 观察空间 (Observation Space)

复合字典空间 `Dict`：

- `deck_counts`: 卡组分布向量。
- `artifacts`: 已持有神器 (Multi-Hot)。
- `game_stats`: 包含金币、归一化分数、关卡、`play_style` (流派变量) 等 13 维状态。
- `shop_availability`: 商店货架状态。

## 📈 进阶调优

如果您想进一步提升 AI 性能，可以在 `train.py` 中调整以下参数：

- **`ent_coef`**: 增加此值可让 AI 更疯狂地探索新策略。
- **`gamma`**: 默认为 0.99。如果希望 AI 极度关注大后期（Level 20），可尝试 0.995。
- **`clip_range`**: PPO 的裁剪范围，通常 0.1-0.3 之间。

------

**Happy Training! 愿 AI 早日发现人类未曾设想的“无限倍率”之路！** 🎰
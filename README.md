# Gem Tycoon - 宝石大亨

老虎机 × Roguelike × 卡组构筑的单页网页游戏。

## 快速开始

直接在浏览器中打开 `Gem Tycoon：Demo.html` 即可游玩，无需安装任何依赖。

> 需要联网（CDN 加载 React 18 / Tailwind CSS / Lucide Icons）

## 游戏玩法

- **旋转老虎机**：每次生成 3×3 共 9 张卡牌，满足特定排列可触发倍率加成
- **构筑卡组**：在商店购买新卡牌，删除弱牌，打造高效得分引擎
- **收集神器**：被动道具彻底改变计分规则，组合出不同流派
- **魔法物品**：一次性消耗品，用于转化、复制、清除卡牌
- **闯关模式**：每个关卡 4 回合，达标分数递增，撑得越久分数越高

## 项目结构

```
GemTycoon-Demo/
├── Gem Tycoon：Demo.html    # 游戏本体（单文件，React + Tailwind）
├── 游戏百科.md              # 卡牌 / 神器 / 魔法 / 排列系统数据
├── rl-training/             # 强化学习 AI 训练代码（Python）
│   ├── train.py
│   ├── test_model.py
│   ├── requirements.txt
│   ├── game/gem_env.py
│   └── 宝石大亨 AI (Gem Tycoon RL Agent).md
└── README.md
```

## 游戏百科

完整的卡牌数据、神器效果、魔法物品说明、排列系统和计分公式详见 [游戏百科.md](游戏百科.md)。

## 强化学习 AI

`rl-training/` 目录包含基于 Maskable PPO 的强化学习训练代码，使用 Gymnasium 环境模拟游戏逻辑。详见 [rl-training/](rl-training/) 目录。

## 技术栈

- React 18 (esm.sh CDN)
- Tailwind CSS (CDN)
- Lucide React Icons
- Web Audio API（音效合成）

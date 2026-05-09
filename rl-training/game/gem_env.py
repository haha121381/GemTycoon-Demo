import sys

sys.modules['tensorflow'] = None
import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

# --- 常量定义 ---
#
CARD_DEFINITIONS = {
    'sapphire': {'chips': 10, 'mult': 0, 'type': 'gem', 'price': 11, 'rarity': 1},
    'emerald': {'chips': 15, 'mult': 0, 'type': 'gem', 'price': 13, 'rarity': 1},
    'ruby': {'chips': 20, 'mult': 0, 'type': 'gem', 'price': 16, 'rarity': 2},
    'diamond': {'chips': 20, 'mult': 2, 'type': 'gem', 'price': 19, 'rarity': 3},
    'pink_diamond': {'chips': 0, 'mult': 5, 'type': 'gem', 'price': 14, 'rarity': 3},
    'amethyst': {'chips': 0, 'mult': 0, 'type': 'gem', 'price': 20, 'rarity': 2, 'x_mult': 2},
    'topaz': {'chips': 10, 'mult': 1, 'type': 'gem', 'price': 18, 'rarity': 3, 'gold': 1},
    'rainbow': {'chips': 0, 'mult': 0, 'type': 'gem', 'price': 17, 'rarity': 3, 'wild': True},
    'coin': {'chips': 0, 'mult': 0, 'type': 'gold', 'price': 12, 'rarity': 1, 'gold': 3},
    'rock': {'chips': 0, 'mult': 0, 'type': 'rock', 'price': 1, 'rarity': 0},
}
CARD_KEYS = list(CARD_DEFINITIONS.keys())

#
ARTIFACT_IDS = [
    'dazzling', 'magic_box', 'ore_specimen', 'angle_grinder', 'missing_heart',
    'blue_green_blind', 'old_eyes', 'pink_promise', 'red_luck', 'perfectionist',
    'purple_air', 'shopaholic', 'purple_gold_vip', 'piggy_bank', 'platinum_vip',
    'black_gold_vip', 'colorful_vip', 'hourglass', 'lens', 'polisher',
    'hammer', 'greedy_hand', 'philosophers_stone', 'shiny_gold', 'more_is_better'
]

MAGIC_ITEMS = [
    'purify', 'vanish', 'clone', 'chaos_stew', 'rock_curse',
    'surprise_gift', 'dimensional_pocket', 'chaos_spawn',
    'windfall', 'midas_touch', 'alchemy_pot', 'gambling_stone',
    'tnt', 'salvage', 'turn_tables', 'reforge_anvil'
]

MAGIC_PRICES = {
    'purify': 15, 'vanish': 25, 'clone': 15, 'chaos_stew': 10, 'rock_curse': 15,
    'surprise_gift': 20, 'dimensional_pocket': 30, 'chaos_spawn': 25,
    'windfall': 8, 'midas_touch': 20, 'alchemy_pot': 10, 'gambling_stone': 10,
    'tnt': 20, 'salvage': 10, 'turn_tables': 15, 'reforge_anvil': 12
}


class GemTycoonEnv(gym.Env):
    def __init__(self):
        super(GemTycoonEnv, self).__init__()
        # Actions:
        # 0: Spin
        # 1-4: Buy
        # 5: Reroll
        # 6: Next Round
        # 7-16: Target Card Type (10种)
        # 17-21: Target Artifact Slot (5种)
        # [修改] 移除了 Action 22 (Cancel)
        self.action_space = spaces.Discrete(22)

        self.observation_space = spaces.Dict({
            "deck_counts": spaces.Box(low=0, high=999, shape=(len(CARD_KEYS),), dtype=np.int32),
            "artifacts": spaces.MultiBinary(len(ARTIFACT_IDS)),
            "game_stats": spaces.Box(low=0, high=np.inf, shape=(13,), dtype=np.float32),
            "shop_availability": spaces.MultiBinary(4)
        })
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.deck = ['sapphire'] * 2 + ['emerald'] * 2 + ['coin'] * 2 + ['rock'] * 4
        self.money = 10
        self.artifacts_list = []

        self.round = 1
        self.level = 1
        self.score = 0
        self.target_score = 200
        self.spins_left = 4
        self.max_spins = 4

        self.max_artifacts = 5
        self.shop_rerolls = 1
        self.dim_pocket_count = 0

        self.red_luck_growth = 0.0
        self.lens_growth = 0.0

        self.pending_magic = None
        self.pending_item_idx = -1
        self.pending_state = 0.0

        self.shop_items = [None] * 4
        self.phase = 0

        r = random.random()
        if r < 0.7:
            self.play_style = 0.0
        elif r < 0.8:
            self.play_style = 1.0
        elif r < 0.9:
            self.play_style = 2.0
        else:
            self.play_style = 3.0

        self._generate_shop()
        return self._get_obs(), {}

    def action_masks(self):
        """Action Masking"""
        mask = np.zeros(22, dtype=bool)  # Size reduced

        # 1. Pending 状态 (必须选择目标)
        if self.pending_state > 0:
            # [修改] 不再允许 Cancel (Action 22 removed)
            if self.pending_state == 1.0:  # Card Target
                for i, key in enumerate(CARD_KEYS):
                    if key in self.deck: mask[7 + i] = True
            elif self.pending_state == 2.0:  # Artifact Target
                for i in range(len(self.artifacts_list)): mask[17 + i] = True
            return mask

        # 2. Spinning 阶段
        if self.phase == 0:
            if self.spins_left > 0: mask[0] = True
            return mask

        # 3. Shopping 阶段
        if self.phase == 1:
            for i in range(4):
                if self.shop_items[i] is not None:
                    item = self.shop_items[i]
                    cost = self._calculate_price(item)
                    can_afford = self.money >= cost
                    has_space = True
                    is_usable = True  # [新增] 检查物品是否可用

                    if item['type'] == 'artifact' and len(self.artifacts_list) >= self.max_artifacts:
                        has_space = False
                    if item['type'] == 'magic' and item['id'] == 'dimensional_pocket' and self.dim_pocket_count >= 5:
                        has_space = False

                    # [关键修改] 如果没有合法目标，直接禁止购买
                    if item['type'] == 'magic':
                        # 回收/重铸 需要有神器
                        if item['id'] in ['salvage', 'reforge_anvil'] and len(self.artifacts_list) == 0:
                            is_usable = False
                        # 删卡/复制 理论上只要有卡就行，但作为保险可以检查 deck 长度
                        if item['id'] in ['purify', 'vanish', 'clone'] and len(self.deck) == 0:
                            is_usable = False

                    if can_afford and has_space and is_usable:
                        mask[1 + i] = True

            reroll_cost = 2 if self._has_art('purple_gold_vip') else 5
            if self.shop_rerolls > 0 and self.money >= reroll_cost: mask[5] = True
            mask[6] = True
            return mask

        return mask

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # --- 1. Pending Magic State (No Cancel) ---
        if self.pending_state > 0:
            res = self._handle_pending_action(action)
            if res['success']:
                item = self.shop_items[self.pending_item_idx]
                cost = self._calculate_price(item)
                self.money -= cost
                self.shop_items[self.pending_item_idx] = None
                self._clear_pending()

                reward += 1.0
                if self.play_style == 3.0 and self.pending_magic in ['purify', 'vanish', 'tnt']:
                    reward += 2.0
            else:
                # [保护机制] 如果 Masking 正常工作，这里不该发生。
                # 但万一发生了，为了防止卡死，扣大分并强制退出 Pending (放弃购买)
                reward -= 5.0
                self._clear_pending()  # 强制结束防止死循环
            return self._get_obs(), reward, terminated, truncated, {}

        # --- 2. Phase: SPINNING ---
        if self.phase == 0:
            if action == 0:  # Spin
                spin_score = self._handle_spin()
                reward += np.log1p(spin_score) * 0.5

                if self.play_style == 1.0 and self._has_art('hammer') and 'rock' in self.deck: reward += 0.5

                if self.spins_left <= 0:
                    if self.score >= self.target_score:
                        self._enter_shop_phase()
                        reward += 5.0
                    else:
                        terminated = True;
                        reward -= 10.0
            else:
                reward -= 5.0

        # --- 3. Phase: SHOPPING ---
        elif self.phase == 1:
            if 1 <= action <= 4:  # Buy
                idx = action - 1
                item = self.shop_items[idx]
                cost = self._calculate_price(item)

                if item['type'] == 'magic' and item['id'] in ['purify', 'vanish', 'clone', 'salvage', 'reforge_anvil']:
                    self.pending_item_idx = idx
                    self.pending_magic = item['id']
                    self.pending_state = 2.0 if item['id'] in ['salvage', 'reforge_anvil'] else 1.0
                    reward += 0.0
                else:
                    self.money -= cost
                    self._acquire_item(item)
                    self.shop_items[idx] = None

                    if self.play_style == 1.0 and item['id'] in ['hammer', 'philosophers_stone', 'ore_specimen',
                                                                 'rock_curse']: reward += 5.0
                    if self.play_style == 2.0 and item['id'] in ['piggy_bank', 'shiny_gold',
                                                                 'greedy_hand']: reward += 5.0
                    if self.play_style == 3.0 and item['type'] == 'gem': reward += 1.0

            elif action == 5:  # Reroll
                cost = 2 if self._has_art('purple_gold_vip') else 5
                self.money -= cost
                self.shop_rerolls -= 1
                self._generate_shop()

            elif action == 6:  # Next
                interest = self._next_level_logic()
                self.phase = 0
                if interest > 0: reward += interest * 0.1
                if self.play_style == 2.0 and interest >= 3: reward += 2.0
                reward += 1.0

        if self.level > 20: terminated = True; reward += 100.0
        return self._get_obs(), reward, terminated, truncated, {}

    # --- Helpers ---
    def _enter_shop_phase(self):
        self.phase = 1;
        self.shop_rerolls = 3 if self._has_art('shopaholic') else 1;
        self._generate_shop()

    def _next_level_logic(self):
        interest = 0
        if self._has_art('piggy_bank'): interest = self.money // 10; self.money += interest
        self.round += 1
        if self.round > 4: self.round = 1; self.level += 1
        if self.level == 1:
            self.target_score = [200, 250, 350, 500][min(self.round - 1, 3)]
        else:
            fac, add = 1.2, 200
            if 6 <= self.level <= 9:
                fac, add = 1.25, 100
            elif self.level >= 10:
                fac, add = 1.4, 50
            self.target_score = int(self.target_score * fac) + add
        self.spins_left = self.max_spins + (1 if self._has_art('hourglass') else 0)
        self.score = 0;
        return interest

    def _handle_spin(self):
        self.spins_left -= 1
        if self._has_art('lens'): self.lens_growth += 2.0
        grid_cards = random.choices(self.deck, k=9)
        spin_chips = 0;
        spin_mult = 0;
        gem_bonus = 10 if self._has_art('polisher') else 0;
        gold_bonus = 10 if self._has_art('greedy_hand') else 0
        grid_objs = [];
        amethyst_cnt = 0;
        ruby_cnt = 0

        for c_name in grid_cards:
            base = CARD_DEFINITIONS[c_name];
            obj = {'id': c_name, 'type': base['type'], 'chips': base['chips'], 'mult': base['mult'],
                   'wild': base.get('wild', False), 'gold': base.get('gold', 0), 'is_gem': base['type'] == 'gem'}
            if self._has_art('old_eyes') and obj['type'] in ['rock', 'gold']: obj['is_gem'] = True
            if obj['is_gem']: obj['chips'] += gem_bonus
            if obj['id'] == 'coin': obj['chips'] += gold_bonus
            if obj['id'] == 'rock' and self._has_art('hammer'): obj['chips'] = 10
            if obj['is_gem'] and self._has_art('angle_grinder'): obj['mult'] += 1
            if obj['id'] == 'rock' and self._has_art('philosophers_stone'): obj['mult'] += 3
            if obj['id'] == 'coin' and self._has_art('shiny_gold'): obj['mult'] += 0.8
            if c_name == 'amethyst': amethyst_cnt += 1
            if c_name == 'ruby': ruby_cnt += 1
            grid_objs.append(obj)
        for o in grid_objs: spin_chips += o['chips']; spin_mult += o['mult']; self.money += o['gold']
        patterns = [([0, 1, 2], 7, "R"), ([3, 4, 5], 7, "R"), ([6, 7, 8], 7, "R"), ([0, 3, 6], 7, "C"),
                    ([1, 4, 7], 7, "C"), ([2, 5, 8], 7, "C"), ([0, 4, 8], 8, "D"), ([2, 4, 6], 8, "D"),
                    ([0, 1, 3, 4], 10, "S"), ([1, 2, 4, 5], 10, "S"), ([3, 4, 6, 7], 10, "S"), ([4, 5, 7, 8], 10, "S"),
                    ([0, 2, 6, 8], 15, "Cor"), ([1, 3, 5, 7], 15, "Cr")]
        has_dazzling = self._has_art('dazzling');
        has_missing = self._has_art('missing_heart');
        has_bg = self._has_art('blue_green_blind')
        for idxs, val, name in patterns:
            if self._check_synergy([grid_objs[i] for i in idxs], has_bg):
                b = val * 2 if has_dazzling else val
                if name == "Cr" and has_missing: b += 55
                spin_mult += b
        if self._has_art('magic_box'): spin_mult += len(self.deck) * 2
        if self._has_art('ore_specimen'): spin_chips += self.deck.count('rock') * 15
        if self._has_art('more_is_better'): spin_chips += len(self.artifacts_list) * 25
        if self._has_art('red_luck'): spin_chips += self.red_luck_growth; spin_mult += self.red_luck_growth
        if self._has_art('lens'): spin_mult += self.lens_growth
        if self._has_art('pink_promise') and any(c['id'] == 'pink_diamond' for c in grid_objs): spin_mult *= 2
        if self._has_art('perfectionist') and 'rock' not in self.deck: spin_mult = int(spin_mult * 1.5)
        if amethyst_cnt > 0: fac = 3 if self._has_art('purple_air') else 2; spin_mult *= (fac ** amethyst_cnt)
        self.score += int(spin_chips * spin_mult)
        if self._has_art('red_luck'): self.red_luck_growth += ruby_cnt
        return int(spin_chips * spin_mult)

    def _generate_shop(self):
        self.shop_items = [];
        slots = 4 if self._has_art('black_gold_vip') else 3;
        free_idx = random.randint(0, slots - 1) if self._has_art('colorful_vip') else -1
        valid_magic = [m for m in MAGIC_ITEMS if m != 'dimensional_pocket' or self.dim_pocket_count < 5]
        for i in range(slots):
            r = random.random();
            item = None
            if r < 0.4:
                key = random.choice(CARD_KEYS); price = CARD_DEFINITIONS[key]['price']; item = {'type': 'card',
                                                                                                'id': key,
                                                                                                'price': random.randint(
                                                                                                    1,
                                                                                                    20) if key == 'rock' else price}
            elif r < 0.7:
                avail = [a for a in ARTIFACT_IDS if a not in self.artifacts_list]
                if avail: key = random.choice(avail); item = {'type': 'artifact', 'id': key, 'price': 20}
            else:
                if valid_magic: key = random.choice(valid_magic); item = {'type': 'magic', 'id': key,
                                                                          'price': MAGIC_PRICES[key]}
            if item:
                if self._has_art('platinum_vip'): item['price'] = max(1, item['price'] - 5)
                if i == free_idx: item['price'] = 0
                self.shop_items.append(item)
            else:
                self.shop_items.append(None)

    def _acquire_item(self, item):
        if item['type'] == 'card':
            self.deck.append(item['id'])
        elif item['type'] == 'artifact':
            self.artifacts_list.append(item['id'])
        elif item['type'] == 'magic':
            self._apply_instant_magic(item['id'])

    def _apply_instant_magic(self, mid):
        if mid == 'chaos_stew':
            self.deck = [random.choice(CARD_KEYS) for _ in self.deck]
        elif mid == 'alchemy_pot':
            self.money += self.deck.count('coin') * 3
        elif mid == 'chaos_spawn':
            if 'rainbow' in self.deck: self.deck.remove('rainbow'); self.deck.extend(
                [k for k in CARD_KEYS if k != 'rainbow'])
        elif mid == 'windfall':
            self.money += min(self.money, 50)
        elif mid == 'rock_curse':
            self.deck.extend(['rock'] * 5)
        elif mid == 'tnt':
            cnt = self.deck.count('rock'); self.deck = [c for c in self.deck if c != 'rock']; self.money += cnt * 5
        elif mid == 'transmute_rock':
            if 'rock' in self.deck: self.deck.remove('rock'); self.deck.append(
                random.choice(['sapphire', 'emerald', 'ruby', 'diamond']))
        elif mid == 'gambling_stone':
            new_deck = [];
            for c in self.deck:
                if c == 'coin':
                    new_deck.append(
                        random.choice(['rock'] + [k for k in CARD_KEYS if CARD_DEFINITIONS[k]['type'] == 'gem']))
                else:
                    new_deck.append(c)
            self.deck = new_deck
        elif mid == 'dimensional_pocket':
            self.max_artifacts += 1; self.dim_pocket_count += 1
        elif mid == 'turn_tables':
            if len(self.artifacts_list) >= 2: i1, i2 = random.sample(range(len(self.artifacts_list)), 2);
            self.artifacts_list[i1], self.artifacts_list[i2] = self.artifacts_list[i2], self.artifacts_list[i1]
        elif mid == 'surprise_gift':
            if random.random() < 0.5:
                avail = [a for a in ARTIFACT_IDS if a not in self.artifacts_list]
                if avail and len(self.artifacts_list) < self.max_artifacts:
                    self.artifacts_list.append(random.choice(avail))
                else:
                    self.deck.append(random.choice(CARD_KEYS))
            else:
                self.deck.append(random.choice(CARD_KEYS))

    def _handle_pending_action(self, action):
        # [修改] 不再响应 Action 22
        magic = self.pending_magic
        if self.pending_state == 1.0 and 7 <= action <= 16:
            target = CARD_KEYS[action - 7]
            if magic == 'purify' and target in self.deck: self.deck.remove(target); return {'success': True,
                                                                                            'cancel': False}
            if magic == 'vanish' and target in self.deck: self.deck = [c for c in self.deck if c != target]; return {
                'success': True, 'cancel': False}
            if magic == 'clone' and target in self.deck: self.deck.append(target); return {'success': True,
                                                                                           'cancel': False}
        elif self.pending_state == 2.0 and 17 <= action <= 21:
            idx = action - 17
            if idx < len(self.artifacts_list):
                if magic == 'salvage': self.artifacts_list.pop(idx); self.money += 15; return {'success': True,
                                                                                               'cancel': False}
                if magic == 'reforge_anvil': self.artifacts_list.pop(idx); avail = [a for a in ARTIFACT_IDS if
                                                                                    a not in self.artifacts_list];
                if avail: self.artifacts_list.append(random.choice(avail))
                return {'success': True, 'cancel': False}
        return {'success': False, 'cancel': False}

    def _check_synergy(self, cards, bg_blind):
        non_wilds = [c for c in cards if not c['wild']];
        if not non_wilds: return True
        first = non_wilds[0];
        if any(not c['is_gem'] for c in non_wilds): return False
        target = first['id']
        for c in non_wilds:
            if c['id'] == target: continue
            if bg_blind and target in ['sapphire', 'emerald'] and c['id'] in ['sapphire', 'emerald']: continue
            return False
        return True

    def _has_art(self, aid):
        return aid in self.artifacts_list

    def _calculate_price(self, item):
        return item['price']

    def _clear_pending(self):
        self.pending_magic = None; self.pending_item_idx = -1; self.pending_state = 0.0

    def _get_obs(self):
        counts = np.zeros(len(CARD_KEYS), dtype=np.int32)
        for c in self.deck: counts[CARD_KEYS.index(c)] += 1
        arts = np.zeros(len(ARTIFACT_IDS), dtype=np.int8)
        for a in self.artifacts_list: arts[ARTIFACT_IDS.index(a)] = 1
        stats = np.array([
            self.money, self.score, self.target_score, self.spins_left, self.level,
            self.red_luck_growth, self.lens_growth, self.shop_rerolls, self.max_artifacts,
            self.pending_state, self.dim_pocket_count, self.phase, self.play_style
        ], dtype=np.float32)
        shop = np.zeros(4, dtype=np.int8)
        for i, it in enumerate(self.shop_items):
            if it: shop[i] = 1
        return {"deck_counts": counts, "artifacts": arts, "game_stats": stats, "shop_availability": shop}
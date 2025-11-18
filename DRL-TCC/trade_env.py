# trade_env.py
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

@dataclass
class Order:
    side: str      # "bid" ou "ask"
    price: float
    qty: int
    birth_step: int  # passo em que a ordem foi colocada

class TradeEngineEnv(gym.Env):
    """
    Market making passivo com ordens persistentes por lado.
    - 4 ações: BID, ASK, BOTH, NO_QUOTE
    - Até N ordens por lado vivas (FIFO)
    - Cancelamento automático após TTL de ticks
    - Recompensa: ΔMTM em t+1 (com damping) - penalidade de inventário
    - Arredondamento por tick (bid down, ask up)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        log_name,
        log_dir: str | None = None,
        log_prefix: str = "train_log_",
        quote_size: int = 100,
        spread: float = 0.01,
        tick_size: float = 0.01, # tick size de ações acima de R$1,00
        max_inventory: int = 1_000,
        max_steps: int | None = None,
        dumped_factor: float = 0.2,
        inventory_factor: float = 2e-5,
        starting_cash: float = 100_000.0,
        # novos parâmetros de persistência
        max_orders_per_side: int = 1,
        order_ttl_ticks: int = 1,
        # reward shaping
        mtm_weight: float = 0.005,
        pos_realized_weight: float = 5,
        neg_realized_weight: float = 2,
        inventory_reduction_weight: float = 0.05,
        near_miss_weight: float = 0.01,
        no_quote_inventory_penalty: float = 0.05,
        no_quote_streak_penalty: float = 0.05,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        base_log_dir = Path(log_dir) if log_dir else Path(__file__).resolve().parent / "train_logs"
        base_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = str(base_log_dir)
        self.log_prefix = log_prefix

        # Detecta presença de colunas de trade
        self.has_trades = ('md_entry_px' in self.df.columns) and ('aggressor' in self.df.columns)

        # Normaliza melhor bid/ask para todas as linhas (inclui trades)
        # Usa en_b_px_1/en_s_px_1 quando houver; caso NaN, usa top_px_bid/top_px_offer se existirem;
        # por fim, faz forward-fill para cobrir linhas de trade sem book explícito.
        bid_series = self.df['en_b_px_1'] if 'en_b_px_1' in self.df.columns else pd.Series(index=self.df.index, dtype=float)
        ask_series = self.df['en_s_px_1'] if 'en_s_px_1' in self.df.columns else pd.Series(index=self.df.index, dtype=float)
        # Forward fill para cobrir linhas sem book
        self.df['best_bid'] = bid_series.ffill()
        self.df['best_ask'] = ask_series.ffill()

        need_px = ['best_bid', 'best_ask']
        assert all(c in self.df.columns for c in need_px), \
            "DataFrame precisa ter colunas de topo do livro (en_b_px_1/en_s_px_1)"

        self.df_length = len(self.df)

        self.quote_size = int(quote_size)
        self.spread = float(spread)
        self.tick_size = float(tick_size)
        self.max_inventory = int(max_inventory)
        self.max_steps = max_steps if max_steps is not None else self.df_length - 1

        self.dumped_factor = float(dumped_factor)
        self.inventory_factor = float(inventory_factor)

        # persistência
        self.max_orders_per_side = int(max_orders_per_side)
        self.order_ttl_ticks = int(order_ttl_ticks)
        # reward shaping knobs
        self.no_quote_inventory_penalty = float(no_quote_inventory_penalty)
        self.no_quote_streak_penalty = float(no_quote_streak_penalty)
        self.inventory_reduction_weight = float(inventory_reduction_weight)
        self.near_miss_weight = float(near_miss_weight)
        self.mtm_weight = float(mtm_weight)
        self.pos_realized_weight = float(pos_realized_weight)
        self.neg_realized_weight = float(neg_realized_weight)

        # Tamanhos (opcional). Faz forward-fill se existirem.
        self.use_sizes = ('en_b_sz_1' in self.df.columns) and ('en_s_sz_1' in self.df.columns)
        if self.use_sizes:
            self.df['best_bid_sz'] = self.df['en_b_sz_1'].ffill()
            self.df['best_ask_sz'] = self.df['en_s_sz_1'].ffill()
            
        obs_shape = (5,) if self.use_sizes else (3,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # Ações: 0 = QUOTE_BID, 1 = QUOTE_ASK, 2 = QUOTE_BOTH, 3 = NO_QUOTE
        self.action_space = spaces.Discrete(4)

        # Estado
        self.current_step = 0
        self.inventory = 0
        self.cash = float(starting_cash)
        self.starting_cash = float(starting_cash)
        self.prev_mtm = 0.0
        self.done = False

        # Ordens vivas
        self.working_bids: List[Order] = []
        self.working_asks: List[Order] = []

        self.logs = []
        self.log_name = log_name
        self.no_quote_streak = 0
        self._last_normalized_reward = None

    # ---------- helpers ----------
    def _row(self, idx: int):
        return self.df.iloc[idx]
    
    def _truncate_to_two_decimals(self, x: float) -> float:
        return int(x * 100) / 100.0

    def _mid(self, bid_px: float, ask_px: float):
        mid = (bid_px + ask_px) / 2.0
        return self._truncate_to_two_decimals(mid)

    def _get_obs(self, idx: int | None = None):
        if idx is None:
            idx = self.current_step
        r = self._row(idx)
        inventory_ratio = self.inventory / self.max_inventory
        if self.use_sizes:
            # Usa séries forward-filled
            obs = np.array([
                r.get('best_bid_sz', np.nan), r['best_bid'],
                r['best_ask'], r.get('best_ask_sz', np.nan),
                inventory_ratio
            ], dtype=np.float32)
        else:
            obs = np.array([
                r['best_bid'], r['best_ask'], inventory_ratio
            ], dtype=np.float32)
        return obs

    def _mark_to_market(self, bid_px: float, ask_px: float):
        """
        Mark-to-market inventory using executable prices to flatten to zero now.
        - If long (inventory > 0): mark at current bid (sell into bid).
        - If short (inventory < 0): mark at current ask (buy back at ask).
        - If flat: cash only.
        """
        inv = self.inventory
        if inv > 0:
            px = bid_px
        elif inv < 0:
            px = ask_px
        else:
            return self.cash
        return self.cash + inv * px

    def _round_down_tick(self, x: float) -> float:
        t = self.tick_size
        return np.floor(x / t) * t

    def _round_up_tick(self, x: float) -> float:
        t = self.tick_size
        return np.ceil(x / t) * t

    def _round_to_tick(self, x: float) -> float:
        t = self.tick_size
        return np.round(x / t) * t

    # ---------- core MM helpers ----------
    def _model_prices(self, bid_t: float, ask_t: float):
        mid_t = self._mid(bid_t, ask_t)
        spread_tick = max(self.tick_size, self._round_to_tick(self.spread))
        model_bid = self._round_down_tick(mid_t - spread_tick)
        model_ask = self._round_up_tick(mid_t + spread_tick)
        return model_bid, model_ask, mid_t

    def _place_order(self, side: str, price: float):
        """Coloca 1 ordem (qty fixa) e respeita o limite por lado (FIFO)."""
        order = Order(side=side, price=price, qty=self.quote_size, birth_step=self.current_step)
        if side == "bid":
            self.working_bids.append(order)
            # mantém no máximo N: se exceder, remove a mais antiga (FIFO)
            if len(self.working_bids) > self.max_orders_per_side:
                self.working_bids.pop(0)
        else:
            self.working_asks.append(order)
            if len(self.working_asks) > self.max_orders_per_side:
                self.working_asks.pop(0)

    def _age_and_cancel(self):
        """Remove ordens cujo TTL foi atingido (≥ order_ttl_ticks)."""
        cutoff = self.current_step - self.order_ttl_ticks
        self.working_bids = [o for o in self.working_bids if o.birth_step > cutoff]
        self.working_asks = [o for o in self.working_asks if o.birth_step > cutoff]

    # ---------- gym api ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = 0
        self.cash = self.starting_cash
        self.prev_mtm = 0.0
        self.done = False
        self.working_bids.clear()
        self.working_asks.clear()
        self.logs = []
        self.no_quote_streak = 0
        self._last_normalized_reward = None

        r0 = self._row(self.current_step)
        bid0 = float(r0['best_bid'])
        ask0 = float(r0['best_ask'])
        self.prev_mtm = self._mark_to_market(bid0, ask0)
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        if self.done:
            return self._get_obs(), 0.0, False, True, {}

        # preços em t
        row_t = self._row(self.current_step)
        bid_t = float(row_t['best_bid'])
        ask_t = float(row_t['best_ask'])

        # cotações modelo (grid do tick)
        model_bid, model_ask, mid_t = self._model_prices(bid_t, ask_t)

        # 1) Decisão de AÇÃO (não cancela o que já existe; apenas adiciona)
        # 0=QUOTE_BID, 1=QUOTE_ASK, 2=QUOTE_BOTH, 3=NO_QUOTE
        # garante tipo/int válido
        try:
            action = int(action)
        except Exception:
            # caso extremo: se não converter, ignora cotação (NO_QUOTE)
            action = 3
        if action in (0, 2):  # BID
            self._place_order("bid", model_bid)
        if action in (1, 2):  # ASK
            self._place_order("ask", model_ask)
            
        inv_norm = abs(self.inventory) / self.max_inventory
        inv_reduction_reward = 0.0
        if self.inventory > 0 and action in (1, 2): # ASK/BOTH
            inv_reduction_reward = self.inventory_reduction_weight * inv_norm
        if self.inventory < 0 and action in (0, 2): # BID/BOTH
            inv_reduction_reward = self.inventory_reduction_weight * inv_norm

        # 2) Avança para t+1 (execução passiva acontece comparando com t+1)
        next_step = self.current_step + 1
        if next_step >= self.df_length:
            self.done = True
            info = {}
            return self._get_obs(), 0.0, False, True, info

        row_t1 = self._row(next_step)
        bid_t1 = float(row_t1['best_bid'])
        ask_t1 = float(row_t1['best_ask'])
        mid_t1 = self._mid(bid_t1, ask_t1)

        # 3) Cancela por TTL (ordens muito velhas)
        self._age_and_cancel()

        # 4) Execução em t+1 (pode executar várias ordens por lado)
        exec_buy_qty = 0
        exec_sell_qty = 0
        realized_this_step = 0.0

        # Se houver um trade em t+1, podemos executar a preço do trade conforme agressor
        trade_px = None
        trade_aggr = None
        trade_sz = None
        trade_filled_buy = 0
        trade_filled_sell = 0
        if self.has_trades:
            try:
                trade_px_val = row_t1.get('md_entry_px', np.nan)
                aggr_val = row_t1.get('aggressor', np.nan)
                size_val = row_t1.get('md_entry_size', np.nan)
                if pd.notna(trade_px_val) and pd.notna(aggr_val):
                    trade_px = float(trade_px_val)
                    trade_aggr = int(aggr_val)
                    trade_sz = float(size_val) if pd.notna(size_val) else None
            except Exception:
                trade_px = None
                trade_aggr = None
                trade_sz = None

        # BIDs executam por trade (aggressor 3 = vendedor bateu bid) ou book-cross (ask_t1 <= price)
        alive_bids: List[Order] = []
        for o in self.working_bids:
            filled = False
            # tentativa por trade
            if (trade_px is not None) and (trade_aggr == 3):
                if abs(o.price - trade_px) < 1e-12:
                    # define qty que pode executar (respeita inventory e tamanho do trade se fornecido)
                    possible_qty = o.qty
                    if trade_sz is not None:
                        possible_qty = int(min(possible_qty, max(0, trade_sz - trade_filled_buy)))
                    if possible_qty > 0 and (self.inventory + possible_qty <= self.max_inventory):
                        cost = possible_qty * o.price
                        self.cash -= cost
                        self.inventory += possible_qty
                        exec_buy_qty += possible_qty
                        trade_filled_buy += possible_qty
                        realized_this_step += possible_qty * (mid_t1 - o.price)
                        filled = True
            # fallback por book cross
            if (not filled) and (ask_t1 <= o.price) and (self.inventory + o.qty <= self.max_inventory):
                cost = o.qty * o.price
                self.cash -= cost
                self.inventory += o.qty
                exec_buy_qty += o.qty
                realized_this_step += o.qty * (mid_t1 - o.price)
                filled = True

            if not filled:
                alive_bids.append(o)
        self.working_bids = alive_bids

        # ASKs executam por trade (aggressor 4 = comprador tomou offer) ou book-cross (bid_t1 >= price)
        alive_asks: List[Order] = []
        for o in self.working_asks:
            filled = False
            # tentativa por trade
            if (trade_px is not None) and (trade_aggr == 4):
                if abs(o.price - trade_px) < 1e-12:
                    possible_qty = o.qty
                    if trade_sz is not None:
                        possible_qty = int(min(possible_qty, max(0, trade_sz - trade_filled_sell)))
                    if possible_qty > 0 and (self.inventory - possible_qty >= -self.max_inventory):
                        proceeds = possible_qty * o.price
                        self.cash += proceeds
                        self.inventory -= possible_qty
                        exec_sell_qty += possible_qty
                        trade_filled_sell += possible_qty
                        realized_this_step += possible_qty * (o.price - mid_t1)
                        filled = True
            # fallback por book cross
            if (not filled) and (bid_t1 >= o.price) and (self.inventory - o.qty >= -self.max_inventory):
                proceeds = o.qty * o.price
                self.cash += proceeds
                self.inventory -= o.qty
                exec_sell_qty += o.qty
                realized_this_step += o.qty * (o.price - mid_t1)
                filled = True

            if not filled:
                alive_asks.append(o)
        self.working_asks = alive_asks

        # 5) MTM e recompensa em t+1
        mtm_t1 = self._mark_to_market(bid_t1, ask_t1)
        pnl = mtm_t1 - self.starting_cash
        delta_mtm = mtm_t1 - self.prev_mtm
        damp_mtm = delta_mtm * (1 - self.dumped_factor) if delta_mtm > 0 else delta_mtm
        inv_penalty = self.inventory_factor * (self.inventory ** 2)
        final_realized_this_step = self.neg_realized_weight*realized_this_step if realized_this_step < 0 else self.pos_realized_weight * realized_this_step
        reward = self.mtm_weight*damp_mtm + final_realized_this_step - inv_penalty
            
        near_miss_bonus = 0.0
        filled = (exec_buy_qty + exec_sell_qty) > 0
        if not filled:
            # não executou, mas o mid moveu em direção à sua cotação: near miss bonus
            if action in (1, 2) and (mid_t1 > mid_t + self.tick_size): # ASK/BOTH
                near_miss_bonus = self.near_miss_weight
                reward += near_miss_bonus
            if action in (0, 2) and (mid_t1 < mid_t - self.tick_size): # BID/BOTH
                near_miss_bonus = self.near_miss_weight
                reward += near_miss_bonus
                
        reward += inv_reduction_reward
        
        inventory_no_quote_penalty = 0.0
        streak_penalty_value = 0.0
        if action in (0, 1, 2):
            self.no_quote_streak = 0
        else:
            self.no_quote_streak += 1
            if abs(self.inventory) > 0:
                inventory_no_quote_penalty = self.no_quote_inventory_penalty * inv_norm
                reward -= inventory_no_quote_penalty
            streak_penalty_value = self.no_quote_streak_penalty * self.no_quote_streak
            reward -= streak_penalty_value

        # 6) acumulados
        self.prev_mtm = mtm_t1

        # 7) avança tempo
        self.current_step = next_step
        truncated = self.current_step >= self.max_steps
        self.done = truncated
        obs = self._get_obs()
        
        # 8) logging / info: em treino, manter info mínimo para reduzir IPC
        self._last_normalized_reward = None

        logs_entry = {
            "step": self.current_step,
            "action": ["QUOTE_BID", "QUOTE_ASK", "QUOTE_BOTH", "NO_QUOTE"][action],
            "action_id": action,
            "bid_t": int(bid_t*1000)/1000, "ask_t": int(ask_t*1000)/1000,
            "bid_t1": int(bid_t1*1000)/1000, "ask_t1": int(ask_t1*1000)/1000,
            "model_bid": int(model_bid*1000)/1000, "model_ask": int(model_ask*1000)/1000,
            "n_working_bids": len(self.working_bids),
            "n_working_asks": len(self.working_asks),
            "trade_px_t1": int(trade_px*1000)/1000 if trade_px else trade_px,
            "trade_aggr_t1": "ASK" if trade_aggr == 3 else ("BID" if trade_aggr == 4 else trade_aggr),
            "trade_size_t1": trade_sz,
            "trade_filled_buy_qty": trade_filled_buy,
            "trade_filled_sell_qty": trade_filled_sell,
            "inventory": self.inventory,
            "cash": self.cash,
            "pnl": int(pnl*1000)/1000,
            "mtm_t1": mtm_t1,
            "delta_mtm": delta_mtm,
            "damp_mtm": self.mtm_weight*damp_mtm,
            "realized_this_step": int(final_realized_this_step*1000)/1000,
            "inv_penalty": inv_penalty,
            "inv_reduction_reward": int(inv_reduction_reward*1000)/1000,
            "near_miss_bonus": int(near_miss_bonus*1000)/1000,
            "no_quote_inventory_penalty": int(inventory_no_quote_penalty*1000)/1000,
            "no_quote_streak_penalty": int(streak_penalty_value*1000)/1000,
            "no_quote_streak": self.no_quote_streak,
            "normalized_reward": self._last_normalized_reward,
            "reward": int(reward*1000)/1000,
        }
        self.logs.append(logs_entry)

        terminated = False
        return obs, float(reward), terminated, truncated, {}

    def record_normalized_reward(self, normalized_reward: float):
        """Atualiza o último log com a recompensa normalizada fornecida por um wrapper externo."""
        if not self.logs:
            self._last_normalized_reward = None
            return
        try:
            value = float(normalized_reward)
            value = int(value * 1000) / 1000
        except (TypeError, ValueError):
            value = normalized_reward
        self.logs[-1]["normalized_reward"] = value
        self._last_normalized_reward = value

    def render(self):
        r = self._row(self.current_step)
        print(
            f"t={self.current_step} | bid={float(r['best_bid']):.2f} "
            f"|| ask={float(r['best_ask']):.2f} | inv={self.inventory} | cash={self.cash:.2f} "
            f"| nBids={len(self.working_bids)} nAsks={len(self.working_asks)}"
        )

    def close(self):
        if self.logs:
            output_path = Path(self.log_dir) / f"{self.log_prefix}{self.log_name}"
            pd.DataFrame(self.logs).to_csv(output_path, index=False)
            print(f"[TradeEngineEnv] Log salvo em {output_path}")

# if __name__ == "__main__":
#     DATA_DIR = os.path.join(os.path.dirname(__file__), "train_data")
#     data = pd.read_csv(os.path.join(DATA_DIR, "PETR4_05032025.csv"))
#     env = TradeEngineEnv(data, 
#         log_name="test_run", 
#         max_steps=len(data)-1,
#     )
#     obs, info = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         env.render()
#     env.close()

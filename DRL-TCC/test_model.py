"""
Rotina de avaliação do modelo PPO treinado para market making.

Para cada arquivo de dia presente na pasta de dados, o script:
  - Inicializa o ambiente `TradeEngineEnv` apenas com aquele dia;
  - Executa o modelo carregado de `MM_model_trained.zip`;
  - Coleta métricas de PnL e inventário;
  - Gera gráficos das séries de inventário, PnL e best bid/ask ao longo do dia.

Os resultados são salvos em `evaluation_results/` dentro deste diretório.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from trade_env import TradeEngineEnv

from aggregate_metrics import aggregate_metrics_main

MODEL_PATH = Path(__file__).resolve().parent / "MM_model_trained.zip"
DATA_DIR = Path(__file__).resolve().parent / "test_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "evaluation_results"

ACTION_LABELS = ["QUOTE_BID", "QUOTE_ASK", "QUOTE_BOTH", "NO_QUOTE"]

def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # garante backend não interativo
        import matplotlib.pyplot as plt

        return matplotlib, plt
    except ModuleNotFoundError:
        return None, None


MATPLOTLIB, PLT = _load_matplotlib()


def _action_to_int(action) -> int:
    if isinstance(action, np.ndarray):
        return int(action.flatten()[0])
    if isinstance(action, (list, tuple)):
        return int(action[0])
    if hasattr(action, "item"):
        return int(action.item())
    return int(action)


def ensure_output_dirs(base_dir: Path) -> Tuple[Path, Path, Path]:
    metrics_dir = base_dir / "metrics"
    plots_dir = base_dir / "plots"
    logs_dir = base_dir / "logs"
    for target in (base_dir, metrics_dir, plots_dir, logs_dir):
        target.mkdir(parents=True, exist_ok=True)
    return metrics_dir, plots_dir, logs_dir


def collect_metrics(log_df: pd.DataFrame) -> Dict[str, float]:
    """Calcula métricas básicas a partir do DataFrame de logs do ambiente."""
    if log_df.empty:
        return {
            "final_pnl": 0.0,
            "pnl_mean": 0.0,
            "pnl_std": 0.0,
            "inventory_mean": 0.0,
            "inventory_max": 0.0,
            "inventory_min": 0.0,
            "drawdown_pct": 0.0,
            **{f"prob_{label.lower()}_mean": 0.0 for label in ACTION_LABELS},
            **{f"action_count_{label.lower()}": 0 for label in ACTION_LABELS},
        }

    pnl_series = log_df["pnl"].astype(float)
    final_pnl = float(pnl_series.iloc[-1])
    pnl_mean = float(pnl_series.mean())
    pnl_std = float(pnl_series.std(ddof=0))

    inventory_mean = float(log_df["inventory"].mean())
    inventory_max = float(log_df["inventory"].max())
    inventory_min = float(log_df["inventory"].min())

    pnl_max = float(pnl_series.max())
    pnl_min = float(pnl_series.min())
    if pnl_max > 0:
        drawdown_pct = float((pnl_max - pnl_min) / pnl_max * 100.0)
    else:
        drawdown_pct = 0.0

    action_counts = log_df["action"].value_counts().to_dict()
    prob_means = {}
    for label in ACTION_LABELS:
        col = f"prob_{label.lower()}"
        key = f"{col}_mean"
        prob_means[key] = float(log_df[col].mean()) if col in log_df.columns else 0.0

    return {
        "final_pnl": final_pnl,
        "pnl_mean": pnl_mean,
        "pnl_std": pnl_std,
        "inventory_mean": inventory_mean,
        "inventory_max": inventory_max,
        "inventory_min": inventory_min,
        "drawdown_pct": drawdown_pct,
        **prob_means,
        **{
            f"action_count_{label.lower()}": int(action_counts.get(label, 0))
            for label in ACTION_LABELS
        },
    }


def plot_day(log_df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Gera gráficos de inventário, PnL, melhores preços e ações escolhidas."""
    if log_df.empty or PLT is None:
        return

    steps = log_df["step"]
    fig, axes = PLT.subplots(4, 1, figsize=(12, 13), sharex=True)

    axes[0].plot(steps, log_df["inventory"], color="tab:blue")
    axes[0].set_ylabel("Inventário")
    axes[0].set_title(f"Inventário - {title}")

    action_ax = axes[1]
    if "action" in log_df.columns:
        marker_by_action = {
            "QUOTE_BID": "^",
            "QUOTE_ASK": "v",
            "QUOTE_BOTH": "s",
            "NO_QUOTE": "x",
        }
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        available_actions = set(log_df["action"])
        for idx, label in enumerate(ACTION_LABELS):
            if label not in available_actions:
                continue
            mask = log_df["action"] == label
            action_steps = steps[mask]
            action_levels = [idx] * len(action_steps)
            action_ax.scatter(
                action_steps,
                action_levels,
                marker=marker_by_action.get(label, "o"),
                color=colors[idx % len(colors)],
                label=label,
                s=25,
                alpha=0.8,
            )
        action_ax.set_yticks(range(len(ACTION_LABELS)))
        action_ax.set_yticklabels(ACTION_LABELS)
        action_ax.set_ylabel("Ação")
        action_ax.set_xlabel("Passo")
        action_ax.set_title(f"Ações escolhidas - {title}")
        action_ax.legend(loc="upper right", ncol=2)
    else:
        action_ax.axis("off")

    axes[2].plot(steps, log_df["pnl"], color="tab:green")
    axes[2].set_ylabel("PnL")
    axes[2].set_title(f"PnL - {title}")

    axes[3].plot(steps, log_df["bid_t"], label="Best Bid", color="tab:purple")
    axes[3].plot(steps, log_df["ask_t"], label="Best Ask", color="tab:red")
    axes[3].set_ylabel("Preço")
    axes[3].set_xlabel("Passo")
    axes[3].set_title(f"Melhores preços - {title}")
    axes[3].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    PLT.close(fig)


def evaluate_single_day(
    model: PPO,
    csv_path: Path,
    logs_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    day_name = csv_path.stem
    df = pd.read_csv(csv_path)

    env = TradeEngineEnv(
        df,
        log_name=f"{day_name}.csv",
        max_steps=len(df)+1,
        log_dir=str(logs_dir),
        log_prefix="eval_",
    )

    obs, _ = env.reset()
    done = False
    while not done:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)

        action, _ = model.predict(obs, deterministic=True)
        action_int = _action_to_int(action)

        obs, _, terminated, truncated, _ = env.step(action_int)
        if env.logs:
            log_entry = env.logs[-1]
            for idx, label in enumerate(ACTION_LABELS):
                log_entry[f"prob_{label.lower()}"] = float(probs[idx])
        done = terminated or truncated

    log_df = pd.DataFrame(env.logs)

    env.close()
    metrics = collect_metrics(log_df)
    return log_df, metrics


def main() -> None:
    output_dir = OUTPUT_DIR / "deterministic"
    metrics_dir, plots_dir, logs_dir = ensure_output_dirs(output_dir)

    if PLT is None:
        print("Aviso: matplotlib não está disponível. Instale `matplotlib` para gerar gráficos.")
        return

    model = PPO.load(MODEL_PATH, device="cpu")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {DATA_DIR}")

    aggregated_metrics: List[Dict[str, float]] = []

    print(f"Avaliando {len(csv_files)} arquivos usando {MODEL_PATH.name}...")
    print("Modo: Determinístico")
    for csv_path in csv_files:
        day_name = csv_path.stem
        print(f"- Processando {day_name}...")

        log_df, metrics = evaluate_single_day(
            model=model,
            csv_path=csv_path,
            logs_dir=logs_dir,
        )

        metric_path = metrics_dir / f"{day_name}_metrics.json"
        metric_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

        plot_path = plots_dir / f"{day_name}.png"
        plot_day(log_df, day_name, plot_path)

        daily_metrics = {"day": day_name, **metrics}
        aggregated_metrics.append(daily_metrics)

    aggregated_df = pd.DataFrame(aggregated_metrics)
    aggregated_csv = output_dir / "daily_metrics.csv"
    aggregated_df.to_csv(aggregated_csv, index=False)

    print("Avaliação concluída.")
    print(f"- Métricas agregadas: {aggregated_csv}")
    print(f"- Métricas individuais (JSON): {metrics_dir}")
    print(f"- Gráficos: {plots_dir}")
    print(f"- Logs do ambiente: {logs_dir}")


if __name__ == "__main__":
    main()
    aggregate_metrics_main()

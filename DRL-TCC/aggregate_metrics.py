"""
Agrega os resultados diários produzidos por `test_model.py`.

Gera dois arquivos:
  - `aggregated_by_ticker.csv`: métricas consolidadas por ticker.
  - `aggregated_by_liquidity.csv`: métricas consolidadas por categoria de liquidez.

As categorias de liquidez seguem a convenção:
  * IDs 1-5   -> "liq_01_05"
  * IDs 6-10  -> "liq_06_10"
  * IDs 11-15 -> "liq_11_15"
  * Qualquer outro ID -> "liq_outros"
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

DEFAULT_INPUT = Path(__file__).resolve().parent / "evaluation_results"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent


def extract_day_components(day_str: str) -> Tuple[int, str, str]:
    """
    Converte um nome de dia no formato `ID_TICKER_YYYYMMDD` nos elementos
    (ID numérico, ticker, data). Se o formato estiver incorreto, lança ValueError.
    """
    parts = day_str.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Formato inesperado para 'day': {day_str!r}")
    try:
        rank = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"ID numérico inválido em {day_str!r}") from exc
    ticker = parts[1]
    date_str = parts[2]
    return rank, ticker, date_str


def liquidity_category(rank: int) -> str:
    if 1 <= rank <= 5:
        return "liq_01_05"
    if 6 <= rank <= 10:
        return "liq_06_10"
    if 11 <= rank <= 15:
        return "liq_11_15"
    return "liq_outros"


def add_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    ranks: Iterable[int] = []
    tickers: Iterable[str] = []
    dates: Iterable[str] = []
    categories: Iterable[str] = []

    for day_str in df["day"]:
        rank, ticker, date_str = extract_day_components(day_str)
        ranks.append(rank)
        tickers.append(ticker)
        dates.append(date_str)
        categories.append(liquidity_category(rank))

    df = df.copy()
    df["ticker_rank"] = ranks
    df["ticker"] = tickers
    df["trade_date"] = dates
    df["liquidity_category"] = categories
    return df


def aggregate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    action_cols = [c for c in df.columns if c.startswith("action_count_")]
    prob_cols = [c for c in df.columns if c.startswith("prob_")]

    rows = []
    for group_value, gdf in df.groupby(group_col):
        row: Dict[str, float | int | str] = {
            group_col: group_value,
            "n_days": int(len(gdf)),
            "final_pnl_sum": float(gdf["final_pnl"].sum()),
            "final_pnl_mean": float(gdf["final_pnl"].mean()),
            "final_pnl_std": float(gdf["final_pnl"].std(ddof=0)),
        }

        row["pnl_mean_avg"] = float(gdf["pnl_mean"].mean())
        row["pnl_std_avg"] = float(gdf["pnl_std"].mean())
        row["inventory_mean_avg"] = float(gdf["inventory_mean"].mean())
        row["inventory_max_avg"] = float(gdf["inventory_max"].mean())
        row["inventory_min_avg"] = float(gdf["inventory_min"].mean())
        if "drawdown_pct" in gdf.columns:
            row["drawdown_pct_avg"] = float(gdf["drawdown_pct"].mean())

        for col in action_cols:
            row[f"{col}_sum"] = int(gdf[col].sum())

        for col in prob_cols:
            row[f"{col}_avg"] = float(gdf[col].mean())
            row[f"{col}_std"] = float(gdf[col].std(ddof=0))

        rows.append(row)

    result = pd.DataFrame(rows)
    if group_col == "liquidity_category":
        category_order = [
            "liq_01_05",
            "liq_06_10",
            "liq_11_15",
            "liq_outros",
        ]
        result["__order"] = result[group_col].map({c: i for i, c in enumerate(category_order)})
        result = result.sort_values(["__order", group_col]).drop(columns="__order")
    else:
        result = result.sort_values([group_col])
    result.reset_index(drop=True, inplace=True)
    return result


def aggregate_metrics_main() -> None:
    input_csv = DEFAULT_INPUT / "deterministic" / "daily_metrics.csv"
    output_dir = DEFAULT_OUTPUT_DIR / "evaluation_results" / "deterministic"

    if not input_csv.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"O arquivo {input_csv} está vazio.")

    df = add_metadata_columns(df)

    by_ticker = aggregate(df, "ticker")
    by_category = aggregate(df, "liquidity_category")

    ticker_path = output_dir / "aggregated_by_ticker.csv"
    category_path = output_dir / "aggregated_by_liquidity.csv"

    by_ticker.to_csv(ticker_path, index=False)
    by_category.to_csv(category_path, index=False)

    print("Agregação concluída.")
    print(f"- Resultado por ticker: {ticker_path}")
    print(f"- Resultado por categoria de liquidez: {category_path}")


if __name__ == "__main__":
    aggregate_metrics_main()

"""
Aplicação Streamlit para demonstrar o modelo de market making do TCC.

Permite selecionar dias específicos para avaliação, executa o modelo PPO
treinado e exibe métricas diárias, gráficos e agregados.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from stable_baselines3 import PPO

from aggregate_metrics import add_metadata_columns, aggregate, extract_day_components
from test_model import (
    DATA_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    ensure_output_dirs,
    evaluate_single_day,
    plot_day,
)


@dataclass
class DayResult:
    day: str
    metrics: Dict[str, float]
    log_df: pd.DataFrame
    plot_path: Optional[Path]


st.set_page_config(
    page_title="Demonstração Market Making",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Demonstração do TCC — Agente de Market Making")
st.caption(
    "Execute o modelo PPO treinado em dias selecionados, visualize métricas e resultados agregados."
)


if not MODEL_PATH.exists():
    st.error(f"Modelo não encontrado em `{MODEL_PATH}`. Treine ou copie o arquivo antes de prosseguir.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path) -> PPO:
    """Carrega o modelo PPO uma única vez por sessão."""
    return PPO.load(model_path, device="cpu")


@st.cache_data(show_spinner=False)
def list_available_days() -> List[str]:
    """Retorna todos os dias disponíveis na pasta de teste."""
    return sorted(csv_path.stem for csv_path in DATA_DIR.glob("*.csv"))


@st.cache_data(show_spinner=False)
def days_grouped_by_ticker(days: List[str]) -> Dict[str, List[str]]:
    """Agrupa os dias disponíveis por ticker."""
    mapping: Dict[str, List[str]] = {}
    for day in days:
        _, ticker, _ = extract_day_components(day)
        mapping.setdefault(ticker, []).append(day)
    return {ticker: sorted(entries) for ticker, entries in mapping.items()}


def clear_day_selection() -> None:
    """Callback para limpar seleção de tickers/dias."""
    st.session_state["selected_tickers"] = []
    st.session_state["selected_days"] = []


def run_evaluation(model: PPO, selected_days: List[str]) -> Dict[str, object]:
    """Executa o modelo para os dias selecionados e consolida resultados."""
    suffix = "deterministic"
    base_dir = OUTPUT_DIR / "webapp" / suffix
    metrics_dir, plots_dir, logs_dir = ensure_output_dirs(base_dir)

    day_details: List[DayResult] = []
    aggregated_rows: List[Dict[str, float]] = []

    for day in selected_days:
        csv_path = DATA_DIR / f"{day}.csv"
        if not csv_path.exists():
            st.warning(f"Arquivo `{csv_path.name}` não encontrado; ignorando.")
            continue

        log_df, metrics = evaluate_single_day(
            model=model,
            csv_path=csv_path,
            logs_dir=logs_dir,
        )

        plot_path = plots_dir / f"{day}.png"
        plot_day(log_df, day, plot_path)

        day_details.append(
            DayResult(
                day=day,
                metrics=metrics,
                log_df=log_df,
                plot_path=plot_path if plot_path.exists() else None,
            )
        )
        aggregated_rows.append({"day": day, **metrics})

    daily_df = pd.DataFrame(aggregated_rows)
    if not daily_df.empty:
        enriched_df = add_metadata_columns(daily_df)
        aggregated_by_ticker = aggregate(enriched_df, "ticker")
        aggregated_by_liquidity = aggregate(enriched_df, "liquidity_category")
    else:
        aggregated_by_ticker = pd.DataFrame()
        aggregated_by_liquidity = pd.DataFrame()

    return {
        "suffix": suffix,
        "base_dir": base_dir,
        "deterministic": True,
        "day_details": day_details,
        "daily_df": daily_df,
        "aggregated_by_ticker": aggregated_by_ticker,
        "aggregated_by_liquidity": aggregated_by_liquidity,
    }


model = load_model(MODEL_PATH)
available_days = list_available_days()
grouped_days = days_grouped_by_ticker(available_days)

with st.sidebar:
    st.header("Configurações")
    default_selection = available_days[:5]

    if "selected_tickers" not in st.session_state:
        st.session_state["selected_tickers"] = []
    if "selected_days" not in st.session_state:
        st.session_state["selected_days"] = default_selection
    st.multiselect(
        "Selecione tickers (adiciona todos os dias automaticamente)",
        options=sorted(grouped_days.keys()),
        key="selected_tickers",
        help="Escolha um ou mais tickers para incluir todos os respectivos dias.",
    )

    prev_tickers = st.session_state.get("_prev_selected_tickers", [])
    current_tickers = st.session_state["selected_tickers"]

    newly_added = sorted(set(current_tickers) - set(prev_tickers))
    removed = sorted(set(prev_tickers) - set(current_tickers))

    if newly_added:
        auto_days = {
            day
            for ticker in newly_added
            for day in grouped_days.get(ticker, [])
        }
        if auto_days:
            st.session_state["selected_days"] = sorted(
                set(st.session_state["selected_days"]).union(auto_days)
            )

    if removed:
        removed_days = {
            day
            for ticker in removed
            for day in grouped_days.get(ticker, [])
        }
        if removed_days:
            st.session_state["selected_days"] = [
                day
                for day in st.session_state["selected_days"]
                if day not in removed_days
            ]

    st.session_state["_prev_selected_tickers"] = current_tickers.copy()

    st.multiselect(
        "Dias para avaliar",
        options=available_days,
        default=st.session_state["selected_days"],
        key="selected_days",
        help="Ajuste manualmente a seleção de dias se necessário.",
    )

    st.button(
        "Limpar seleção de dias",
        use_container_width=True,
        on_click=clear_day_selection,
    )

    st.markdown("---")
    execute = st.button("Executar avaliação", type="primary", use_container_width=True)


if execute:
    if not st.session_state["selected_days"]:
        st.warning("Selecione ao menos um dia antes de executar a avaliação.")
    else:
        with st.spinner("Executando avaliação do modelo..."):
            result = run_evaluation(
                model,
                st.session_state["selected_days"],
            )
            result["label"] = "Determinístico"
            st.session_state["last_run"] = result
        st.success("Avaliação concluída! Verifique os resultados abaixo.")


last_run = st.session_state.get("last_run")
if not last_run:
    st.info("Configure os parâmetros na barra lateral e clique em **Executar avaliação**.")
    st.stop()


def render_metrics_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """Formata o dicionário de métricas para exibição."""
    df = pd.DataFrame(metrics, index=[0]).T.reset_index()
    df.columns = ["Métrica", "Valor"]
    return df


st.subheader(f"Métricas por dia — {last_run['label']}")
for day_result in last_run["day_details"]:
    with st.expander(day_result.day, expanded=False):
        metrics_df = render_metrics_table(day_result.metrics)
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        if day_result.plot_path:
            st.image(str(day_result.plot_path), caption=f"Gráficos gerados para {day_result.day}")
        elif day_result.log_df.empty:
            st.warning("Não há dados de log para gerar gráficos.")
        else:
            st.line_chart(
                day_result.log_df.set_index("step")[["pnl", "inventory"]],
                height=240,
            )

        st.caption(f"Logs salvos em `{last_run['base_dir'] / 'logs'}`.")


if last_run["daily_df"].empty:
    st.warning(
        "Nenhuma métrica foi gerada. Verifique se os arquivos selecionados são válidos."
    )
else:
    st.subheader(f"Métricas agregadas — {last_run['label']}")
    tab_daily, tab_ticker, tab_liquidity = st.tabs(
        [
            f"Por dia ({last_run['label']})",
            f"Por ticker ({last_run['label']})",
            f"Por categoria ({last_run['label']})",
        ]
    )

    with tab_daily:
        st.dataframe(last_run["daily_df"], use_container_width=True, hide_index=True)
        st.download_button(
            "Baixar métricas por dia (CSV)",
            data=last_run["daily_df"].to_csv(index=False).encode("utf-8"),
            file_name=f"daily_metrics_{last_run['suffix']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_ticker:
        if last_run["aggregated_by_ticker"].empty:
            st.info("Execute a avaliação para visualizar agregados por ticker.")
        else:
            st.dataframe(last_run["aggregated_by_ticker"], use_container_width=True, hide_index=True)
            st.download_button(
                "Baixar agregados por ticker (CSV)",
                data=last_run["aggregated_by_ticker"].to_csv(index=False).encode("utf-8"),
                file_name=f"aggregated_by_ticker_{last_run['suffix']}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_liquidity:
        if last_run["aggregated_by_liquidity"].empty:
            st.info("Execute a avaliação para visualizar agregados por categoria de liquidez.")
        else:
            st.dataframe(last_run["aggregated_by_liquidity"], use_container_width=True, hide_index=True)
            st.download_button(
                "Baixar agregados por liquidez (CSV)",
                data=last_run["aggregated_by_liquidity"].to_csv(index=False).encode("utf-8"),
                file_name=f"aggregated_by_liquidity_{last_run['suffix']}.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.caption(
    "Os arquivos gerados ficam disponíveis em `evaluation_results/webapp/` para consulta posterior."
)

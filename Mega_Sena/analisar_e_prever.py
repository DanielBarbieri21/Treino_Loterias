from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lottery_engine import LotteryConfig, run_full_analysis


CONFIG = LotteryConfig(
    name="Mega-Sena",
    slug="mega_sena",
    file_path=str(Path(__file__).resolve().parent / "mega_sena.xlsx"),
    number_min=1,
    number_max=60,
    pick_count=6,
)


st.set_page_config(page_title="Mega-Sena Profissional", layout="wide")
st.title("Mega-Sena | Motor Profissional de Ranking")
st.caption("O sistema só recomenda jogos depois de medir várias estratégias em backtest temporal.")

validation_size = st.sidebar.slider("Janela de validação", 60, 240, 140, 10)
test_size = st.sidebar.slider("Janela de teste", 20, 120, 60, 10)
total_games = st.sidebar.slider("Quantidade de jogos", 5, 20, 10, 1)

if st.button("Executar análise", type="primary"):
    result = run_full_analysis(CONFIG, validation_size=validation_size, test_size=test_size, total_games=total_games)
    evaluation = result["evaluation"]
    test_result = evaluation["test_result"]
    ranking = result["ranking"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Sorteios", len(result["draws"]))
    col2.metric("Estratégia", evaluation["selected_strategy"]["name"])
    col3.metric("Média no teste", f"{test_result['mean_hits']:.2f}")

    st.subheader("Jogo Principal")
    st.markdown(f"### {', '.join(f'{n:02d}' for n in result['main_game'])}")

    st.subheader("Backtest")
    st.write(
        {
            "aleatorio_esperado": round(CONFIG.expected_random_hits, 2),
            "media_teste": round(test_result["mean_hits"], 2),
            "mediana_teste": round(test_result["median_hits"], 2),
            "maximo_teste": test_result["max_hits"],
            "minimo_teste": test_result["min_hits"],
        }
    )

    st.subheader("Estratégias avaliadas")
    strategies_df = pd.DataFrame(evaluation["validation_results"])[["strategy", "mean_hits", "median_hits", "max_hits", "std_hits"]]
    st.dataframe(strategies_df, use_container_width=True, hide_index=True)

    st.subheader("Top combinações")
    candidates_df = pd.DataFrame(result["candidates"])
    st.dataframe(candidates_df, use_container_width=True, hide_index=True)

    st.subheader("Ranking dos números")
    st.dataframe(
        ranking[["number", "score", "freq_10", "freq_20", "freq_50", "decay", "repeat_rate", "cooc_last", "gap"]].head(20),
        use_container_width=True,
        hide_index=True,
    )

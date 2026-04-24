from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lottery_engine import LotteryConfig, history_to_binary, load_history, pattern_snapshot, run_full_analysis


CONFIG = LotteryConfig(
    name="Lotofácil",
    slug="lotofacil",
    file_path=str(Path(__file__).resolve().parent / "treino.xlsx"),
    number_min=1,
    number_max=25,
    pick_count=15,
)


def carregar_dados():
    df = load_history(CONFIG)
    return df, history_to_binary(df, CONFIG)


def analisar_padroes_temporais(df_num: pd.DataFrame):
    return pattern_snapshot(df_num, CONFIG)["temporais"]


def analisar_padroes_sequenciais(df_num: pd.DataFrame):
    return pattern_snapshot(df_num, CONFIG)["sequenciais"]


def analisar_padroes_grupos(df_num: pd.DataFrame):
    return pattern_snapshot(df_num, CONFIG)["grupos"]


def analisar_padroes_repeticao(df_num: pd.DataFrame):
    return pattern_snapshot(df_num, CONFIG)["repeticao"]


def criar_features_avancadas(df_num: pd.DataFrame, *_args, **_kwargs):
    return pattern_snapshot(df_num, CONFIG)["metrics"]


def gerar_15_numeros_inteligentes(*_args, n_candidatos: int = 10, **_kwargs):
    result = run_full_analysis(CONFIG, total_games=n_candidatos)
    top_game = result["main_game"]
    ranking = result["ranking"]
    candidates = [(item["numbers"], item["score"]) for item in result["candidates"]]
    probabilities = ranking.sort_values("number")["score"].to_numpy()
    return top_game, probabilities, candidates

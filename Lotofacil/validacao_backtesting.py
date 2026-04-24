from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lottery_engine import LotteryConfig, evaluate_strategies, load_history, draws_from_frame


CONFIG = LotteryConfig(
    name="Lotofácil",
    slug="lotofacil",
    file_path=str(Path(__file__).resolve().parent / "treino.xlsx"),
    number_min=1,
    number_max=25,
    pick_count=15,
)


def backtest_modelo(_df_numeros=None, _criar_features_func=None, _gerar_numeros_func=None, n_testes: int = 60, janela_treino: int = 120):
    df = load_history(CONFIG)
    draws = draws_from_frame(df)
    evaluation = evaluate_strategies(draws, CONFIG, validation_size=janela_treino, test_size=n_testes)
    resultado = evaluation["test_result"]
    resultado["estatisticas"] = {
        "total_testes": len(resultado["hits"]),
        "media_acertos": resultado["mean_hits"],
        "mediana_acertos": resultado["median_hits"],
        "max_acertos": resultado["max_hits"],
        "min_acertos": resultado["min_hits"],
        "std_acertos": resultado["std_hits"],
        "acerto_aleatorio_esperado": CONFIG.expected_random_hits,
        "melhoria_sobre_aleatorio": ((resultado["mean_hits"] - CONFIG.expected_random_hits) / CONFIG.expected_random_hits) * 100,
        "estrategia": evaluation["selected_strategy"]["name"],
    }
    return resultado


def visualizar_resultados_backtesting(resultados, salvar: bool = True):
    stats = resultados["estatisticas"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(resultados["hits"], marker="o", linewidth=1)
    ax.axhline(stats["media_acertos"], color="tab:red", linestyle="--", label=f"Média {stats['media_acertos']:.2f}")
    ax.axhline(stats["acerto_aleatorio_esperado"], color="tab:green", linestyle="--", label=f"Aleatório {stats['acerto_aleatorio_esperado']:.2f}")
    ax.set_title(f"Lotofácil | Estratégia {stats['estrategia']}")
    ax.set_xlabel("Sorteio avaliado")
    ax.set_ylabel("Acertos")
    ax.legend()
    ax.grid(alpha=0.3)
    if salvar:
        plt.savefig("backtesting_resultados_lotofacil.png", dpi=200, bbox_inches="tight")
    return fig


def gerar_relatorio_backtesting(resultados, salvar_txt: bool = True):
    stats = resultados["estatisticas"]
    relatorio = (
        f"Lotofácil\n"
        f"Estratégia: {stats['estrategia']}\n"
        f"Testes: {stats['total_testes']}\n"
        f"Média de acertos: {stats['media_acertos']:.2f}\n"
        f"Mediana: {stats['mediana_acertos']:.2f}\n"
        f"Máximo: {stats['max_acertos']}\n"
        f"Mínimo: {stats['min_acertos']}\n"
        f"Desvio padrão: {stats['std_acertos']:.2f}\n"
        f"Aleatório esperado: {stats['acerto_aleatorio_esperado']:.2f}\n"
        f"Melhoria sobre aleatório: {stats['melhoria_sobre_aleatorio']:.2f}%\n"
        f"Distribuição: {np.bincount(resultados['hits']).tolist()}\n"
    )
    if salvar_txt:
        Path("relatorio_backtesting_lotofacil.txt").write_text(relatorio, encoding="utf-8")
    return relatorio

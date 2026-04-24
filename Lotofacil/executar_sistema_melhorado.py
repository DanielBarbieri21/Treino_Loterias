from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lottery_engine import LotteryConfig, run_full_analysis, save_text_report


CONFIG = LotteryConfig(
    name="Lotofácil",
    slug="lotofacil",
    file_path=str(Path(__file__).resolve().parent / "treino.xlsx"),
    number_min=1,
    number_max=25,
    pick_count=15,
)


def main():
    result = run_full_analysis(CONFIG, validation_size=140, test_size=60, total_games=10)
    evaluation = result["evaluation"]
    test_result = evaluation["test_result"]

    print("=" * 72)
    print("LOTOfácil | Motor Profissional de Ranking")
    print("=" * 72)
    print(f"Sorteios carregados: {len(result['draws'])}")
    print(f"Estratégia selecionada: {evaluation['selected_strategy']['name']}")
    print(f"Backtest teste: média {test_result['mean_hits']:.2f} | mediana {test_result['median_hits']:.2f} | máximo {test_result['max_hits']}")
    print(f"Aleatório esperado: {CONFIG.expected_random_hits:.2f}")
    print()
    print(f"Jogo principal: {result['main_game']}")
    print()
    print("Top 10 combinações:")
    for index, candidate in enumerate(result["candidates"], start=1):
        print(f"{index:02d}. {candidate['numbers']} | score={candidate['score']:.4f}")
    print()
    report_path = Path(__file__).resolve().parent / "relatorio_profissional_lotofacil.txt"
    save_text_report(result, report_path)
    print(f"Relatório salvo em: {report_path.name}")


if __name__ == "__main__":
    main()

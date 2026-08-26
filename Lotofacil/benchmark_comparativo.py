"""
Benchmark e Backtesting Comparativo de Metodos - Lotofacil
===========================================================
Compara de forma rigorosa e transparente 5 abordagens nos ultimos 50 sorteios:
  0. Baseline Aleatorio (Controle)
  1. Ensemble BiLSTM + XGBoost (Padrao)
  2. Metodo 1: Filtros Estruturais de 7 Camadas + Eliminacao Reversa
  3. Metodo 2: Fechamento / Desdobramento Matematico Inteligente com IA
  4. Metodo 3: Algoritmo Genetico de Portfolio (Diversidade + Cobertura)

Coleta para cada metodo:
  - Media de acertos
  - Mediana de acertos
  - Maximo de acertos em um unico concurso
  - Total de bilhetes premiados: 11, 12, 13, 14 e 15 pontos
  - Percentual de sorteios com pelo menos 1 bilhete premiado (>= 11 pontos)
"""

from __future__ import annotations

import os, sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Imports do projeto
sys.path.insert(0, str(Path(__file__).parent))

from buscar_dados_api_v2 import carregar_binario, atualizar_excel, ARQUIVO_PADRAO
from tensorflow_model import (
    treinar as treinar_lstm,
    prever_proximas_probs as lstm_probs,
    WINDOW,
)
from treino_tensorflow import (
    treinar_xgboost,
    criar_features_sklearn,
    ensemble_probs,
    gerar_melhores_jogos as gerar_jogos_ensemble_padrao,
)
from filtros_estruturais import gerar_jogos_com_filtros_e_eliminacao
from fechamento_matematico import gerar_fechamento_inteligente
from algoritmo_genetico import otimizar_portfolio_genetico

# ──────────────────────────────────────────────────────────────
# Funcao de Avaliacao de Resultados
# ──────────────────────────────────────────────────────────────

def contar_premios(hits_lista: List[int]) -> Dict[str, int]:
    """Conta quantidade de bilhetes com 11, 12, 13, 14 e 15 acertos."""
    return {
        "p11": sum(1 for h in hits_lista if h == 11),
        "p12": sum(1 for h in hits_lista if h == 12),
        "p13": sum(1 for h in hits_lista if h == 13),
        "p14": sum(1 for h in hits_lista if h == 14),
        "p15": sum(1 for h in hits_lista if h == 15),
        "total_premiados": sum(1 for h in hits_lista if h >= 11),
    }


# ──────────────────────────────────────────────────────────────
# Pipeline de Benchmark
# ──────────────────────────────────────────────────────────────

def executar_benchmark(
    n_sorteios_teste: int = 40,
    n_jogos_por_concurso: int = 10,
    seed: int = 42,
) -> Dict:
    print("\n" + "=" * 70)
    print(f"  🏁 BENCHMARK COMPARATIVO DE MÉTODOS LOTOFÁCIL ({n_sorteios_teste} SORTEIOS)")
    print("=" * 70 + "\n")

    # 1. Carregar dados atualizados
    print("📡 [1/4] Carregando e sincronizando base com API da Caixa...")
    binario, df_raw = carregar_binario(verbose=False)
    N_total = len(binario)
    print(f"       Total de concursos na base: {N_total}\n")

    # 2. Treinar Modelos Base (LSTM e XGBoost)
    print("🧠 [2/4] Treinando modelos de Inteligência Artificial...")
    split_idx = N_total - n_sorteios_teste
    binario_treino = binario[:split_idx]

    print("       -> Treinando BiLSTM (TensorFlow)...")
    lstm_model, _ = treinar_lstm(
        binario_treino,
        window=WINDOW,
        test_size=30,
        epochs=50,
        batch_size=32,
        verbose=0,
    )

    print("       -> Extraindo features tabulares vetorizadas...")
    X_features_full = criar_features_sklearn(binario, window=WINDOW)

    print("       -> Treinando XGBoost...")
    xgb_model, _, _ = treinar_xgboost(binario_treino, window=WINDOW, test_size=30)
    print("       Modelos treinados com sucesso.\n")

    # 3. Execucao do Backtesting Comparativo nos ultimos n_sorteios_teste
    print(f"📊 [3/4] Executando Backtesting em {n_sorteios_teste} sorteios históricos...\n")

    nomes_metodos = [
        "0. Baseline Aleatorio",
        "1. Ensemble IA Padrao",
        "2. Filtros 7 Camadas + Eliminacao",
        "3. Fechamento Matematico IA",
        "4. Algoritmo Genetico (Portfolio)",
    ]

    historico_hits: Dict[str, List[int]] = {m: [] for m in nomes_metodos}
    max_hits_por_sorteio: Dict[str, List[int]] = {m: [] for m in nomes_metodos}
    sorteios_com_premio: Dict[str, int] = {m: 0 for m in nomes_metodos}

    rng = np.random.default_rng(seed)

    for step, i in enumerate(range(split_idx, N_total), start=1):
        # Dados ate o concurso i (sem vazamento do sorteio i)
        historico_passado = binario[:i]
        real_sorteado = set(np.where(binario[i] == 1)[0] + 1)
        ultimo_sorteio_conhecido = sorted(list(np.where(binario[i-1] == 1)[0] + 1))

        # Obter probabilidades do ensemble IA para o concurso i
        p_lstm = lstm_probs(lstm_model, historico_passado, window=WINDOW)
        idx_feature = i - WINDOW - 1
        X_sklearn = X_features_full[idx_feature : idx_feature + 1]
        probs_ia = ensemble_probs(p_lstm, xgb_model, X_sklearn, peso_lstm=9.3, peso_xgb=9.0)

        # ── Metodo 0: Aleatorio ──
        jogos_m0 = [
            sorted(list(rng.choice(25, size=15, replace=False) + 1))
            for _ in range(n_jogos_por_concurso)
        ]

        # ── Metodo 1: Ensemble Padrao ──
        res_m1 = gerar_jogos_ensemble_padrao(
            probs_ia, historico_passado, n_candidatos=800, n_jogos=n_jogos_por_concurso, seed=seed+i
        )
        jogos_m1 = [j["numeros"] for j in res_m1]

        # ── Metodo 2: Filtros 7 Camadas + Eliminacao Reversa ──
        res_m2 = gerar_jogos_com_filtros_e_eliminacao(
            probs=probs_ia,
            ultimo_sorteio=ultimo_sorteio_conhecido,
            n_eliminar=5,
            n_jogos=n_jogos_por_concurso,
            n_tentativas=1200,
            seed=seed+i
        )
        jogos_m2 = [j["numeros"] for j in res_m2]

        # ── Metodo 3: Fechamento Matematico IA ──
        res_m3 = gerar_fechamento_inteligente(
            probs=probs_ia,
            n_jogos=n_jogos_por_concurso,
            tamanho_pool=19,
            n_fixas=0,
            seed=seed+i
        )
        jogos_m3 = [j["numeros"] for j in res_m3]

        # ── Metodo 4: Algoritmo Genetico ──
        res_m4 = otimizar_portfolio_genetico(
            probs=probs_ia,
            ultimo_sorteio=ultimo_sorteio_conhecido,
            n_jogos=n_jogos_por_concurso,
            pop_size=16,
            n_geracoes=8,
            seed=seed+i,
            verbose=False
        )
        jogos_m4 = [j["numeros"] for j in res_m4]

        todos_jogos = [jogos_m0, jogos_m1, jogos_m2, jogos_m3, jogos_m4]

        # Avaliar acertos de cada metodo contra o sorteio real
        for nome_m, jogos in zip(nomes_metodos, todos_jogos):
            hits_jogos = [len(set(j) & real_sorteado) for j in jogos]
            historico_hits[nome_m].extend(hits_jogos)
            
            max_h = max(hits_jogos)
            max_hits_por_sorteio[nome_m].append(max_h)
            if max_h >= 11:
                sorteios_com_premio[nome_m] += 1

        if step % 10 == 0 or step == n_sorteios_teste:
            print(f"  [{step:02d}/{n_sorteios_teste}] Sorteios avaliados...")

    # 4. Consolidar Resultados
    print("\n" + "=" * 70)
    print("  🏆 TABELA COMPARATIVA DE RESULTADOS")
    print("=" * 70 + "\n")

    tabela_linhas = []
    for nome_m in nomes_metodos:
        all_hits = np.array(historico_hits[nome_m])
        max_hits = np.array(max_hits_por_sorteio[nome_m])
        premios = contar_premios(historico_hits[nome_m])
        
        taxa_sorteios_premiados = (sorteios_com_premio[nome_m] / n_sorteios_teste) * 100

        tabela_linhas.append({
            "Metodo": nome_m,
            "Media Acertos": float(all_hits.mean()),
            "Mediana": float(np.median(all_hits)),
            "Melhor Jogo (Max)": int(all_hits.max()),
            "Media Melhores/Jogo": float(max_hits.mean()),
            "Bilhetes 11 pts": premios["p11"],
            "Bilhetes 12 pts": premios["p12"],
            "Bilhetes 13 pts": premios["p13"],
            "Bilhetes 14 pts": premios["p14"],
            "Bilhetes 15 pts": premios["p15"],
            "Total Premios (>=11)": premios["total_premiados"],
            "% Concursos Premiados": taxa_sorteios_premiados,
        })

    df_resultado = pd.DataFrame(tabela_linhas)
    # Ordenar pelo maior numero de bilhetes premiados e media
    df_resultado.sort_values(by=["Total Premios (>=11)", "Media Acertos"], ascending=False, inplace=True)
    df_resultado.reset_index(drop=True, inplace=True)

    print(df_resultado.to_string(index=False))

    # 5. Gerar Recomendacao para o PROXIMO Concurso com o Metodo Campeao
    melhor_metodo_nome = df_resultado.iloc[0]["Metodo"]
    print(f"\n🥇 MÉTODO CAMPEÃO: {melhor_metodo_nome}")

    # Retreinar com 100% dos dados para gerar o proximo concurso
    p_lstm_final = lstm_probs(lstm_model, binario, window=WINDOW)
    X_last_full = criar_features_sklearn(binario, window=WINDOW)[-1:]
    probs_ia_final = ensemble_probs(p_lstm_final, xgb_model, X_last_full, peso_lstm=9.3, peso_xgb=9.0)
    ultimo_sorteio_real = sorted(list(np.where(binario[-1] == 1)[0] + 1))
    concurso_atual = int(df_raw["concurso"].max())

    if "Fechamento" in melhor_metodo_nome:
        jogos_finais = gerar_fechamento_inteligente(probs_ia_final, n_jogos=10, tamanho_pool=19)
    elif "Filtros" in melhor_metodo_nome:
        jogos_finais = gerar_jogos_com_filtros_e_eliminacao(probs_ia_final, ultimo_sorteio_real, n_jogos=10)
    elif "Genetico" in melhor_metodo_nome:
        jogos_finais = otimizar_portfolio_genetico(probs_ia_final, ultimo_sorteio_real, n_jogos=10, pop_size=30, n_geracoes=20)
    else:
        jogos_finais = gerar_jogos_ensemble_padrao(probs_ia_final, binario, n_jogos=10)

    print(f"\n🎯 JOGOS RECOMENDADOS PARA O PRÓXIMO CONCURSO ({concurso_atual + 1}):")
    for idx, j in enumerate(jogos_finais, 1):
        num_str = ", ".join(f"{n:02d}" for n in j["numeros"])
        print(f"  Jogo {idx:02d}: [{num_str}]  (Score: {j['score']:.4f})")

    # Salvar Relatório em TXT
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_relatorio = Path(__file__).parent / f"benchmark_comparativo_{ts}.txt"
    
    linhas_relatorio = [
        "=" * 70,
        "  RELATORIO DE BENCHMARK COMPARATIVO - LOTOFACIL",
        f"  Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        f"  Sorteios Avaliados no Teste: {n_sorteios_teste}",
        f"  Ultimo Concurso na Base: {concurso_atual}",
        "=" * 70,
        "",
        "TABELA RESUMO DE PERFORMANCE:",
        df_resultado.to_string(index=False),
        "",
        "=" * 70,
        f"JOGOS RECOMENDADOS PELO METODO CAMPEAO ({melhor_metodo_nome}):",
        "=" * 70,
    ]
    for idx, j in enumerate(jogos_finais, 1):
        num_str = ", ".join(f"{n:02d}" for n in j["numeros"])
        linhas_relatorio.append(f"  Jogo {idx:02d}: [{num_str}]  (Score: {j['score']:.4f})")

    linhas_relatorio += [
        "",
        "----------------------------------------------------------------------",
        "Aviso: Loterias envolvem probabilidade e aleatoriedade.",
        "Utilize este sistema para otimizacao matematica das suas apostas.",
        "=" * 70,
    ]

    arquivo_relatorio.write_text("\n".join(linhas_relatorio), encoding="utf-8")
    print(f"\n📄 Relatório detalhado salvo em: {arquivo_relatorio}\n")

    return {
        "df_resultado": df_resultado,
        "melhor_metodo": melhor_metodo_nome,
        "jogos_finais": jogos_finais,
    }


if __name__ == "__main__":
    executar_benchmark(n_sorteios_teste=40, n_jogos_por_concurso=10)

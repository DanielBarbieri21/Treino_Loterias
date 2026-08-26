"""
Pipeline Completo Lotofacil - TensorFlow + XGBoost + API Caixa
================================================================
Execucao:
    python treino_tensorflow.py

O script:
  1. Atualiza o Excel com os concursos mais recentes via API oficial da Caixa
  2. Faz engenharia de features avancada
  3. Treina LSTM Bidirecional (TensorFlow) + XGBoost em validacao temporal
  4. Calcula peso de cada modelo pelo seu historico de acertos (backtesting)
  5. Gera ensemble ponderado pelos pesos de backtesting
  6. Seleciona os melhores jogos por Score Multi-Criterio
  7. Salva relatorio .txt com previsoes e metricas
"""

from __future__ import annotations

import sys, io, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import warnings
import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Suprimir logs verbose do TF

# ── Imports locais ────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from buscar_dados_api_v2 import (
    atualizar_excel,
    carregar_binario,
    ARQUIVO_PADRAO,
)
from tensorflow_model import (
    enrich_sequences,
    treinar as treinar_lstm,
    prever_proximas_probs as lstm_probs,
    backtest_lstm,
    WINDOW,
)

# ──────────────────────────────────────────────────────────────
# 2. FEATURES PARA XGBOOST / SKLEARN

# ──────────────────────────────────────────────────────────────

def criar_features_sklearn(binario: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """
    Gera features tabulares a partir do histórico binário.
    Uma linha por sorteio (a partir do índice `window`).

    Features por número (1-25):
      - freq_5, freq_10, freq_20, freq_50  : frequência nas últimas N linhas
      - gap                                 : quantos sorteios desde a última aparição
      - decay (α=0.08)                      : frequência com decaimento exponencial
    Mais:
      - soma_bin                            : total de 1s no último sorteio
      - qtd_pares_ant, qtd_impares_ant      : par/ímpar do sorteio anterior
    """
    N, F = binario.shape
    alpha = 0.08

    rows = []
    for i in range(window, N):
        past = binario[:i]
        feat = []

        for num_idx in range(F):
            col = past[:, num_idx]
            # Frequências rolling
            for w in (5, 10, 20, 50):
                feat.append(float(col[-w:].mean()) if len(col) >= w else float(col.mean()))

            # Gap desde última aparição
            positions = np.where(col == 1)[0]
            gap = float(i - positions[-1] - 1) if len(positions) else float(i)
            feat.append(gap)

            # Decay exponencial
            decay_val = 0.0
            for bit in col:
                decay_val = (1 - alpha) * decay_val + alpha * float(bit)
            feat.append(decay_val)

        # Contexto do último sorteio
        last = past[-1]
        feat.append(float(last.sum()))                          # soma
        feat.append(float(sum(1 for j in range(F) if (j+1) % 2 == 0 and last[j] == 1)))  # pares
        feat.append(float(sum(1 for j in range(F) if (j+1) % 2 != 0 and last[j] == 1)))  # ímpares

        rows.append(feat)

    return np.array(rows, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# 3. TREINAR XGBOOST
# ──────────────────────────────────────────────────────────────

def treinar_xgboost(
    binario: np.ndarray,
    window: int = WINDOW,
    test_size: int = 60,
) -> Tuple[MultiOutputClassifier, np.ndarray, np.ndarray]:
    """
    Treina MultiOutputClassifier(XGBoost) com validação temporal.
    Retorna o modelo, X e y (arrays completos).
    """
    X = criar_features_sklearn(binario, window=window)
    y = binario[window:]

    split = max(len(X) - test_size, int(len(X) * 0.85))
    X_train, y_train = X[:split], y[:split]

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model = MultiOutputClassifier(xgb, n_jobs=-1)
    model.fit(X_train, y_train)

    return model, X, y


# ──────────────────────────────────────────────────────────────
# 4. BACKTESTING DO XGBOOST
# ──────────────────────────────────────────────────────────────

def backtest_xgboost(
    model: MultiOutputClassifier,
    X: np.ndarray,
    y: np.ndarray,
    start: int = -60,
    top_k: int = 15,
) -> dict:
    """Avalia o XGBoost nos últimos abs(start) sorteios."""
    N = len(X)
    start_idx = N + start if start < 0 else start

    hits = []
    for i in range(start_idx, N):
        x_row = X[i : i + 1]

        probs = np.zeros(25)
        for j, clf in enumerate(model.estimators_):
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(x_row)
                if proba.shape[1] > 1:
                    probs[j] = proba[0, 1]

        pred   = set(np.argsort(probs)[-top_k:] + 1)
        actual = set(np.where(y[i])[0] + 1)
        hits.append(len(pred & actual))

    arr = np.array(hits)
    return {
        "mean_hits":   float(arr.mean()),
        "median_hits": float(np.median(arr)),
        "max_hits":    int(arr.max()),
        "min_hits":    int(arr.min()),
        "hits":        hits,
    }


# ──────────────────────────────────────────────────────────────
# 5. ENSEMBLE PONDERADO
# ──────────────────────────────────────────────────────────────

def ensemble_probs(
    lstm_p: np.ndarray,
    xgb_model: MultiOutputClassifier,
    X_last: np.ndarray,
    peso_lstm: float,
    peso_xgb: float,
) -> np.ndarray:
    """
    Combina probabilidades do LSTM e do XGBoost ponderadas pelos pesos de backtesting.
    """
    # Probabilidades XGBoost
    xgb_p = np.zeros(25)
    for j, clf in enumerate(xgb_model.estimators_):
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_last)
            if proba.shape[1] > 1:
                xgb_p[j] = proba[0, 1]

    # Normalizar individualmente
    lstm_p_n = lstm_p / (lstm_p.sum() + 1e-10)
    xgb_p_n  = xgb_p  / (xgb_p.sum()  + 1e-10)

    # Média ponderada pelos acertos de backtesting
    total_peso = peso_lstm + peso_xgb + 1e-10
    combined   = (peso_lstm * lstm_p_n + peso_xgb * xgb_p_n) / total_peso

    return combined


# ──────────────────────────────────────────────────────────────
# 6. SCORE MULTI-CRITÉRIO
# ──────────────────────────────────────────────────────────────

def score_jogo(
    numeros: List[int],
    probs: np.ndarray,
    binario: np.ndarray,
    pesos: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calcula o score de um conjunto de 15 números usando múltiplos critérios.

    Critérios:
      prob_sum       : soma das probabilidades do ensemble
      co_ocorrencia  : soma de co-ocorrências históricas dos pares
      par_impar      : penalidade por desvio do padrão histórico (ideal ~7-8 pares)
      faixas         : recompensa por cobertura das 5 faixas (1-5, ..., 21-25)
      freq_recente   : soma das frequências nas últimas 10 rodadas
    """
    if pesos is None:
        pesos = {
            "prob_sum":      5.0,
            "co_ocorrencia": 2.5,
            "par_impar":     1.5,
            "faixas":        2.0,
            "freq_recente":  1.5,
        }

    n_arr = [n - 1 for n in numeros]   # índices 0-based

    # ── Probabilidade do ensemble ──────────────────────────
    prob_sum = float(sum(probs[i] for i in n_arr))

    # ── Co-ocorrência histórica ────────────────────────────
    co_matrix = np.zeros((25, 25))
    for row in binario[-200:]:           # usar últimos 200 para não ser tão pesado
        idxs = np.where(row == 1)[0]
        for a, b in itertools.combinations(idxs, 2):
            co_matrix[a, b] += 1
            co_matrix[b, a] += 1
    co_score = float(sum(co_matrix[i, j] for i, j in itertools.combinations(n_arr, 2)))
    # Normalizar pela escala
    co_max = co_matrix.max()
    co_score_n = co_score / (co_max * 105 + 1e-10)   # 105 = C(15,2)

    # ── Equilíbrio par/ímpar ───────────────────────────────
    n_pares = sum(1 for n in numeros if n % 2 == 0)
    hist_pares = float(np.mean([row[[j for j in range(25) if (j+2) % 2 == 0]].sum()
                                 for row in binario[-100:]]))
    par_pen = 1.0 - abs(n_pares - hist_pares) / 15.0

    # ── Cobertura por faixas ───────────────────────────────
    faixas = [0, 0, 0, 0, 0]
    for n in numeros:
        faixas[(n - 1) // 5] += 1
    # Penalizar faixas vazias ou muito cheias
    faixa_score = sum(1.0 for f in faixas if f > 0) / 5.0   # 0→1 quanto mais faixas cobertas

    # ── Frequência recente (últimas 10 rodadas) ────────────
    freq_10 = binario[-10:].mean(axis=0)
    freq_rec = float(sum(freq_10[i] for i in n_arr))

    # ── Score final ponderado ──────────────────────────────
    score = (
        pesos["prob_sum"]      * prob_sum      +
        pesos["co_ocorrencia"] * co_score_n    +
        pesos["par_impar"]     * par_pen        +
        pesos["faixas"]        * faixa_score   +
        pesos["freq_recente"]  * freq_rec
    )
    return float(score)


# ──────────────────────────────────────────────────────────────
# 7. GERAÇÃO DOS MELHORES JOGOS
# ──────────────────────────────────────────────────────────────

def gerar_melhores_jogos(
    probs: np.ndarray,
    binario: np.ndarray,
    n_candidatos: int = 2000,
    n_jogos: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """
    Gera n_candidatos combinações de 15 números e seleciona as n_jogos
    com melhor score multi-critério.
    """
    rng = np.random.default_rng(seed)

    # Normalizar probs para amostragem
    probs_norm = probs / (probs.sum() + 1e-10)

    candidatos = set()

    # Determinístico: top-15 puro
    top15 = tuple(sorted(int(i + 1) for i in np.argsort(probs)[-15:]))
    candidatos.add(top15)

    # Variações: top-20 com amostragem aleatória
    top20_idx = np.argsort(probs)[-20:]
    while len(candidatos) < n_candidatos:
        # Método 1: pegar do top-20
        if rng.random() < 0.55:
            chosen = rng.choice(top20_idx, size=15, replace=False)
        # Método 2: amostragem ponderada por probabilidade
        else:
            chosen = rng.choice(25, size=15, replace=False, p=probs_norm)
        candidatos.add(tuple(sorted(int(i + 1) for i in chosen)))

    # Pontuar todos
    scored = [
        {
            "numeros": list(c),
            "score":   score_jogo(list(c), probs, binario),
        }
        for c in candidatos
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n_jogos]


# ──────────────────────────────────────────────────────────────
# 8. RELATÓRIO TEXTUAL
# ──────────────────────────────────────────────────────────────

def salvar_relatorio(
    jogos: List[Dict],
    bt_lstm: dict,
    bt_xgb: dict,
    peso_lstm: float,
    peso_xgb: float,
    ultimo_concurso: int,
    destino: Path,
) -> None:
    linhas = [
        "=" * 60,
        "  PREVISÃO LOTOFÁCIL — TensorFlow + XGBoost Ensemble",
        f"  Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        f"  Baseado no concurso: {ultimo_concurso}",
        "=" * 60,
        "",
        "📊 BACKTESTING (últimos 60 sorteios)",
        f"  LSTM  → média {bt_lstm['mean_hits']:.2f} acertos | max {bt_lstm['max_hits']} | peso {peso_lstm:.3f}",
        f"  XGBst → média {bt_xgb['mean_hits']:.2f} acertos | max {bt_xgb['max_hits']} | peso {peso_xgb:.3f}",
        "",
        "🎯 TOP JOGOS (por Score Multi-Critério)",
        "",
    ]

    for i, jogo in enumerate(jogos, start=1):
        linhas.append(f"  {i:02d}. {jogo['numeros']}  [score={jogo['score']:.4f}]")

    linhas += [
        "",
        "─" * 60,
        "⚠️  Lembre-se: resultados de loterias são aleatórios.",
        "    Use este sistema como ferramenta analítica, não como garantia.",
        "=" * 60,
    ]

    destino.write_text("\n".join(linhas), encoding="utf-8")
    print(f"\n📄 Relatório salvo: {destino}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Lotofacil - Pipeline TensorFlow + XGBoost")
    print("=" * 60 + "\n")

    # ── 1. Atualizar dados e carregar historico ───────────────
    print("[1/6] Atualizando dados via API Caixa e carregando historico...")
    binario, df_raw = carregar_binario(verbose=True)
    print(f"      Sorteios carregados: {len(binario)}  |  numeros: {binario.shape[1]}\n")

    # Ultimo concurso para o relatorio
    ultimo_concurso = int(df_raw["concurso"].max()) if "concurso" in df_raw.columns else len(binario)

    # ── 2. Treinar LSTM ───────────────────────────────────────
    print("[2/6] Treinando modelo LSTM (TensorFlow)...")
    lstm_model, lstm_history = treinar_lstm(
        binario,
        window=WINDOW,
        test_size=60,
        epochs=80,
        batch_size=32,
        verbose=1,
    )
    val_loss = min(lstm_history.get("val_loss", [float("inf")]))
    print(f"      Melhor val_loss: {val_loss:.4f}\n")

    # ── 3. Treinar XGBoost ────────────────────────────────────
    print("[3/6] Treinando XGBoost...")
    xgb_model, X_xgb, y_xgb = treinar_xgboost(binario, window=WINDOW, test_size=60)
    print("      XGBoost treinado.\n")

    # ── 4. Backtesting ────────────────────────────────────────
    print("[4/6] Executando backtesting (ultimos 60 sorteios)...")

    print("      -> LSTM...")
    bt_lstm = backtest_lstm(lstm_model, binario, window=WINDOW, start=-60)

    print("      -> XGBoost...")
    bt_xgb = backtest_xgboost(xgb_model, X_xgb, y_xgb, start=-60)

    print(f"\n  LSTM  -> media {bt_lstm['mean_hits']:.2f} | mediana {bt_lstm['median_hits']:.1f} | max {bt_lstm['max_hits']}")
    print(f"  XGBst -> media {bt_xgb['mean_hits']:.2f} | mediana {bt_xgb['median_hits']:.1f} | max {bt_xgb['max_hits']}\n")

    # Pesos proporcionais a media de acertos
    peso_lstm = bt_lstm["mean_hits"]
    peso_xgb  = bt_xgb["mean_hits"]

    # ── 5. Gerar previsoes ────────────────────────────────────
    print("[5/6] Gerando previsoes com ensemble ponderado...")

    lstm_p  = lstm_probs(lstm_model, binario, window=WINDOW)
    X_last  = criar_features_sklearn(binario, window=WINDOW)[-1:].copy()
    probs_final = ensemble_probs(lstm_p, xgb_model, X_last, peso_lstm, peso_xgb)

    jogos = gerar_melhores_jogos(
        probs_final,
        binario,
        n_candidatos=3000,
        n_jogos=10,
        seed=42,
    )

    # ── 6. Exibir e salvar resultados ─────────────────────────
    print("[6/6] Salvando resultados...\n")

    print("=" * 60)
    print("  TOP 10 JOGOS RECOMENDADOS")
    print("=" * 60)
    for i, jogo in enumerate(jogos, start=1):
        print(f"  {i:02d}. {jogo['numeros']}  [score={jogo['score']:.4f}]")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(__file__).parent / f"previsao_tf_{ts}.txt"
    salvar_relatorio(jogos, bt_lstm, bt_xgb, peso_lstm, peso_xgb, ultimo_concurso, dest)

    modelo_path = Path(__file__).parent / "modelo_lstm.keras"
    lstm_model.save(str(modelo_path))
    print(f"Modelo LSTM salvo: {modelo_path}\n")

    return jogos


if __name__ == "__main__":
    main()

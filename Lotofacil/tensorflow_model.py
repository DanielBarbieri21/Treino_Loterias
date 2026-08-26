"""
Modelo TensorFlow/Keras - Lotofacil
=====================================
Arquitetura:
  Input (janela_temporal, 25) -> BiLSTM(128) -> Dropout(0.3)
                              -> Dense(64, swish) -> Dropout(0.2)
                              -> Dense(25, sigmoid)

Cada saida e a probabilidade do numero (1-25) aparecer no proximo sorteio.
"""

from __future__ import annotations

import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from typing import Tuple, Optional
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Hiperparâmetros padrão
# ──────────────────────────────────────────────────────────────
WINDOW       = 20    # concursos passados usados como entrada
LSTM_UNITS   = 128   # neurônios por camada LSTM
DENSE_UNITS  = 64
DROPOUT      = 0.30
DROPOUT2     = 0.20
EPOCHS       = 80
BATCH_SIZE   = 32
PATIENCE     = 15    # early stopping
LR           = 1e-3


# ──────────────────────────────────────────────────────────────
# Feature engineering para LSTM
# ──────────────────────────────────────────────────────────────

def build_sequences(
    binario: np.ndarray,
    window: int = WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constrói pares (X, y) a partir da matriz binária de sorteios.

    Args:
        binario : array (N, 25) com 0/1 indicando presença de cada número
        window  : tamanho da janela temporal

    Returns:
        X : (N-window, window, 25)   – sequências de entrada
        y : (N-window, 25)           – rótulo do próximo sorteio
    """
    X_list, y_list = [], []
    for i in range(window, len(binario)):
        X_list.append(binario[i - window : i])   # janela histórica
        y_list.append(binario[i])                 # alvo

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def enrich_sequences(binario: np.ndarray, window: int = WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adiciona features derivadas a cada timestep da sequência:
      - bits 0-24  : presença do número (0/1)
      - bit  25    : soma da linha normalizada
      - bits 26-30 : contagem por faixa (1-5, 6-10, 11-15, 16-20, 21-25) norm.
      - bits 31-55 : frequência acumulada dos últimos `window` concursos por número

    Total de features por timestep: 25 + 1 + 5 + 25 = 56
    """
    N, F = binario.shape   # F == 25

    # Pré-calcular frequências rolling
    extended = []
    for t in range(len(binario)):
        row   = binario[t]

        # Soma normalizada
        soma_norm = row.sum() / F

        # Distribuição por faixas
        faixas = np.array([
            row[0:5].sum(),
            row[5:10].sum(),
            row[10:15].sum(),
            row[15:20].sum(),
            row[20:25].sum(),
        ], dtype=np.float32) / 5.0   # normaliza por tamanho da faixa

        # Frequência nos últimos `window` concursos
        start = max(0, t - window)
        freq  = binario[start:t].mean(axis=0) if t > 0 else np.zeros(F, dtype=np.float32)

        extended.append(np.concatenate([row, [soma_norm], faixas, freq]))

    extended_arr = np.array(extended, dtype=np.float32)  # (N, 56)

    X_list, y_list = [], []
    for i in range(window, N):
        X_list.append(extended_arr[i - window : i])
        y_list.append(binario[i])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# Construção do modelo
# ──────────────────────────────────────────────────────────────

def build_model(input_shape: Tuple[int, int]) -> "tf.keras.Model":
    """
    Cria e compila o modelo BiLSTM.

    Args:
        input_shape : (window, n_features)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    inp = keras.Input(shape=input_shape, name="sequencia")

    # ── Bloco 1: BiLSTM ──────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4)),
        name="bilstm_1",
    )(inp)
    x = layers.Dropout(DROPOUT, name="drop_1")(x)

    # ── Bloco 2: LSTM ────────────────────────────────────────
    x = layers.LSTM(LSTM_UNITS // 2, kernel_regularizer=regularizers.l2(1e-4), name="lstm_2")(x)
    x = layers.Dropout(DROPOUT, name="drop_2")(x)

    # ── Bloco Dense ──────────────────────────────────────────
    x = layers.Dense(DENSE_UNITS, activation="swish", kernel_regularizer=regularizers.l2(1e-4), name="dense_1")(x)
    x = layers.Dropout(DROPOUT2, name="drop_3")(x)

    # ── Saída: 25 probabilidades ─────────────────────────────
    out = layers.Dense(25, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="LotoLSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────
# Treinamento
# ──────────────────────────────────────────────────────────────

def treinar(
    binario: np.ndarray,
    window: int = WINDOW,
    test_size: int = 60,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    verbose: int = 1,
) -> Tuple["tf.keras.Model", dict]:
    """
    Treina o modelo usando validação temporal estrita.

    Args:
        binario   : array (N, 25) binário dos sorteios
        window    : tamanho da janela de entrada
        test_size : últimos N sorteios reservados como teste
        epochs    : épocas máximas (early stopping pode interromper antes)
        verbose   : 0=silencioso, 1=barra de progresso, 2=por época

    Returns:
        model    : modelo treinado
        history  : histórico do treinamento (dict com loss, val_loss, etc.)
    """
    import tensorflow as tf
    from tensorflow import keras

    X, y = enrich_sequences(binario, window=window)

    # Divisão temporal: sem data leakage
    split = max(len(X) - test_size, int(len(X) * 0.85))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"  📐 Shape treino: {X_train.shape}  |  val: {X_val.shape}")

    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    return model, history.history


# ──────────────────────────────────────────────────────────────
# Predição
# ──────────────────────────────────────────────────────────────

def prever_proximas_probs(
    model: "tf.keras.Model",
    binario: np.ndarray,
    window: int = WINDOW,
) -> np.ndarray:
    """
    Retorna array (25,) com a probabilidade estimada de cada número
    (1-25) aparecer no próximo sorteio, baseando-se nos últimos `window`
    concursos do histórico.
    """
    N, F = binario.shape

    # Construir features para a última janela
    extended = []
    for t in range(max(0, N - window), N):
        row       = binario[t]
        soma_norm = row.sum() / F
        faixas    = np.array([
            row[0:5].sum(),
            row[5:10].sum(),
            row[10:15].sum(),
            row[15:20].sum(),
            row[20:25].sum(),
        ], dtype=np.float32) / 5.0

        start = max(0, t - window)
        freq  = binario[start:t].mean(axis=0) if t > 0 else np.zeros(F, dtype=np.float32)
        extended.append(np.concatenate([row, [soma_norm], faixas, freq]))

    seq = np.array(extended[-window:], dtype=np.float32)
    seq = seq[np.newaxis, ...]   # (1, window, features)

    probs = model.predict(seq, verbose=0)[0]  # (25,)
    return probs.astype(float)


# ──────────────────────────────────────────────────────────────
# Backtesting do LSTM
# ──────────────────────────────────────────────────────────────

def backtest_lstm(
    model: "tf.keras.Model",
    binario: np.ndarray,
    window: int = WINDOW,
    start: int = -60,
    top_k: int = 15,
) -> dict:
    """
    Avalia o modelo nos últimos abs(start) concursos.
    Para cada concurso i, prediz com os dados anteriores e conta acertos.

    Returns:
        dict com mean_hits, median_hits, max_hits, hits (lista)
    """
    N = len(binario)
    start_idx = N + start if start < 0 else start

    hits = []
    for i in range(start_idx, N):
        past   = binario[:i]
        probs  = prever_proximas_probs(model, past, window=window)
        pred   = set(np.argsort(probs)[-top_k:] + 1)   # converte índice→número 1-based
        actual = set(np.where(binario[i])[0] + 1)
        hits.append(len(pred & actual))

    arr = np.array(hits)
    return {
        "mean_hits":   float(arr.mean()),
        "median_hits": float(np.median(arr)),
        "max_hits":    int(arr.max()),
        "min_hits":    int(arr.min()),
        "hits":        hits,
    }

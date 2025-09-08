import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, hamming_loss, jaccard_score
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

# ================================
# 📂 Carregar dados
# ================================
ARQUIVO_TREINO = "treino.xlsx"
df = pd.read_excel(ARQUIVO_TREINO)

# Filtrar apenas números válidos (1 a 25)
df_numeros = df.applymap(
    lambda x: int(x) if str(x).isdigit() and 1 <= int(x) <= 25 else np.nan
).dropna(how="all")

df_numeros = df_numeros.reset_index(drop=True)

# Converter y em binário (one-hot encoding 1-25)
y_binario = pd.DataFrame(0, index=np.arange(len(df_numeros)), columns=range(1, 26))
for i, linha in enumerate(df_numeros.values):
    for n in linha:
        if not pd.isna(n):
            y_binario.at[i, int(n)] = 1

# ================================
# 🧮 Calcular Matriz de Co-ocorrência Histórica (Nova Adição)
# ================================
# Matriz simétrica 26x26 (índices 1-25, 0 ignorado)
co_matrix = np.zeros((26, 26))
for i in range(len(df_numeros)):
    nums = [int(n) for n in df_numeros.iloc[i].dropna() if not pd.isna(n)]
    for pair in itertools.combinations(nums, 2):
        co_matrix[pair[0], pair[1]] += 1
        co_matrix[pair[1], pair[0]] += 1  # Simétrica

# Normalizar co_matrix por frequência total (opcional, para probabilidades)
total_pares = co_matrix.sum() / 2  # Dividido por 2 pois simétrica
if total_pares > 0:
    co_matrix = co_matrix / total_pares

# Exibir top 20 pares mais frequentes (para análise)
st.sidebar.subheader("🔍 Top 20 Pares Mais Frequentes (Análise Histórica)")
pares_freq = []
for i in range(1, 26):
    for j in range(i + 1, 26):
        freq = co_matrix[i, j]
        if freq > 0:
            pares_freq.append(((i, j), freq))
pares_freq.sort(key=lambda x: x[1], reverse=True)
top_pares_df = pd.DataFrame(pares_freq[:20], columns=["Par", "Frequência Normalizada"])
st.sidebar.table(top_pares_df)


# ================================
# ✨ Feature Engineering Melhorado (Adiciona Co-ocorrências)
# ================================
def criar_features(df_num, co_matrix):
    df_num = df_num.reset_index(drop=True)  # garante índice sequencial
    features = pd.DataFrame(index=df_num.index)

    # Frequência acumulada (original)
    freq_acumulada = pd.DataFrame(0, index=df_num.index, columns=range(1, 26))
    for i in range(1, len(df_num)):
        freq_acumulada.iloc[i] = freq_acumulada.iloc[i - 1]
        for n in df_num.iloc[i].dropna():
            freq_acumulada.iloc[i, int(n) - 1] += 1  # ajuste -1 para col 0-24?

    # Última ocorrência (original)
    ultima_ocorrencia = pd.DataFrame(-1, index=df_num.index, columns=range(1, 26))
    for i in range(len(df_num)):
        if i > 0:
            ultima_ocorrencia.iloc[i] = ultima_ocorrencia.iloc[i - 1]
        for n in df_num.iloc[i].dropna():
            ultima_ocorrencia.iloc[i, int(n) - 1] = i

    # Distância desde última ocorrência (original)
    distancia = (features.index.values.reshape(-1, 1) - ultima_ocorrencia.values)

    # Estatísticas gerais do concurso (original)
    qtd_pares = df_num.apply(lambda row: sum(n % 2 == 0 for n in row.dropna()), axis=1)
    qtd_multiplos3 = df_num.apply(lambda row: sum(n % 3 == 0 for n in row.dropna()), axis=1)

    # NOVA: Features de Co-ocorrência
    # Para cada linha i, calcule média de co-ocorrência com números do sorteio anterior (i-1)
    co_features = []
    for i in range(len(df_num)):
        if i == 0:
            co_features.append([0] * 25)  # Zeros para o primeiro
        else:
            prev_nums = [int(n) for n in df_num.iloc[i - 1].dropna() if not pd.isna(n)]
            row_co = []
            for j in range(1, 26):
                if len(prev_nums) > 0:
                    co_mean = np.mean([co_matrix[j, p] for p in prev_nums if p != j])
                else:
                    co_mean = 0
                row_co.append(co_mean)
            co_features.append(row_co)
    co_df = pd.DataFrame(co_features, index=df_num.index, columns=[f"co_mean_{j}" for j in range(1, 26)])

    # Concatenar todas as features (originais + novas co)
    features = pd.concat([
        pd.DataFrame(freq_acumulada.values, index=df_num.index, columns=[f"freq_{i}" for i in range(1, 26)]),
        pd.DataFrame(distancia, index=df_num.index, columns=[f"dist_{i}" for i in range(1, 26)]),
        qtd_pares.rename("qtd_pares"),
        qtd_multiplos3.rename("qtd_multiplos3"),
        co_df  # Nova adição
    ], axis=1)

    return features.fillna(0)


X = criar_features(df_numeros, co_matrix)

# ================================
# 🔀 Validação Temporal
# ================================
tscv = TimeSeriesSplit(n_splits=10)  # Aumentado para 10 splits (melhoria)

# ================================
# 🤖 Modelos (Mantido, mas com mais tuning se quiser)
# ================================
rf_base = RandomForestClassifier(random_state=42)
xgb_base = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)

# Hyperparameter tuning para XGBoost (expandido ligeiramente)
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}
search_xgb = RandomizedSearchCV(
    xgb_base, param_distributions=param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1  # Aumentado n_iter e cv
)
search_xgb.fit(X, y_binario)
best_xgb = search_xgb.best_estimator_

# MultiOutput com Voting (RF + XGB)
ensemble = MultiOutputClassifier(
    VotingClassifier(estimators=[("rf", rf_base), ("xgb", best_xgb)], voting="soft")
)

ensemble.fit(X, y_binario)

# ================================
# 📊 Avaliação (Melhorada com mais métricas)
# ================================
y_pred = ensemble.predict(X)

report = classification_report(y_binario, y_pred, target_names=[str(i) for i in range(1, 26)], output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Métricas extras (originais)
hamming = hamming_loss(y_binario, y_pred)
jaccard = jaccard_score(y_binario, y_pred, average="samples")

# NOVA: Métrica de acertos parciais (ex.: média de acertos por sorteio)
acertos_parciais = []
for i in range(len(y_binario)):
    acerto_real = sum(y_binario.iloc[i])
    acerto_pred = sum(y_pred[i])
    acerto_intersecao = sum(y_binario.iloc[i] & y_pred[i])
    acertos_parciais.append(acerto_intersecao)
media_acertos = np.mean(acertos_parciais)


# ================================
# 🎲 Função para Score de Sequência Baseado em Co-ocorrência (Nova)
# ================================
def score_sequencia(nums, co_matrix):
    """
    Calcula o score de uma sequência baseado na soma de co-ocorrências dos pares.
    Quanto maior, melhor (mais padrões históricos).
    """
    nums = [int(n) for n in nums]
    score = 0
    for pair in itertools.combinations(nums, 2):
        score += co_matrix[pair[0], pair[1]]
    return score


# ================================
# 🎲 Função para gerar previsões probabilísticas (Melhorada com Scoring)
# ================================
def gerar_previsoes_probabilisticas(modelo, n_previsoes=10, n_numeros=15, co_matrix=None, n_candidatos=100):
    previsoes = []

    # Probabilidades médias de cada número (original)
    probs = np.zeros(25)

    # Pega último sorteio (linha mais recente)
    ultima_entrada = X.values[-1].reshape(1, -1)

    # Cada número (1–25) tem seu próprio classificador binário
    for j, clf in enumerate(modelo.estimators_):
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(ultima_entrada)  # array (1, 2)
            if proba.shape[1] > 1:
                probs[j] = proba[0, 1]  # probabilidade da classe "1"

    # Normaliza soma = 1
    probs = probs / probs.sum()

    # Gera candidatos e seleciona top por score de co-ocorrência
    candidatos = []
    for _ in range(n_candidatos):  # Gere mais candidatos para seleção
        numeros = np.random.choice(np.arange(1, 26), size=n_numeros, replace=False, p=probs)
        numeros = sorted([int(n) for n in numeros])
        if co_matrix is not None:
            score = score_sequencia(numeros, co_matrix)
        else:
            score = 0  # Fallback se sem co_matrix
        candidatos.append((numeros, score))

    # Ordena por score descrescente e pega top n_previsoes
    candidatos.sort(key=lambda x: x[1], reverse=True)
    previsoes = [c[0] for c in candidatos[:n_previsoes]]

    return previsoes, probs


# ================================
# 🎲 Função para gerar previsões com misto (Melhorada com Scoring)
# ================================
def gerar_previsoes_misto(modelo, n_previsoes=6, n_numeros=15, n_mais=11, n_menos=4, co_matrix=None):
    previsoes = []

    # Probabilidades médias de cada número (original)
    probs = np.zeros(25)

    # Pega último sorteio (linha mais recente)
    ultima_entrada = X.values[-1].reshape(1, -1)

    # Cada número (1–25) tem seu próprio classificador binário
    for j, clf in enumerate(modelo.estimators_):
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(ultima_entrada)  # array (1, 2)
            if proba.shape[1] > 1:
                probs[j] = proba[0, 1]  # probabilidade da classe "1"

    # Normaliza soma = 1
    probs = probs / probs.sum()

    # Ordena os números por probabilidade (índices de 0 a 24, correspondendo a 1-25)
    indices = np.arange(25)
    sorted_indices = np.argsort(probs)[::-1]

    # Divide em mais prováveis e menos prováveis
    mais_provaveis = sorted_indices[:15]
    menos_provaveis = sorted_indices[15:]

    # Gera candidatos mistos e seleciona por score
    candidatos = []
    for _ in range(50):  # Gere mais para seleção
        # Seleciona 11 números mais prováveis
        top_numeros = mais_provaveis[:n_mais].copy()

        # Seleciona 4 números menos prováveis aleatoriamente
        menos_numeros = np.random.choice(menos_provaveis, size=n_menos, replace=False)

        # Combina os números
        numeros = np.concatenate([top_numeros, menos_numeros])

        # Converte índices (0-24) para números (1-25)
        numeros = [int(n + 1) for n in numeros]

        # Embaralha
        np.random.shuffle(numeros)
        numeros_sorted = sorted(numeros)

        if co_matrix is not None:
            score = score_sequencia(numeros_sorted, co_matrix)
        else:
            score = 0
        candidatos.append((numeros_sorted, score))

    # Ordena por score e pega top
    candidatos.sort(key=lambda x: x[1], reverse=True)
    previsoes = [c[0] for c in candidatos[:n_previsoes]]

    return previsoes


# Modelo final
modelo_final = ensemble
joblib.dump(modelo_final, "modelo_final.pkl")

# Gerar previsões originais (10 sequências, melhoradas)
previsoes_originais, probs = gerar_previsoes_probabilisticas(
    modelo_final, n_previsoes=10, n_numeros=15, co_matrix=co_matrix, n_candidatos=200
)

# Gerar previsões com misto (6 sequências, melhoradas)
previsoes_misto = gerar_previsoes_misto(
    modelo_final, n_previsoes=6, n_numeros=15, n_mais=11, n_menos=4, co_matrix=co_matrix
)

# ================================
# 🌐 Interface Streamlit (Atualizada com Novas Análises)
# ================================
st.title("🎲 Previsão Lotofácil com Machine Learning Melhorado")

# Frequência dos números (original)
st.subheader("📊 Frequência dos números no histórico")
fig, ax = plt.subplots(figsize=(10, 4))
df_numeros.stack().value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Número")
ax.set_ylabel("Frequência")
st.pyplot(fig)

# Relatório (original)
st.subheader("📑 Relatório de Classificação (Ensemble)")
st.dataframe(df_report.style.format("{:.2f}"))

st.write(f"**Hamming Loss:** {hamming:.4f}")
st.write(f"**Jaccard Score (samples):** {jaccard:.4f}")

# NOVA: Métrica de acertos parciais
st.write(f"**Média de Acertos Parciais por Sorteio:** {media_acertos:.2f}")

# Matriz de confusão (classe 1 exemplo, original)
st.subheader("🔍 Matriz de Confusão - Classe 1")
cm = confusion_matrix(y_binario.iloc[:, 0], y_pred[:, 0])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Matriz de Confusão - Classe 1")
st.pyplot(fig)

# Probabilidades previstas de cada número (original)
st.subheader("📈 Probabilidade prevista de cada número (último sorteio)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(1, 26), probs)
ax.set_xticks(range(1, 26))
ax.set_xlabel("Número")
ax.set_ylabel("Probabilidade prevista")
st.pyplot(fig)

# Ranking dos 15 números mais prováveis (original)
st.subheader("🏆 Top 15 números mais prováveis")
ranking = pd.DataFrame({
    "Número": np.arange(1, 26),
    "Probabilidade": probs
}).sort_values(by="Probabilidade", ascending=False).reset_index(drop=True)
st.table(ranking.head(15).style.format({"Probabilidade": "{:.4f}"}))

# NOVA: Visualização da Matriz de Co-ocorrência (Heatmap Top)
st.subheader("🔗 Heatmap de Co-ocorrência (Top Correlações Históricas)")
fig, ax = plt.subplots(figsize=(10, 8))
# Focar em submatriz 1-25
sns.heatmap(co_matrix[1:26, 1:26], annot=False, cmap="YlOrRd", ax=ax, cbar_kws={'label': 'Frequência Normalizada'})
ax.set_title("Matriz de Co-ocorrência entre Números (1-25)")
ax.set_xlabel("Número")
ax.set_ylabel("Número")
st.pyplot(fig)

# Previsões originais (melhoradas)
st.subheader("🎯 10 Previsões Probabilísticas Melhoradas (com Scoring de Correlações)")
for i, prev in enumerate(previsoes_originais, start=1):
    score = score_sequencia(prev, co_matrix)
    st.write(f"**Previsão {i} (Score Co-oc: {score:.2f}):** {prev}")

# Previsões com misto (melhoradas)
st.subheader("🎯 6 Previsões Mistas Melhoradas (11 mais prováveis + 4 menos, com Scoring)")
for i, prev in enumerate(previsoes_misto, start=1):
    score = score_sequencia(prev, co_matrix)
    st.write(f"**Previsão {i} (Score Co-oc: {score:.2f}):** {prev}")

# NOVA: Seção de Análise Adicional
st.subheader("📊 Análise Adicional: Backtesting Simples")
# Exemplo simples: Verificar acertos parciais nas últimas 5 previsões vs. histórico recente (ajuste se quiser mais)
if len(previsoes_originais) > 0:
    ultima_real = [int(n) for n in df_numeros.iloc[-1].dropna()]
    for i, prev in enumerate(previsoes_originais[:3], start=1):  # Top 3 como exemplo
        acertos = len(set(prev) & set(ultima_real))
        st.write(f"**Simulação: Previsão {i} vs. Último Sorteio Real - Acertos:** {acertos}/15")
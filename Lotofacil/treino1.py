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
# ✨ Feature Engineering
# ================================
def criar_features(df_num):
    df_num = df_num.reset_index(drop=True)  # garante índice sequencial
    features = pd.DataFrame(index=df_num.index)

    # Frequência acumulada
    freq_acumulada = pd.DataFrame(0, index=df_num.index, columns=range(1, 26))
    for i in range(1, len(df_num)):
        freq_acumulada.iloc[i] = freq_acumulada.iloc[i-1]
        for n in df_num.iloc[i].dropna():
            freq_acumulada.iloc[i, int(n)-1] += 1  # ajuste -1

    # Última ocorrência
    ultima_ocorrencia = pd.DataFrame(-1, index=df_num.index, columns=range(1, 26))
    for i in range(len(df_num)):
        if i > 0:
            ultima_ocorrencia.iloc[i] = ultima_ocorrencia.iloc[i-1]
        for n in df_num.iloc[i].dropna():
            ultima_ocorrencia.iloc[i, int(n)-1] = i

    # Distância desde última ocorrência
    distancia = (features.index.values.reshape(-1, 1) - ultima_ocorrencia.values)

    # Estatísticas gerais do concurso
    qtd_pares = df_num.apply(lambda row: sum(n % 2 == 0 for n in row.dropna()), axis=1)
    qtd_multiplos3 = df_num.apply(lambda row: sum(n % 3 == 0 for n in row.dropna()), axis=1)

    # Concatenar
    features = pd.concat([
        pd.DataFrame(freq_acumulada.values, index=df_num.index, columns=[f"freq_{i}" for i in range(1, 26)]),
        pd.DataFrame(distancia, index=df_num.index, columns=[f"dist_{i}" for i in range(1, 26)]),
        qtd_pares.rename("qtd_pares"),
        qtd_multiplos3.rename("qtd_multiplos3"),
    ], axis=1)

    return features.fillna(0)

X = criar_features(df_numeros)

# ================================
# 🔀 Validação Temporal
# ================================
tscv = TimeSeriesSplit(n_splits=5)

# ================================
# 🤖 Modelos
# ================================
rf_base = RandomForestClassifier(random_state=42)
xgb_base = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)

# Hyperparameter tuning para XGBoost (exemplo simplificado)
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}
search_xgb = RandomizedSearchCV(
    xgb_base, param_distributions=param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1
)
search_xgb.fit(X, y_binario)
best_xgb = search_xgb.best_estimator_

# MultiOutput com Voting (RF + XGB)
ensemble = MultiOutputClassifier(
    VotingClassifier(estimators=[("rf", rf_base), ("xgb", best_xgb)], voting="soft")
)

ensemble.fit(X, y_binario)

# ================================
# 📊 Avaliação
# ================================
y_pred = ensemble.predict(X)

report = classification_report(y_binario, y_pred, target_names=[str(i) for i in range(1, 26)], output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Métricas extras
hamming = hamming_loss(y_binario, y_pred)
jaccard = jaccard_score(y_binario, y_pred, average="samples")

# ================================
# 🎲 Função para gerar previsões probabilísticas
# ================================
def gerar_previsoes_probabilisticas(modelo, n_previsoes=10, n_numeros=15):
    previsoes = []

    # Probabilidades médias de cada número
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

    # Gera as previsões
    for _ in range(n_previsoes):
        numeros = np.random.choice(np.arange(1, 26), size=n_numeros, replace=False, p=probs)
        numeros = [int(n) for n in numeros]
        previsoes.append(sorted(numeros))

    return previsoes, probs

# Modelo final
modelo_final = ensemble
joblib.dump(modelo_final, "modelo_final.pkl")

previsoes, probs = gerar_previsoes_probabilisticas(modelo_final, n_previsoes=10, n_numeros=15)

# ================================
# 🌐 Interface Streamlit
# ================================
st.title("🎲 Previsão Lotofácil com Machine Learning")

# Frequência dos números
st.subheader("📊 Frequência dos números no histórico")
fig, ax = plt.subplots(figsize=(10, 4))
df_numeros.stack().value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Número")
ax.set_ylabel("Frequência")
st.pyplot(fig)

# Relatório
st.subheader("📑 Relatório de Classificação (Ensemble)")
st.dataframe(df_report.style.format("{:.2f}"))

st.write(f"**Hamming Loss:** {hamming:.4f}")
st.write(f"**Jaccard Score (samples):** {jaccard:.4f}")

# Matriz de confusão (classe 1 exemplo)
st.subheader("🔍 Matriz de Confusão - Classe 1")
cm = confusion_matrix(y_binario.iloc[:, 0], y_pred[:, 0])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Matriz de Confusão - Classe 1")
st.pyplot(fig)

# Probabilidades previstas de cada número
st.subheader("📈 Probabilidade prevista de cada número (último sorteio)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(1, 26), probs)
ax.set_xticks(range(1, 26))
ax.set_xlabel("Número")
ax.set_ylabel("Probabilidade prevista")
st.pyplot(fig)

# Ranking dos 15 números mais prováveis
st.subheader("🏆 Top 15 números mais prováveis")
ranking = pd.DataFrame({
    "Número": np.arange(1, 26),
    "Probabilidade": probs
}).sort_values(by="Probabilidade", ascending=False).reset_index(drop=True)
st.table(ranking.head(15).style.format({"Probabilidade": "{:.4f}"}))

# Previsões probabilísticas
st.subheader("🎯 10 Previsões Probabilísticas (15 números cada)")
for i, prev in enumerate(previsoes, start=1):
    st.write(f"**Previsão {i}:** {prev}")

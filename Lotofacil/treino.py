import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.multioutput import MultiOutputClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
import joblib

# ================================
# 📂 Carregar dados
# ================================
ARQUIVO_TREINO = "mega_sena.xlsx"
df = pd.read_excel(ARQUIVO_TREINO)

# Filtrar apenas números válidos (1 a 25)
df_numeros = df.applymap(
    lambda x: int(x) if str(x).isdigit() and 1 <= int(x) <= 25 else np.nan
).dropna(how="all")

# Converter y em binário (one-hot encoding 1-25)
y_binario = pd.DataFrame(0, index=np.arange(len(df_numeros)), columns=range(1, 26))
for i, linha in enumerate(df_numeros.values):
    for n in linha:
        if not pd.isna(n):
            y_binario.at[i, int(n)] = 1

# X simples: índice dos sorteios
X = np.arange(len(df_numeros)).reshape(-1, 1)

# ================================
# 🔀 Divisão treino/teste
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binario, test_size=0.2, random_state=42
)

# ================================
# 🤖 Modelos
# ================================
dummy = MultiOutputClassifier(DummyClassifier(strategy="most_frequent"))
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb = MultiOutputClassifier(XGBClassifier(
    eval_metric="logloss", use_label_encoder=False, random_state=42
))
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# ================================
# 📊 Função de relatório
# ================================
def relatorio_metricas(y_true, y_pred, nome):
    report = classification_report(
        y_true, y_pred, target_names=[str(i) for i in range(1, 26)], output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    df_report.index.name = f"Métricas - {nome}"
    return df_report

df_dummy = relatorio_metricas(y_test, y_pred_dummy, "Dummy")
df_rf = relatorio_metricas(y_test, y_pred_rf, "RandomForest")
df_xgb = relatorio_metricas(y_test, y_pred_xgb, "XGBoost")

# ================================
# 🎲 Função para gerar previsões probabilísticas
# ================================
def gerar_previsoes_probabilisticas(modelo, n_previsoes=10, n_numeros=15):
    previsoes = []

    # Probabilidades médias de cada número (1-25)
    probs = np.zeros(25)
    for i, clf in enumerate(modelo.estimators_):
        proba = clf.predict_proba([[len(df_numeros)]])[0]
        if len(proba) > 1:
            probs[i] = proba[1]

    probs = probs / probs.sum()  # normalizar soma=1

    for _ in range(n_previsoes):
        numeros = np.random.choice(np.arange(1, 26), size=n_numeros, replace=False, p=probs)
        numeros = [int(n) for n in numeros]  # 🔹 converte np.int64 -> int
        previsoes.append(sorted(numeros))

    return previsoes

# Modelo final
modelo_final = xgb
joblib.dump(modelo_final, "modelo_xgb.pkl")

previsoes = gerar_previsoes_probabilisticas(modelo_final, n_previsoes=10, n_numeros=15)

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

# Relatórios
st.subheader("📑 Relatórios de Classificação")
st.write("**Dummy Classifier (baseline):**")
st.dataframe(df_dummy.style.format("{:.2f}"))
st.write("**RandomForest:**")
st.dataframe(df_rf.style.format("{:.2f}"))
st.write("**XGBoost:**")
st.dataframe(df_xgb.style.format("{:.2f}"))

# Matriz de confusão (classe 1 exemplo)
st.subheader("🔍 Matriz de Confusão - RandomForest (classe 1)")
cm = confusion_matrix(y_test.iloc[:, 0], y_pred_rf[:, 0])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Matriz de Confusão - Classe 1")
st.pyplot(fig)

# Curva ROC (classe 1)
st.subheader("📈 Curva ROC - XGBoost (classe 1)")
y_score = modelo_final.estimators_[0].predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test.iloc[:, 0], y_score)
roc_auc = roc_auc_score(y_test.iloc[:, 0], y_score)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Curva ROC (classe 1)")
ax.legend()
st.pyplot(fig)

# Previsões probabilísticas
st.subheader("🎯 10 Previsões Probabilísticas (15 números cada)")
for i, prev in enumerate(previsoes, start=1):
    st.write(f"**Previsão {i}:** {prev}")

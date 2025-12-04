"""
🎯 Sistema Avançado de Análise de Padrões para Lotofácil
==========================================================
Este sistema analisa múltiplos tipos de padrões para identificar
falhas e aumentar a probabilidade de acerto.

Padrões analisados:
- Temporais: ciclos, tendências, sazonalidade
- Sequências: consecutivos, intervalos, gaps
- Grupos: faixas numéricas, clusters, proximidade
- Repetição: retorno após X sorteios, frequência cíclica
- Estatísticos: distribuições, correlações avançadas
"""

import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from collections import defaultdict, Counter
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, jaccard_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ================================
# 📂 Carregar Dados
# ================================
ARQUIVO_TREINO = "treino.xlsx"

@st.cache_data
def carregar_dados():
    """Carrega e prepara os dados históricos"""
    try:
        df = pd.read_excel(ARQUIVO_TREINO)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None
    
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
    
    return df_numeros, y_binario

df_numeros, y_binario = carregar_dados()

if df_numeros is None:
    st.stop()

# ================================
# 🔍 ANÁLISE DE PADRÕES TEMPORAIS
# ================================
def analisar_padroes_temporais(df_num):
    """Analisa padrões temporais: ciclos, tendências, sazonalidade"""
    padroes = {}
    
    # 1. Frequência por posição temporal (últimos N sorteios)
    janelas = [5, 10, 20, 50]
    for janela in janelas:
        if len(df_num) >= janela:
            ultimos = df_num.iloc[-janela:]
            freq_recente = {}
            for num in range(1, 26):
                freq_recente[num] = ultimos.apply(lambda row: num in row.values, axis=1).sum()
            padroes[f'freq_janela_{janela}'] = freq_recente
    
    # 2. Análise de ciclos (padrões que se repetem)
    ciclos = {}
    for num in range(1, 26):
        ocorrencias = []
        for i, row in df_num.iterrows():
            if num in row.values:
                ocorrencias.append(i)
        
        if len(ocorrencias) > 2:
            intervalos = [ocorrencias[i+1] - ocorrencias[i] for i in range(len(ocorrencias)-1)]
            if intervalos:
                intervalo_medio = np.mean(intervalos)
                intervalo_std = np.std(intervalos)
                ciclos[num] = {
                    'medio': intervalo_medio,
                    'std': intervalo_std,
                    'proximo_esperado': ocorrencias[-1] + intervalo_medio
                }
    
    padroes['ciclos'] = ciclos
    
    # 3. Tendência (aumentando ou diminuindo frequência)
    tendencias = {}
    for num in range(1, 26):
        ocorrencias = []
        for i, row in df_num.iterrows():
            ocorrencias.append(1 if num in row.values else 0)
        
        if len(ocorrencias) > 10:
            # Dividir em duas metades e comparar
            metade1 = np.mean(ocorrencias[:len(ocorrencias)//2])
            metade2 = np.mean(ocorrencias[len(ocorrencias)//2:])
            tendencias[num] = metade2 - metade1  # Positivo = aumentando
    
    padroes['tendencias'] = tendencias
    
    return padroes

# ================================
# 🔗 ANÁLISE DE PADRÕES DE SEQUÊNCIAS
# ================================
def analisar_padroes_sequenciais(df_num):
    """Analisa padrões de sequências: consecutivos, intervalos, gaps"""
    padroes = {}
    
    # 1. Números consecutivos (ex: 5,6,7)
    consecutivos_por_sorteio = []
    for i, row in df_num.iterrows():
        nums = sorted([int(n) for n in row.dropna()])
        consecutivos = []
        for j in range(len(nums)-1):
            if nums[j+1] - nums[j] == 1:
                consecutivos.append((nums[j], nums[j+1]))
        consecutivos_por_sorteio.append(consecutivos)
    
    padroes['consecutivos_por_sorteio'] = consecutivos_por_sorteio
    
    # 2. Intervalos médios entre números
    intervalos_medios = []
    for i, row in df_num.iterrows():
        nums = sorted([int(n) for n in row.dropna()])
        intervalos = [nums[j+1] - nums[j] for j in range(len(nums)-1)]
        intervalos_medios.append(np.mean(intervalos) if intervalos else 0)
    
    padroes['intervalo_medio'] = np.mean(intervalos_medios)
    
    # 3. Gaps (lacunas grandes entre números)
    gaps_por_sorteio = []
    for i, row in df_num.iterrows():
        nums = sorted([int(n) for n in row.dropna()])
        gaps = [nums[j+1] - nums[j] for j in range(len(nums)-1)]
        gaps_grandes = [g for g in gaps if g > 3]
        gaps_por_sorteio.append(len(gaps_grandes))
    
    padroes['gaps_por_sorteio'] = gaps_por_sorteio
    
    # 4. Distribuição de números por faixas
    faixas = {
        '1-5': 0, '6-10': 0, '11-15': 0, '16-20': 0, '21-25': 0
    }
    for i, row in df_num.iterrows():
        nums = [int(n) for n in row.dropna()]
        for num in nums:
            if 1 <= num <= 5:
                faixas['1-5'] += 1
            elif 6 <= num <= 10:
                faixas['6-10'] += 1
            elif 11 <= num <= 15:
                faixas['11-15'] += 1
            elif 16 <= num <= 20:
                faixas['16-20'] += 1
            elif 21 <= num <= 25:
                faixas['21-25'] += 1
    
    padroes['distribuicao_faixas'] = faixas
    
    return padroes

# ================================
# 🎯 ANÁLISE DE PADRÕES DE GRUPOS
# ================================
def analisar_padroes_grupos(df_num):
    """Analisa padrões de grupos: clusters, proximidade, co-ocorrências"""
    padroes = {}
    
    # 1. Matriz de co-ocorrência completa
    co_matrix = np.zeros((26, 26))
    for i, row in df_num.iterrows():
        nums = [int(n) for n in row.dropna()]
        for pair in itertools.combinations(nums, 2):
            co_matrix[pair[0], pair[1]] += 1
            co_matrix[pair[1], pair[0]] += 1
    
    padroes['co_matrix'] = co_matrix
    
    # 2. Grupos frequentes (trios, quartetos)
    grupos_trios = defaultdict(int)
    grupos_quartetos = defaultdict(int)
    
    for i, row in df_num.iterrows():
        nums = sorted([int(n) for n in row.dropna()])
        # Trios consecutivos ou próximos
        for trio in itertools.combinations(nums, 3):
            if max(trio) - min(trio) <= 5:  # Números próximos
                grupos_trios[trio] += 1
        
        # Quartetos próximos
        for quarteto in itertools.combinations(nums, 4):
            if max(quarteto) - min(quarteto) <= 8:
                grupos_quartetos[quarteto] += 1
    
    padroes['grupos_trios_frequentes'] = dict(Counter(grupos_trios).most_common(20))
    padroes['grupos_quartetos_frequentes'] = dict(Counter(grupos_quartetos).most_common(20))
    
    # 3. Números que raramente aparecem juntos
    anti_coocorrencias = []
    total_sorteios = len(df_num)
    for i in range(1, 26):
        for j in range(i+1, 26):
            coocorrencias = 0
            for _, row in df_num.iterrows():
                nums = [int(n) for n in row.dropna()]
                if i in nums and j in nums:
                    coocorrencias += 1
            
            if coocorrencias < total_sorteios * 0.1:  # Menos de 10% das vezes
                anti_coocorrencias.append((i, j, coocorrencias))
    
    padroes['anti_coocorrencias'] = sorted(anti_coocorrencias, key=lambda x: x[2])
    
    return padroes

# ================================
# 🔄 ANÁLISE DE PADRÕES DE REPETIÇÃO
# ================================
def analisar_padroes_repeticao(df_num):
    """Analisa padrões de repetição: retorno após X sorteios"""
    padroes = {}
    
    # 1. Probabilidade de retorno após N sorteios
    retorno_apos = {}
    for num in range(1, 26):
        ocorrencias = []
        for i, row in df_num.iterrows():
            ocorrencias.append(1 if num in row.values else 0)
        
        retorno_apos[num] = {}
        for n_sorteios in range(1, 11):  # Retorno após 1 a 10 sorteios
            retornos = 0
            total_oportunidades = 0
            for i in range(len(ocorrencias) - n_sorteios):
                if ocorrencias[i] == 1:  # Número apareceu
                    total_oportunidades += 1
                    if ocorrencias[i + n_sorteios] == 1:  # Retornou após N sorteios
                        retornos += 1
            
            if total_oportunidades > 0:
                retorno_apos[num][n_sorteios] = retornos / total_oportunidades
    
    padroes['retorno_apos'] = retorno_apos
    
    # 2. Sequências de ausência (quanto tempo sem aparecer)
    ausencia_atual = {}
    for num in range(1, 26):
        ultima_aparicao = -1
        for i in range(len(df_num)-1, -1, -1):
            if num in df_num.iloc[i].values:
                ultima_aparicao = i
                break
        
        ausencia_atual[num] = len(df_num) - 1 - ultima_aparicao
    
    padroes['ausencia_atual'] = ausencia_atual
    
    # 3. Padrão de alternância (aparece, não aparece, aparece)
    alternancias = {}
    for num in range(1, 26):
        ocorrencias = []
        for i, row in df_num.iterrows():
            ocorrencias.append(1 if num in row.values else 0)
        
        alternancias_contadas = 0
        for i in range(len(ocorrencias) - 2):
            if ocorrencias[i] == 1 and ocorrencias[i+1] == 0 and ocorrencias[i+2] == 1:
                alternancias_contadas += 1
        
        alternancias[num] = alternancias_contadas / max(len(ocorrencias) - 2, 1)
    
    padroes['alternancias'] = alternancias
    
    return padroes

# ================================
# 📊 EXECUTAR TODAS AS ANÁLISES
# ================================
st.title("🎯 Sistema Avançado de Análise de Padrões - Lotofácil")

with st.spinner("Analisando padrões temporais..."):
    padroes_temporais = analisar_padroes_temporais(df_numeros)

with st.spinner("Analisando padrões sequenciais..."):
    padroes_sequenciais = analisar_padroes_sequenciais(df_numeros)

with st.spinner("Analisando padrões de grupos..."):
    padroes_grupos = analisar_padroes_grupos(df_numeros)

with st.spinner("Analisando padrões de repetição..."):
    padroes_repeticao = analisar_padroes_repeticao(df_numeros)

# ================================
# ✨ FEATURE ENGINEERING AVANÇADO
# ================================
def criar_features_avancadas(df_num, padroes_temporais, padroes_sequenciais, 
                             padroes_grupos, padroes_repeticao):
    """Cria features avançadas baseadas em todos os padrões identificados"""
    features = pd.DataFrame(index=df_num.index)
    
    # Features temporais
    for janela in [5, 10, 20, 50]:
        if f'freq_janela_{janela}' in padroes_temporais:
            for num in range(1, 26):
                if num in padroes_temporais[f'freq_janela_{janela}']:
                    col_name = f'temp_freq_{janela}_{num}'
                    if len(df_num) >= janela:
                        valores = []
                        for i in range(len(df_num)):
                            if i < janela:
                                valores.append(0)
                            else:
                                ultimos = df_num.iloc[max(0, i-janela):i]
                                freq = ultimos.apply(lambda row: num in row.values, axis=1).sum()
                                valores.append(freq)
                        features[col_name] = valores
                    else:
                        features[col_name] = 0
    
    # Features de ciclos
    if 'ciclos' in padroes_temporais:
        for num in range(1, 26):
            if num in padroes_temporais['ciclos']:
                ciclo = padroes_temporais['ciclos'][num]
                col_name = f'ciclo_proximo_{num}'
                valores = []
                for i in range(len(df_num)):
                    distancia_proximo = abs(i - ciclo['proximo_esperado'])
                    valores.append(distancia_proximo)
                features[col_name] = valores
    
    # Features de tendência
    if 'tendencias' in padroes_temporais:
        for num in range(1, 26):
            if num in padroes_temporais['tendencias']:
                features[f'tendencia_{num}'] = padroes_temporais['tendencias'][num]
    
    # Features sequenciais
    if 'intervalo_medio' in padroes_sequenciais:
        valores_intervalo = []
        for i, row in df_num.iterrows():
            nums = sorted([int(n) for n in row.dropna()])
            intervalos = [nums[j+1] - nums[j] for j in range(len(nums)-1)]
            valores_intervalo.append(np.mean(intervalos) if intervalos else 0)
        features['intervalo_medio'] = valores_intervalo
    
    if 'gaps_por_sorteio' in padroes_sequenciais:
        features['gaps_grandes'] = padroes_sequenciais['gaps_por_sorteio']
    
    # Features de co-ocorrência
    if 'co_matrix' in padroes_grupos:
        co_matrix = padroes_grupos['co_matrix']
        for num in range(1, 26):
            col_name = f'cooc_media_{num}'
            valores = []
            for i, row in df_num.iterrows():
                if i == 0:
                    valores.append(0)
                else:
                    nums_anterior = [int(n) for n in df_num.iloc[i-1].dropna()]
                    if nums_anterior:
                        cooc_media = np.mean([co_matrix[num, n] for n in nums_anterior])
                    else:
                        cooc_media = 0
                    valores.append(cooc_media)
            features[col_name] = valores
    
    # Features de repetição
    if 'ausencia_atual' in padroes_repeticao:
        for num in range(1, 26):
            if num in padroes_repeticao['ausencia_atual']:
                ausencia = padroes_repeticao['ausencia_atual'][num]
                col_name = f'ausencia_{num}'
                valores = []
                for i in range(len(df_num)):
                    # Calcular ausência até o ponto i
                    ultima_aparicao = -1
                    for j in range(i, -1, -1):
                        if num in df_num.iloc[j].values:
                            ultima_aparicao = j
                            break
                    valores.append(i - ultima_aparicao if ultima_aparicao >= 0 else i + 1)
                features[col_name] = valores
    
    # Features estatísticas básicas (já existentes, mas melhoradas)
    features['qtd_pares'] = df_num.apply(lambda row: sum(int(n) % 2 == 0 for n in row.dropna()), axis=1)
    features['qtd_impares'] = df_num.apply(lambda row: sum(int(n) % 2 == 1 for n in row.dropna()), axis=1)
    features['soma_total'] = df_num.apply(lambda row: sum(int(n) for n in row.dropna()), axis=1)
    features['media_numeros'] = df_num.apply(lambda row: np.mean([int(n) for n in row.dropna()]) if len(row.dropna()) > 0 else 0, axis=1)
    
    return features.fillna(0)

# Criar features avançadas
with st.spinner("Criando features avançadas..."):
    X_avancado = criar_features_avancadas(
        df_numeros, padroes_temporais, padroes_sequenciais, 
        padroes_grupos, padroes_repeticao
    )

st.success(f"✅ Features criadas: {X_avancado.shape[1]} features para {X_avancado.shape[0]} sorteios")

# ================================
# 🤖 TREINAMENTO DE MODELOS
# ================================
st.subheader("🤖 Treinamento de Modelos Avançados")

# Validação temporal
tscv = TimeSeriesSplit(n_splits=5)

# Modelos
xgb_avancado = MultiOutputClassifier(
    XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
)

rf_avancado = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
)

gb_avancado = MultiOutputClassifier(
    GradientBoostingClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        random_state=42
    )
)

with st.spinner("Treinando modelos..."):
    xgb_avancado.fit(X_avancado, y_binario)
    rf_avancado.fit(X_avancado, y_binario)
    gb_avancado.fit(X_avancado, y_binario)

# Avaliação
y_pred_xgb = xgb_avancado.predict(X_avancado)
y_pred_rf = rf_avancado.predict(X_avancado)
y_pred_gb = gb_avancado.predict(X_avancado)

hamming_xgb = hamming_loss(y_binario, y_pred_xgb)
hamming_rf = hamming_loss(y_binario, y_pred_rf)
hamming_gb = hamming_loss(y_binario, y_pred_gb)

st.write(f"**Hamming Loss XGBoost:** {hamming_xgb:.4f}")
st.write(f"**Hamming Loss RandomForest:** {hamming_rf:.4f}")
st.write(f"**Hamming Loss GradientBoosting:** {hamming_gb:.4f}")

# Modelo final (melhor)
modelo_final = xgb_avancado if hamming_xgb <= min(hamming_rf, hamming_gb) else (rf_avancado if hamming_rf <= hamming_gb else gb_avancado)
joblib.dump(modelo_final, "modelo_avancado.pkl")

# ================================
# 🎲 GERADOR INTELIGENTE DE 15 NÚMEROS
# ================================
def gerar_15_numeros_inteligentes(modelo, X_features, padroes_temporais, 
                                  padroes_sequenciais, padroes_grupos, 
                                  padroes_repeticao, n_candidatos=500):
    """Gera 15 números mais prováveis usando múltiplos padrões"""
    
    # 1. Probabilidades do modelo
    ultima_entrada = X_features.values[-1].reshape(1, -1)
    probs_modelo = np.zeros(25)
    
    for j, clf in enumerate(modelo.estimators_):
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(ultima_entrada)
            if proba.shape[1] > 1:
                probs_modelo[j] = proba[0, 1]
    
    probs_modelo = probs_modelo / (probs_modelo.sum() + 1e-10)
    
    # 2. Ajustes baseados em padrões temporais
    ajustes_temporais = np.ones(25)
    if 'ciclos' in padroes_temporais:
        for num in range(1, 26):
            if num in padroes_temporais['ciclos']:
                ciclo = padroes_temporais['ciclos'][num]
                distancia = abs(len(df_numeros) - ciclo['proximo_esperado'])
                # Aumenta probabilidade se está próximo do ciclo esperado
                if distancia < ciclo['std']:
                    ajustes_temporais[num-1] *= 1.5
    
    if 'ausencia_atual' in padroes_repeticao:
        for num in range(1, 26):
            if num in padroes_repeticao['ausencia_atual']:
                ausencia = padroes_repeticao['ausencia_atual'][num]
                # Aumenta probabilidade se ausência está alta (lei dos grandes números)
                if ausencia > 5:
                    ajustes_temporais[num-1] *= 1.3
    
    # 3. Ajustes baseados em co-ocorrências
    ajustes_cooc = np.ones(25)
    if 'co_matrix' in padroes_grupos:
        co_matrix = padroes_grupos['co_matrix']
        ultimos_nums = [int(n) for n in df_numeros.iloc[-1].dropna()]
        for num in range(1, 26):
            if ultimos_nums:
                cooc_media = np.mean([co_matrix[num, n] for n in ultimos_nums])
                ajustes_cooc[num-1] *= (1 + cooc_media * 0.5)
    
    # 4. Combinar todas as probabilidades
    probs_final = probs_modelo * ajustes_temporais * ajustes_cooc
    probs_final = probs_final / (probs_final.sum() + 1e-10)
    
    # 5. Gerar candidatos e avaliar
    candidatos = []
    co_matrix = padroes_grupos.get('co_matrix', np.zeros((26, 26)))
    
    for _ in range(n_candidatos):
        # Seleciona números baseado em probabilidades ajustadas
        numeros = np.random.choice(np.arange(1, 26), size=15, replace=False, p=probs_final)
        numeros = sorted([int(n) for n in numeros])
        
        # Score baseado em múltiplos fatores
        score = 0
        
        # Score de probabilidade do modelo
        score += np.sum([probs_modelo[n-1] for n in numeros]) * 10
        
        # Score de co-ocorrência
        for pair in itertools.combinations(numeros, 2):
            score += co_matrix[pair[0], pair[1]] * 5
        
        # Score de distribuição (evitar concentração)
        faixas_count = {'1-5': 0, '6-10': 0, '11-15': 0, '16-20': 0, '21-25': 0}
        for num in numeros:
            if 1 <= num <= 5:
                faixas_count['1-5'] += 1
            elif 6 <= num <= 10:
                faixas_count['6-10'] += 1
            elif 11 <= num <= 15:
                faixas_count['11-15'] += 1
            elif 16 <= num <= 20:
                faixas_count['16-20'] += 1
            elif 21 <= num <= 25:
                faixas_count['21-25'] += 1
        
        # Penalizar distribuições muito desbalanceadas
        max_faixa = max(faixas_count.values())
        min_faixa = min(faixas_count.values())
        if max_faixa - min_faixa > 5:
            score *= 0.9
        
        # Score de intervalos (evitar muitos consecutivos)
        nums_sorted = sorted(numeros)
        consecutivos = sum(1 for i in range(len(nums_sorted)-1) if nums_sorted[i+1] - nums_sorted[i] == 1)
        if consecutivos > 5:
            score *= 0.95
        
        candidatos.append((numeros, score))
    
    # Ordenar por score e retornar melhor
    candidatos.sort(key=lambda x: x[1], reverse=True)
    melhor = candidatos[0][0]
    
    return melhor, probs_final, candidatos[:10]

# Gerar previsão
with st.spinner("Gerando 15 números mais prováveis..."):
    melhor_previsao, probs_final, top_candidatos = gerar_15_numeros_inteligentes(
        modelo_final, X_avancado, padroes_temporais, padroes_sequenciais,
        padroes_grupos, padroes_repeticao, n_candidatos=1000
    )

# ================================
# 📊 VISUALIZAÇÕES E RESULTADOS
# ================================
st.subheader("🎯 15 Números Mais Prováveis")
st.write(f"**Previsão Principal:** {melhor_previsao}")
st.write(f"**Score:** {top_candidatos[0][1]:.2f}")

st.subheader("📈 Top 10 Candidatos Gerados")
for i, (cand, score) in enumerate(top_candidatos, 1):
    st.write(f"**{i}.** {cand} (Score: {score:.2f})")

st.subheader("📊 Probabilidades Finais Ajustadas")
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(1, 26), probs_final)
ax.set_xticks(range(1, 26))
ax.set_xlabel("Número")
ax.set_ylabel("Probabilidade Ajustada")
ax.set_title("Probabilidades Finais (Modelo + Padrões)")
plt.xticks(rotation=45)
st.pyplot(fig)

# Ranking dos números
ranking_df = pd.DataFrame({
    'Número': range(1, 26),
    'Probabilidade': probs_final
}).sort_values('Probabilidade', ascending=False)

st.subheader("🏆 Ranking Completo de Probabilidades")
st.dataframe(ranking_df.style.format({'Probabilidade': '{:.4f}'}))

# Análise de padrões encontrados
st.subheader("🔍 Análise de Padrões Encontrados")

col1, col2 = st.columns(2)

with col1:
    st.write("**Padrões Temporais:**")
    if 'ciclos' in padroes_temporais:
        st.write(f"- {len(padroes_temporais['ciclos'])} números com padrões cíclicos identificados")
    if 'tendencias' in padroes_temporais:
        aumentando = sum(1 for v in padroes_temporais['tendencias'].values() if v > 0)
        st.write(f"- {aumentando} números com tendência de aumento")

with col2:
    st.write("**Padrões de Repetição:**")
    if 'ausencia_atual' in padroes_repeticao:
        ausencias_altas = sum(1 for v in padroes_repeticao['ausencia_atual'].values() if v > 5)
        st.write(f"- {ausencias_altas} números com ausência alta (>5 sorteios)")

st.success("✅ Análise completa! Use os 15 números gerados como base para suas apostas.")


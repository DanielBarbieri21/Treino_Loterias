"""
🚀 Script de Execução Completa com Melhorias - Lotofácil
=========================================================
Executa backtesting + geração melhorada de números
"""

import pandas as pd
import numpy as np
from validacao_backtesting import (
    backtest_modelo, 
    visualizar_resultados_backtesting,
    gerar_relatorio_backtesting
)
from gerador_inteligente_melhorado import GeradorInteligenteLotofacil
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 SISTEMA MELHORADO DE ANÁLISE E PREVISÃO - LOTOFÁCIL")
print("="*80)
print()

# ================================
# 1. CARREGAR DADOS
# ================================
print("📂 Carregando dados históricos...")
try:
    df = pd.read_excel("treino.xlsx")
    df_numeros = df.applymap(
        lambda x: int(x) if str(x).isdigit() and 1 <= int(x) <= 25 else np.nan
    ).dropna(how="all").reset_index(drop=True)
    
    print(f"✅ {len(df_numeros)} sorteios carregados")
    print()
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    print("Verifique se o arquivo treino.xlsx existe na pasta Lotofacil/")
    exit(1)

# ================================
# 2. BACKTESTING
# ================================
print("="*80)
print("📊 FASE 1: BACKTESTING - Validação em Dados Reais")
print("="*80)
print()

executar_backtesting = input("Deseja executar backtesting? (s/n): ").lower()

if executar_backtesting == 's':
    n_testes = int(input("Quantos sorteios testar? (recomendado: 30-50): ") or "30")
    
    print(f"\n🔍 Iniciando backtesting com {n_testes} sorteios...")
    print("Isso pode levar alguns minutos...\n")
    
    resultados = backtest_modelo(
        df_numeros, 
        criar_features_func=None,
        gerar_numeros_func=None,
        n_testes=n_testes,
        janela_treino=100
    )
    
    print("\n" + "="*80)
    print("📊 RESULTADOS DO BACKTESTING")
    print("="*80)
    
    stats = resultados['estatisticas']
    print(f"\n📈 Média de Acertos: {stats['media_acertos']:.2f} números por sorteio")
    print(f"📈 Taxa de Acerto: {(stats['media_acertos']/15*100):.1f}%")
    print(f"🎯 Aleatório Esperado: {stats['acerto_aleatorio_esperado']:.2f} números (60%)")
    print(f"📊 Melhoria sobre Aleatório: {stats['melhoria_sobre_aleatorio']:.1f}%\n")
    
    print(f"🏆 Máximo de Acertos: {stats['max_acertos']} números")
    print(f"📉 Mínimo de Acertos: {stats['min_acertos']} números")
    print(f"📊 Desvio Padrão: {stats['std_acertos']:.2f}\n")
    
    # Visualizar resultados
    visualizar_resultados_backtesting(resultados, salvar=True)
    
    # Gerar relatório
    gerar_relatorio_backtesting(resultados, salvar_txt=True)
    
    print("\n✅ Backtesting concluído!")
    print()

# ================================
# 3. TREINAR MODELO FINAL
# ================================
print("="*80)
print("🤖 FASE 2: TREINAMENTO DO MODELO FINAL")
print("="*80)
print()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

# Criar y_binario
y_binario = pd.DataFrame(0, index=np.arange(len(df_numeros)), columns=range(1, 26))
for i, linha in enumerate(df_numeros.values):
    for n in linha:
        if not pd.isna(n):
            y_binario.at[i, int(n)] = 1

# Criar features
print("🔧 Criando features...")
X = pd.DataFrame(index=df_numeros.index)

# Features básicas
X['qtd_pares'] = df_numeros.apply(lambda row: sum(int(n) % 2 == 0 for n in row.dropna()), axis=1)
X['qtd_impares'] = df_numeros.apply(lambda row: sum(int(n) % 2 == 1 for n in row.dropna()), axis=1)
X['soma_total'] = df_numeros.apply(lambda row: sum(int(n) for n in row.dropna()), axis=1)
X['media'] = df_numeros.apply(lambda row: np.mean([int(n) for n in row.dropna()]), axis=1)
X['std'] = df_numeros.apply(lambda row: np.std([int(n) for n in row.dropna()]), axis=1)

# Frequência recente
for num in range(1, 26):
    freq_recente = []
    for i in range(len(df_numeros)):
        janela_inicio = max(0, i - 10)
        ultimos = df_numeros.iloc[janela_inicio:i]
        if len(ultimos) > 0:
            freq = ultimos.apply(lambda row: num in row.values, axis=1).sum()
        else:
            freq = 0
        freq_recente.append(freq)
    X[f'freq_10_{num}'] = freq_recente

X = X.fillna(0)

print(f"✅ {X.shape[1]} features criadas")
print()

# Treinar múltiplos modelos
print("🤖 Treinando modelos...")

modelo_xgb = MultiOutputClassifier(
    XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
)

modelo_rf = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
)

modelo_gb = MultiOutputClassifier(
    GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
)

print("  • Treinando XGBoost...")
modelo_xgb.fit(X, y_binario)

print("  • Treinando Random Forest...")
modelo_rf.fit(X, y_binario)

print("  • Treinando Gradient Boosting...")
modelo_gb.fit(X, y_binario)

print("✅ Modelos treinados!")
print()

# ================================
# 4. GERAR PREVISÕES INTELIGENTES
# ================================
print("="*80)
print("🎯 FASE 3: GERAÇÃO DE PREVISÕES INTELIGENTES")
print("="*80)
print()

gerador = GeradorInteligenteLotofacil()

print("🎲 Gerando previsões usando múltiplas estratégias...")
print()

ultima_entrada = X.iloc[-1:].values
modelos_ensemble = [modelo_xgb, modelo_rf, modelo_gb]

resultados_estrategias, melhor_estrategia = gerador.gerar_previsoes_multiplas(
    modelo_xgb, ultima_entrada, df_numeros, modelos_ensemble
)

print("="*80)
print("📊 RESULTADOS DAS ESTRATÉGIAS")
print("="*80)
print()

for nome, info in resultados_estrategias.items():
    print(f"🎯 {info['nome']}")
    print(f"   Números: {info['numeros']}")
    print(f"   Score:   {info['score']:.4f}")
    print()

print("="*80)
print("🏆 MELHOR ESTRATÉGIA")
print("="*80)
print()

melhor_nome, melhor_info = melhor_estrategia
print(f"✨ {melhor_info['nome']}")
print(f"📊 Score: {melhor_info['score']:.4f}")
print()
print("🎯 NÚMEROS RECOMENDADOS:")
print()
print(f"   {melhor_info['numeros']}")
print()

# Análise dos números recomendados
numeros_rec = melhor_info['numeros']
pares = sum(1 for n in numeros_rec if n % 2 == 0)
impares = 15 - pares
soma = sum(numeros_rec)

print("📈 ANÁLISE DA COMBINAÇÃO:")
print(f"   • Pares: {pares} | Ímpares: {impares}")
print(f"   • Soma Total: {soma}")
print(f"   • Média: {soma/15:.1f}")
print(f"   • Menor: {min(numeros_rec)} | Maior: {max(numeros_rec)}")
print()

# Distribuição por faixas
faixas = {'1-5': 0, '6-10': 0, '11-15': 0, '16-20': 0, '21-25': 0}
for num in numeros_rec:
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

print("📊 DISTRIBUIÇÃO POR FAIXAS:")
for faixa, qtd in faixas.items():
    barra = '█' * qtd
    print(f"   {faixa}: {qtd:2d} {barra}")
print()

# ================================
# 5. OPÇÕES ALTERNATIVAS
# ================================
print("="*80)
print("🔄 OPÇÕES ALTERNATIVAS (Top 10)")
print("="*80)
print()

if 'candidatos' in melhor_info:
    for i, (nums, score) in enumerate(melhor_info['candidatos'][:10], 1):
        print(f"{i:2d}. {nums} (Score: {score:.2f})")
print()

# ================================
# 6. SALVAR RESULTADOS
# ================================
print("="*80)
print("💾 SALVANDO RESULTADOS")
print("="*80)
print()

# Salvar em arquivo
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

with open(f"previsao_lotofacil_{timestamp}.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("🎯 PREVISÃO LOTOFÁCIL - SISTEMA MELHORADO\n")
    f.write("="*80 + "\n\n")
    f.write(f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("NÚMEROS RECOMENDADOS:\n")
    f.write(f"{melhor_info['numeros']}\n\n")
    f.write(f"Estratégia: {melhor_info['nome']}\n")
    f.write(f"Score: {melhor_info['score']:.4f}\n\n")
    f.write("ANÁLISE:\n")
    f.write(f"Pares: {pares} | Ímpares: {impares}\n")
    f.write(f"Soma: {soma} | Média: {soma/15:.1f}\n\n")
    f.write("DISTRIBUIÇÃO:\n")
    for faixa, qtd in faixas.items():
        f.write(f"{faixa}: {qtd}\n")
    f.write("\n")
    
    if 'candidatos' in melhor_info:
        f.write("TOP 10 ALTERNATIVAS:\n")
        for i, (nums, score) in enumerate(melhor_info['candidatos'][:10], 1):
            f.write(f"{i:2d}. {nums} (Score: {score:.2f})\n")

print(f"✅ Previsão salva em: previsao_lotofacil_{timestamp}.txt")
print()

print("="*80)
print("✅ PROCESSO CONCLUÍDO!")
print("="*80)
print()
print("📌 PRÓXIMOS PASSOS:")
print("   1. Revise os resultados do backtesting")
print("   2. Use os números recomendados ou escolha uma alternativa")
print("   3. Teste e compare com resultados reais")
print("   4. Ajuste os pesos se necessário")
print()
print("⚠️  LEMBRETE: Loterias são jogos de azar. Use com responsabilidade!")
print()

"""
🔬 Análise Profunda de Padrões - Mega-Sena
============================================
Análise estatística avançada para identificar padrões ocultos e falhas
Mega-Sena: 6 números de 1 a 60
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, kstest
from collections import Counter, defaultdict
import itertools

NUM_MIN = 1
NUM_MAX = 60
NUM_SORTEADOS = 6

def analise_estatistica_distribuicao(df_numeros):
    """Análise estatística da distribuição dos números"""
    resultados = {}
    
    # Contar frequência de cada número
    frequencias = {}
    for num in range(NUM_MIN, NUM_MAX + 1):
        count = 0
        for _, row in df_numeros.iterrows():
            if num in row.values:
                count += 1
        frequencias[num] = count
    
    resultados['frequencias'] = frequencias
    
    # Teste de uniformidade (chi-quadrado)
    valores_observados = list(frequencias.values())
    soma_observados = sum(valores_observados)
    # Calcular valores esperados (uniforme) e normalizar para soma igual
    valor_esperado_por_numero = soma_observados / NUM_MAX
    valores_esperados = [valor_esperado_por_numero] * NUM_MAX
    
    # Garantir que a soma seja exatamente igual (ajuste de precisão)
    soma_esperados = sum(valores_esperados)
    if abs(soma_esperados - soma_observados) > 1e-10:
        # Ajustar proporcionalmente
        fator = soma_observados / soma_esperados if soma_esperados > 0 else 1
        valores_esperados = [v * fator for v in valores_esperados]
    
    try:
        chi2, p_value = stats.chisquare(valores_observados, valores_esperados)
    except ValueError as e:
        # Se ainda houver erro, usar ddof=0 para ajustar graus de liberdade
        chi2, p_value = stats.chisquare(valores_observados, valores_esperados, ddof=0)
    
    resultados['teste_uniformidade'] = {
        'chi2': chi2,
        'p_value': p_value,
        'uniforme': p_value > 0.05
    }
    
    # Identificar números com frequência anormal
    media_freq = np.mean(valores_observados)
    std_freq = np.std(valores_observados)
    numeros_anormais = []
    
    for num, freq in frequencias.items():
        z_score = (freq - media_freq) / std_freq if std_freq > 0 else 0
        if abs(z_score) > 2:  # Mais de 2 desvios padrão
            numeros_anormais.append({
                'numero': num,
                'frequencia': freq,
                'z_score': z_score,
                'tipo': 'alto' if z_score > 0 else 'baixo'
            })
    
    resultados['numeros_anormais'] = numeros_anormais
    
    return resultados

def analise_padroes_posicionais(df_numeros):
    """Analisa padrões relacionados à posição dos números"""
    resultados = {}
    
    # Distribuição por faixas (1-60 dividido em 6 faixas)
    faixas = {
        '1-10': [],
        '11-20': [],
        '21-30': [],
        '31-40': [],
        '41-50': [],
        '51-60': []
    }
    
    for _, row in df_numeros.iterrows():
        nums = sorted([int(n) for n in row.dropna()])
        for num in nums:
            if 1 <= num <= 10:
                faixas['1-10'].append(num)
            elif 11 <= num <= 20:
                faixas['11-20'].append(num)
            elif 21 <= num <= 30:
                faixas['21-30'].append(num)
            elif 31 <= num <= 40:
                faixas['31-40'].append(num)
            elif 41 <= num <= 50:
                faixas['41-50'].append(num)
            elif 51 <= num <= 60:
                faixas['51-60'].append(num)
    
    resultados['distribuicao_faixas'] = {k: len(v) for k, v in faixas.items()}
    
    # Padrão de início/fim (números menores vs maiores)
    inicio_fim = {
        'inicio_baixo': 0,
        'fim_alto': 0,
        'misto': 0
    }
    
    for _, row in df_numeros.iterrows():
        nums = sorted([int(n) for n in row.dropna()])
        primeiros_3 = nums[:3]
        ultimos_3 = nums[-3:]
        
        baixos_inicio = sum(1 for n in primeiros_3 if n <= 30)
        altos_fim = sum(1 for n in ultimos_3 if n > 30)
        
        if baixos_inicio >= 2 and altos_fim >= 2:
            inicio_fim['misto'] += 1
        elif baixos_inicio >= 2:
            inicio_fim['inicio_baixo'] += 1
        elif altos_fim >= 2:
            inicio_fim['fim_alto'] += 1
    
    resultados['padrao_inicio_fim'] = inicio_fim
    
    return resultados

def analise_correlacoes_avancadas(df_numeros):
    """Análise de correlações avançadas entre números"""
    resultados = {}
    
    # Matriz de correlação de aparecimento conjunto
    matriz_correlacao = np.zeros((NUM_MAX, NUM_MAX))
    
    for _, row in df_numeros.iterrows():
        nums = [int(n) for n in row.dropna()]
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                matriz_correlacao[nums[i]-1, nums[j]-1] += 1
                matriz_correlacao[nums[j]-1, nums[i]-1] += 1
    
    resultados['matriz_correlacao'] = matriz_correlacao
    
    # Pares mais e menos frequentes
    pares_freq = []
    for i in range(NUM_MAX):
        for j in range(i+1, NUM_MAX):
            pares_freq.append(((i+1, j+1), matriz_correlacao[i, j]))
    
    pares_freq.sort(key=lambda x: x[1], reverse=True)
    resultados['pares_mais_frequentes'] = pares_freq[:30]
    resultados['pares_menos_frequentes'] = pares_freq[-30:]
    
    # Trios frequentes
    trios_freq = defaultdict(int)
    for _, row in df_numeros.iterrows():
        nums = sorted([int(n) for n in row.dropna()])
        for trio in itertools.combinations(nums, 3):
            trios_freq[trio] += 1
    
    resultados['trios_mais_frequentes'] = Counter(trios_freq).most_common(30)
    
    return resultados

def analise_padroes_temporais_avancados(df_numeros):
    """Análise avançada de padrões temporais"""
    resultados = {}
    
    # Sequências de aparecimento/ausência
    sequencias = {}
    for num in range(NUM_MIN, NUM_MAX + 1):
        sequencia = []
        for _, row in df_numeros.iterrows():
            sequencia.append(1 if num in row.values else 0)
        sequencias[num] = sequencia
        
        # Analisar padrões na sequência
        alternancias = 0
        for i in range(len(sequencia) - 1):
            if sequencia[i] != sequencia[i+1]:
                alternancias += 1
        
        # Sequências de ausência longas
        ausencia_max = 0
        ausencia_atual = 0
        for val in sequencia:
            if val == 0:
                ausencia_atual += 1
                ausencia_max = max(ausencia_max, ausencia_atual)
            else:
                ausencia_atual = 0
        
        resultados[num] = {
            'alternancias': alternancias,
            'ausencia_maxima': ausencia_max,
            'taxa_alternancia': alternancias / max(len(sequencia) - 1, 1)
        }
    
    resultados['sequencias'] = sequencias
    
    # Padrões cíclicos (usando autocorrelação)
    ciclos_identificados = {}
    for num in range(NUM_MIN, NUM_MAX + 1):
        sequencia = sequencias[num]
        if len(sequencia) > 20:
            # Calcular autocorrelação para diferentes lags
            autocorrs = []
            for lag in range(1, min(20, len(sequencia)//2)):
                corr = np.corrcoef(sequencia[:-lag], sequencia[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append((lag, corr))
            
            if autocorrs:
                melhor_lag = max(autocorrs, key=lambda x: abs(x[1]))
                if abs(melhor_lag[1]) > 0.3:  # Correlação significativa
                    ciclos_identificados[num] = {
                        'periodo': melhor_lag[0],
                        'correlacao': melhor_lag[1]
                    }
    
    resultados['ciclos_identificados'] = ciclos_identificados
    
    return resultados

def identificar_falhas_padroes(df_numeros):
    """Identifica possíveis falhas ou padrões exploráveis"""
    falhas = []
    
    try:
        if df_numeros is None or len(df_numeros) == 0:
            return falhas
        
        # 1. Números com frequência muito abaixo do esperado
        frequencias = {}
        for num in range(NUM_MIN, NUM_MAX + 1):
            count = 0
            for _, row in df_numeros.iterrows():
                nums_row = [int(n) for n in row.dropna() if not pd.isna(n)]
                if num in nums_row:
                    count += 1
            frequencias[num] = count
        
        if len(df_numeros) > 0:
            media_esperada = len(df_numeros) * (NUM_SORTEADOS / NUM_MAX)
            for num, freq in frequencias.items():
                if freq < media_esperada * 0.8:  # 20% abaixo do esperado
                    falhas.append({
                        'tipo': 'frequencia_baixa',
                        'numero': num,
                        'frequencia': freq,
                        'esperado': media_esperada,
                        'diferenca_percentual': (1 - freq/media_esperada) * 100 if media_esperada > 0 else 0
                    })
        
        # 2. Padrões de ausência muito longos
        for num in range(NUM_MIN, NUM_MAX + 1):
            ausencia_atual = 0
            for i in range(len(df_numeros)-1, -1, -1):
                nums_row = [int(n) for n in df_numeros.iloc[i].dropna() if not pd.isna(n)]
                if num in nums_row:
                    break
                ausencia_atual += 1
            
            if ausencia_atual > 15:  # Mais de 15 sorteios sem aparecer
                falhas.append({
                    'tipo': 'ausencia_prolongada',
                    'numero': num,
                    'ausencia': ausencia_atual,
                    'probabilidade_retorno': min(ausencia_atual * 0.05, 0.9)
                })
        
        # 3. Pares que nunca ou raramente aparecem juntos
        coocorrencias = defaultdict(int)
        total_sorteios = len(df_numeros)
        
        for _, row in df_numeros.iterrows():
            nums = [int(n) for n in row.dropna() if not pd.isna(n)]
            for pair in itertools.combinations(nums, 2):
                coocorrencias[tuple(sorted(pair))] += 1
        
        if total_sorteios > 0:
            esperado_minimo = max(total_sorteios * 0.02, 1)  # Esperado pelo menos 2% das vezes, mínimo 1
            for pair, count in coocorrencias.items():
                if count < esperado_minimo:
                    falhas.append({
                        'tipo': 'par_rarissimo',
                        'par': pair,
                        'ocorrencias': count,
                        'esperado_minimo': esperado_minimo
                    })
    
    except Exception as e:
        print(f"Erro ao identificar falhas: {e}")
        return falhas
    
    return falhas

def gerar_relatorio_completo(df_numeros):
    """Gera relatório completo de análise"""
    print("=" * 60)
    print("🔬 RELATÓRIO DE ANÁLISE PROFUNDA - MEGA-SENA")
    print("=" * 60)
    
    # 1. Análise estatística
    print("\n📊 1. ANÁLISE ESTATÍSTICA")
    print("-" * 60)
    stats_result = analise_estatistica_distribuicao(df_numeros)
    
    print(f"\nTeste de Uniformidade:")
    print(f"  Chi-quadrado: {stats_result['teste_uniformidade']['chi2']:.2f}")
    print(f"  P-value: {stats_result['teste_uniformidade']['p_value']:.4f}")
    print(f"  Distribuição uniforme: {stats_result['teste_uniformidade']['uniforme']}")
    
    if stats_result['numeros_anormais']:
        print(f"\n⚠️ Números com frequência anormal:")
        for item in stats_result['numeros_anormais']:
            print(f"  Número {item['numero']}: {item['frequencia']} ocorrências "
                  f"(Z-score: {item['z_score']:.2f}, Tipo: {item['tipo']})")
    
    # 2. Padrões posicionais
    print("\n📍 2. PADRÕES POSICIONAIS")
    print("-" * 60)
    pos_result = analise_padroes_posicionais(df_numeros)
    print(f"\nDistribuição por faixas:")
    for faixa, count in pos_result['distribuicao_faixas'].items():
        print(f"  {faixa}: {count} ocorrências")
    
    print(f"\nPadrão início/fim:")
    for padrao, count in pos_result['padrao_inicio_fim'].items():
        print(f"  {padrao}: {count} sorteios")
    
    # 3. Correlações
    print("\n🔗 3. CORRELAÇÕES AVANÇADAS")
    print("-" * 60)
    corr_result = analise_correlacoes_avancadas(df_numeros)
    print(f"\nTop 10 pares mais frequentes:")
    for i, ((n1, n2), freq) in enumerate(corr_result['pares_mais_frequentes'][:10], 1):
        print(f"  {i}. Par ({n1}, {n2}): {freq:.0f} ocorrências")
    
    print(f"\nTop 5 trios mais frequentes:")
    for i, (trio, freq) in enumerate(corr_result['trios_mais_frequentes'][:5], 1):
        print(f"  {i}. Trio {trio}: {freq} ocorrências")
    
    # 4. Padrões temporais
    print("\n⏰ 4. PADRÕES TEMPORAIS AVANÇADOS")
    print("-" * 60)
    temp_result = analise_padroes_temporais_avancados(df_numeros)
    if temp_result['ciclos_identificados']:
        print(f"\nNúmeros com padrões cíclicos identificados:")
        for num, info in temp_result['ciclos_identificados'].items():
            print(f"  Número {num}: Período {info['periodo']} sorteios "
                  f"(Correlação: {info['correlacao']:.3f})")
    
    # 5. Falhas e padrões exploráveis
    print("\n🎯 5. FALHAS E PADRÕES EXPLORÁVEIS")
    print("-" * 60)
    falhas = identificar_falhas_padroes(df_numeros)
    
    falhas_freq = [f for f in falhas if f['tipo'] == 'frequencia_baixa']
    falhas_ausencia = [f for f in falhas if f['tipo'] == 'ausencia_prolongada']
    
    if falhas_freq:
        print(f"\n⚠️ Números com frequência muito baixa ({len(falhas_freq)}):")
        for f in falhas_freq[:10]:
            print(f"  Número {f['numero']}: {f['frequencia']:.0f} ocorrências "
                  f"(Esperado: {f['esperado']:.0f}, Diferença: {f['diferenca_percentual']:.1f}%)")
    
    if falhas_ausencia:
        print(f"\n⏳ Números com ausência prolongada ({len(falhas_ausencia)}):")
        for f in falhas_ausencia[:10]:
            print(f"  Número {f['numero']}: {f['ausencia']} sorteios sem aparecer "
                  f"(Probabilidade retorno: {f['probabilidade_retorno']*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ Análise completa!")
    print("=" * 60)
    
    return {
        'estatistica': stats_result,
        'posicional': pos_result,
        'correlacoes': corr_result,
        'temporal': temp_result,
        'falhas': falhas
    }

if __name__ == "__main__":
    # Carregar dados
    try:
        df = pd.read_excel("mega_sena.xlsx")
        df_numeros = df.applymap(
            lambda x: int(x) if str(x).isdigit() and NUM_MIN <= int(x) <= NUM_MAX else np.nan
        ).dropna(how="all")
        
        # Gerar relatório
        relatorio = gerar_relatorio_completo(df_numeros)
        
    except Exception as e:
        print(f"❌ Erro: {e}")


"""
🎯 Sistema de Backtesting e Validação Real - Mega-Sena
========================================================
Avalia o desempenho real do modelo em sorteios anteriores
Mega-Sena: 6 números de 1 a 60
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

NUM_MIN = 1
NUM_MAX = 60
NUM_SORTEADOS = 6

def backtest_modelo(df_numeros, criar_features_func, gerar_numeros_func, 
                    n_testes=50, janela_treino=100):
    """
    Realiza backtesting real do modelo
    
    Args:
        df_numeros: DataFrame com histórico de sorteios
        criar_features_func: Função para criar features
        gerar_numeros_func: Função para gerar previsão
        n_testes: Número de sorteios para testar
        janela_treino: Tamanho da janela de treino
    
    Returns:
        dict com resultados do backtesting
    """
    print("🔍 Iniciando Backtesting Real - Mega-Sena...")
    print(f"Testando nos últimos {n_testes} sorteios")
    print(f"Usando janela de treino de {janela_treino} sorteios\n")
    
    resultados = {
        'acertos_por_sorteio': [],
        'previsoes': [],
        'sorteios_reais': [],
        'timestamps': [],
        'acertos_detalhados': []
    }
    
    # Começar do ponto onde temos dados suficientes
    inicio = max(janela_treino, len(df_numeros) - n_testes)
    
    for i in range(inicio, len(df_numeros)):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Dados de treino: todos os sorteios até i-1
        dados_treino = df_numeros.iloc[:i]
        
        # Sorteio real a ser previsto
        sorteio_real = set([int(n) for n in df_numeros.iloc[i].dropna()])
        
        try:
            # Gerar previsão usando apenas dados de treino
            previsao = gerar_previsao_para_backtesting(dados_treino)
            
            # Calcular acertos
            acertos = len(set(previsao) & sorteio_real)
            
            # Armazenar resultados
            resultados['acertos_por_sorteio'].append(acertos)
            resultados['previsoes'].append(previsao)
            resultados['sorteios_reais'].append(sorted(list(sorteio_real)))
            resultados['timestamps'].append(timestamp)
            resultados['acertos_detalhados'].append({
                'sorteio_index': i,
                'previsao': previsao,
                'real': sorted(list(sorteio_real)),
                'acertos': acertos,
                'numeros_acertados': sorted(list(set(previsao) & sorteio_real))
            })
            
            print(f"Sorteio {i - inicio + 1}/{n_testes}: {acertos} acertos")
            
        except Exception as e:
            print(f"Erro no sorteio {i}: {e}")
            continue
    
    # Calcular estatísticas
    resultados['estatisticas'] = calcular_estatisticas_backtesting(resultados)
    
    return resultados

def gerar_previsao_para_backtesting(dados_treino):
    """Gera previsão usando apenas dados de treino (sem leakage)"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from xgboost import XGBClassifier
    
    # Criar y_binario
    y_binario = pd.DataFrame(0, index=np.arange(len(dados_treino)), 
                             columns=range(NUM_MIN, NUM_MAX + 1))
    for i, linha in enumerate(dados_treino.values):
        for n in linha:
            if not pd.isna(n):
                y_binario.at[i, int(n)] = 1
    
    # Criar features básicas (versão simplificada para backtesting)
    X = pd.DataFrame(index=dados_treino.index)
    
    # Features básicas
    X['qtd_pares'] = dados_treino.apply(lambda row: sum(int(n) % 2 == 0 for n in row.dropna()), axis=1)
    X['qtd_impares'] = dados_treino.apply(lambda row: sum(int(n) % 2 == 1 for n in row.dropna()), axis=1)
    X['soma_total'] = dados_treino.apply(lambda row: sum(int(n) for n in row.dropna()), axis=1)
    X['media'] = dados_treino.apply(lambda row: np.mean([int(n) for n in row.dropna()]), axis=1)
    X['std'] = dados_treino.apply(lambda row: np.std([int(n) for n in row.dropna()]), axis=1)
    X['min'] = dados_treino.apply(lambda row: min([int(n) for n in row.dropna()]), axis=1)
    X['max'] = dados_treino.apply(lambda row: max([int(n) for n in row.dropna()]), axis=1)
    X['range'] = X['max'] - X['min']
    
    # Frequência recente (últimos 10 sorteios) - apenas para números mais frequentes
    numeros_frequentes = list(range(1, 61, 3))  # Amostra para reduzir features
    for num in numeros_frequentes:
        freq_recente = []
        for i in range(len(dados_treino)):
            janela_inicio = max(0, i - 10)
            ultimos = dados_treino.iloc[janela_inicio:i]
            if len(ultimos) > 0:
                freq = ultimos.apply(lambda row: num in row.values, axis=1).sum()
            else:
                freq = 0
            freq_recente.append(freq)
        X[f'freq_10_{num}'] = freq_recente
    
    X = X.fillna(0)
    
    # Treinar modelo rápido
    modelo = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    )
    
    modelo.fit(X, y_binario)
    
    # Gerar previsão: usar probabilidades e pegar top 6
    ultima_entrada = X.iloc[-1:].values
    
    # Coletar probabilidades de cada classificador
    probs = np.zeros(NUM_MAX)
    for j, clf in enumerate(modelo.estimators_):
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(ultima_entrada.reshape(1, -1))
            if proba.shape[1] > 1:
                probs[j] = proba[0, 1]
    
    # Pegar top 6 números com maior probabilidade
    top_6_indices = np.argsort(probs)[-NUM_SORTEADOS:]
    previsao = sorted([int(idx + 1) for idx in top_6_indices])
    
    return previsao

def calcular_estatisticas_backtesting(resultados):
    """Calcula estatísticas do backtesting"""
    acertos = resultados['acertos_por_sorteio']
    
    stats = {
        'total_testes': len(acertos),
        'media_acertos': np.mean(acertos),
        'mediana_acertos': np.median(acertos),
        'max_acertos': max(acertos) if acertos else 0,
        'min_acertos': min(acertos) if acertos else 0,
        'std_acertos': np.std(acertos),
        'distribuicao_acertos': {},
        'taxa_acerto_por_numero': {},
        'acerto_aleatorio_esperado': 0.6,  # (6/60) * 6 = 0.6
        'melhoria_sobre_aleatorio': 0,
        'senas': 0,
        'quinas': 0,
        'quadras': 0
    }
    
    # Distribuição de acertos
    for n_acertos in range(NUM_SORTEADOS + 1):
        count = acertos.count(n_acertos)
        stats['distribuicao_acertos'][n_acertos] = count
        
        # Contar prêmios
        if n_acertos == 6:
            stats['senas'] += count
        elif n_acertos == 5:
            stats['quinas'] += count
        elif n_acertos == 4:
            stats['quadras'] += count
    
    # Calcular melhoria sobre aleatório
    if stats['media_acertos'] > 0:
        melhoria = ((stats['media_acertos'] - stats['acerto_aleatorio_esperado']) / 
                   stats['acerto_aleatorio_esperado']) * 100
        stats['melhoria_sobre_aleatorio'] = melhoria
    
    # Taxa de acerto por número
    for num in range(NUM_MIN, NUM_MAX + 1):
        total_aparicoes_reais = 0
        total_acertos_numero = 0
        
        for detalhe in resultados['acertos_detalhados']:
            if num in detalhe['real']:
                total_aparicoes_reais += 1
                if num in detalhe['previsao']:
                    total_acertos_numero += 1
        
        if total_aparicoes_reais > 0:
            taxa = (total_acertos_numero / total_aparicoes_reais) * 100
            stats['taxa_acerto_por_numero'][num] = {
                'taxa': taxa,
                'acertos': total_acertos_numero,
                'aparicoes': total_aparicoes_reais
            }
    
    return stats

def visualizar_resultados_backtesting(resultados, salvar=True):
    """Cria visualizações dos resultados"""
    stats = resultados['estatisticas']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribuição de acertos
    ax1 = axes[0, 0]
    acertos_valores = list(stats['distribuicao_acertos'].keys())
    acertos_frequencias = list(stats['distribuicao_acertos'].values())
    colors = ['gold' if x == 6 else 'silver' if x == 5 else 'orange' if x == 4 else 'lightblue' 
              for x in acertos_valores]
    ax1.bar(acertos_valores, acertos_frequencias, color=colors, edgecolor='black')
    ax1.axvline(stats['media_acertos'], color='red', linestyle='--', 
                label=f'Média: {stats["media_acertos"]:.2f}')
    ax1.axvline(stats['acerto_aleatorio_esperado'], color='green', linestyle='--', 
                label=f'Aleatório: {stats["acerto_aleatorio_esperado"]:.2f}')
    ax1.set_xlabel('Número de Acertos')
    ax1.set_ylabel('Frequência')
    ax1.set_title('Distribuição de Acertos por Sorteio\n(Ouro=Sena, Prata=Quina, Laranja=Quadra)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Evolução temporal dos acertos
    ax2 = axes[0, 1]
    ax2.plot(resultados['acertos_por_sorteio'], marker='o', markersize=4, linewidth=1)
    ax2.axhline(stats['media_acertos'], color='red', linestyle='--', 
                label=f'Média: {stats["media_acertos"]:.2f}')
    ax2.axhline(stats['acerto_aleatorio_esperado'], color='green', linestyle='--', 
                label=f'Aleatório: {stats["acerto_aleatorio_esperado"]:.2f}')
    ax2.set_xlabel('Sorteio')
    ax2.set_ylabel('Número de Acertos')
    ax2.set_title('Evolução Temporal dos Acertos')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-0.5, 6.5)
    
    # 3. Taxa de acerto por número (top 30 números)
    ax3 = axes[1, 0]
    if stats['taxa_acerto_por_numero']:
        numeros_ordenados = sorted(stats['taxa_acerto_por_numero'].items(), 
                                   key=lambda x: x[1]['taxa'], reverse=True)[:30]
        numeros = [n for n, _ in numeros_ordenados]
        taxas = [info['taxa'] for _, info in numeros_ordenados]
        colors = ['green' if t > 15 else 'orange' if t > 10 else 'red' for t in taxas]
        ax3.bar(range(len(numeros)), taxas, color=colors, edgecolor='black', alpha=0.7)
        ax3.set_xticks(range(len(numeros)))
        ax3.set_xticklabels(numeros, rotation=45)
        ax3.axhline(10, color='blue', linestyle='--', label='Esperado (10%)')
        ax3.set_xlabel('Número')
        ax3.set_ylabel('Taxa de Acerto (%)')
        ax3.set_title('Taxa de Acerto - Top 30 Números')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Estatísticas resumidas
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    resumo = f"""
    📊 ESTATÍSTICAS DO BACKTESTING - MEGA-SENA
    
    Total de Testes: {stats['total_testes']}
    
    Acertos:
    • Média: {stats['media_acertos']:.2f} números
    • Mediana: {stats['mediana_acertos']:.2f} números
    • Máximo: {stats['max_acertos']} números
    • Mínimo: {stats['min_acertos']} números
    
    Prêmios:
    🥇 Senas (6 acertos): {stats['senas']}
    🥈 Quinas (5 acertos): {stats['quinas']}
    🥉 Quadras (4 acertos): {stats['quadras']}
    
    Comparação:
    • Aleatório esperado: {stats['acerto_aleatorio_esperado']:.2f}
    • Melhoria: {stats['melhoria_sobre_aleatorio']:.1f}%
    
    Taxa de Acerto Geral: {(stats['media_acertos'] / 6 * 100):.1f}%
    """
    
    ax4.text(0.1, 0.5, resumo, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    
    if salvar:
        plt.savefig('backtesting_resultados_megasena.png', dpi=300, bbox_inches='tight')
        print("📊 Gráfico salvo: backtesting_resultados_megasena.png")
    
    plt.show()
    
    return fig

def gerar_relatorio_backtesting(resultados, salvar_txt=True):
    """Gera relatório detalhado em texto"""
    stats = resultados['estatisticas']
    
    relatorio = f"""
{'='*80}
🎯 RELATÓRIO DE BACKTESTING - MEGA-SENA
{'='*80}

DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
📊 ESTATÍSTICAS GERAIS
{'='*80}

Total de Sorteios Testados: {stats['total_testes']}

Acertos por Sorteio:
  • Média:   {stats['media_acertos']:.2f} números
  • Mediana: {stats['mediana_acertos']:.2f} números
  • Máximo:  {stats['max_acertos']} números
  • Mínimo:  {stats['min_acertos']} números

Prêmios Obtidos:
  🥇 Senas (6 acertos):   {stats['senas']}
  🥈 Quinas (5 acertos):  {stats['quinas']}
  🥉 Quadras (4 acertos): {stats['quadras']}

Comparação com Aleatório:
  • Acerto Aleatório Esperado: {stats['acerto_aleatorio_esperado']:.2f} números
  • Melhoria sobre Aleatório:  {stats['melhoria_sobre_aleatorio']:.1f}%

Taxa de Acerto Geral: {(stats['media_acertos'] / 6 * 100):.1f}%

{'='*80}
📈 DISTRIBUIÇÃO DE ACERTOS
{'='*80}

"""
    
    for n_acertos in sorted(stats['distribuicao_acertos'].keys()):
        freq = stats['distribuicao_acertos'][n_acertos]
        if freq > 0:
            percent = (freq / stats['total_testes']) * 100
            barra = '█' * int(percent)
            emoji = '🥇' if n_acertos == 6 else '🥈' if n_acertos == 5 else '🥉' if n_acertos == 4 else '  '
            relatorio += f"{emoji} {n_acertos} acertos: {freq:3d} vezes ({percent:5.1f}%) {barra}\n"
    
    relatorio += f"""
{'='*80}
🎯 TOP 15 NÚMEROS COM MELHOR TAXA DE ACERTO
{'='*80}

"""
    
    if stats['taxa_acerto_por_numero']:
        numeros_ordenados = sorted(
            stats['taxa_acerto_por_numero'].items(),
            key=lambda x: x[1]['taxa'],
            reverse=True
        )
        
        for i, (num, info) in enumerate(numeros_ordenados[:15], 1):
            relatorio += f"{i:2d}. Número {num:2d}: {info['taxa']:5.1f}% "
            relatorio += f"({info['acertos']}/{info['aparicoes']} aparições)\n"
    
    relatorio += f"""
{'='*80}
🔍 ANÁLISE DETALHADA (Últimos 10 Sorteios)
{'='*80}

"""
    
    for detalhe in resultados['acertos_detalhados'][-10:]:
        emoji = '🥇' if detalhe['acertos'] == 6 else '🥈' if detalhe['acertos'] == 5 else '🥉' if detalhe['acertos'] == 4 else '  '
        relatorio += f"{emoji} Sorteio #{detalhe['sorteio_index']}:\n"
        relatorio += f"  Real:     {detalhe['real']}\n"
        relatorio += f"  Previsto: {detalhe['previsao']}\n"
        relatorio += f"  Acertos:  {detalhe['acertos']} números {detalhe['numeros_acertados']}\n"
        relatorio += "\n"
    
    relatorio += f"""
{'='*80}
✅ CONCLUSÃO
{'='*80}

"""
    
    if stats['senas'] > 0:
        relatorio += f"🎉 EXCELENTE! Acertou {stats['senas']} SENA(S)!\n"
    elif stats['quinas'] > 0:
        relatorio += f"🎯 MUITO BOM! Acertou {stats['quinas']} QUINA(S)!\n"
    elif stats['quadras'] > 0:
        relatorio += f"👍 BOM! Acertou {stats['quadras']} QUADRA(S)!\n"
    
    if stats['melhoria_sobre_aleatorio'] > 10:
        relatorio += "\n✅ POSITIVO: O modelo apresenta melhoria significativa sobre o aleatório.\n"
    elif stats['melhoria_sobre_aleatorio'] > 0:
        relatorio += "\n⚠️  MODERADO: O modelo apresenta leve melhoria sobre o aleatório.\n"
    else:
        relatorio += "\n❌ NEGATIVO: O modelo NÃO apresenta melhoria sobre o aleatório.\n"
    
    relatorio += f"\nMédia de Acertos: {stats['media_acertos']:.2f}/6 ({(stats['media_acertos']/6*100):.1f}%)\n"
    relatorio += f"Aleatório Esperado: {stats['acerto_aleatorio_esperado']:.2f}/6 (10%)\n"
    
    relatorio += f"\n{'='*80}\n"
    
    if salvar_txt:
        filename = f"relatorio_backtesting_megasena_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8')as f:
            f.write(relatorio)
        print(f"📄 Relatório salvo: {filename}")
    
    print(relatorio)
    return relatorio

# Função principal para executar
if __name__ == "__main__":
    print("Para usar este módulo, importe as funções:")
    print("from validacao_backtesting import backtest_modelo, visualizar_resultados_backtesting")

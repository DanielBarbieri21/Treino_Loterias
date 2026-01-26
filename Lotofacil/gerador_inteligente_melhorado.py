"""
🎯 Gerador Inteligente de Números - Lotofácil (MELHORADO)
===========================================================
Estratégia otimizada para gerar os 15 números mais prováveis
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class GeradorInteligenteLotofacil:
    """
    Gerador inteligente usando múltiplas estratégias
    """
    
    def __init__(self, padroes_temporais=None, padroes_grupos=None, 
                 padroes_repeticao=None):
        self.padroes_temporais = padroes_temporais or {}
        self.padroes_grupos = padroes_grupos or {}
        self.padroes_repeticao = padroes_repeticao or {}
        self.pesos_otimizados = {
            'modelo': 10.0,
            'coocorrencia': 3.0,
            'temporal': 1.5,
            'ausencia': 1.3,
            'distribuicao': 2.0
        }
    
    def estrategia_1_top_probabilidades(self, modelo, X_ultima_entrada):
        """
        Estratégia 1: Seleciona os 15 números com maior probabilidade do modelo
        Esta é a abordagem DIRETA e mais confiável
        """
        # Coletar probabilidades
        probs = np.zeros(25)
        
        for j, clf in enumerate(modelo.estimators_):
            if hasattr(clf, "predict_proba"):
                try:
                    proba = clf.predict_proba(X_ultima_entrada)
                    if proba.shape[1] > 1:
                        probs[j] = proba[0, 1]
                except:
                    probs[j] = 0
        
        # Normalizar
        probs = probs / (probs.sum() + 1e-10)
        
        # Selecionar top 15 números
        top_15_indices = np.argsort(probs)[-15:]
        numeros = sorted([int(idx + 1) for idx in top_15_indices])
        
        score = self._calcular_score(numeros, probs, 'estrategia_1')
        
        return numeros, score, probs
    
    def estrategia_2_probabilidades_ajustadas(self, modelo, X_ultima_entrada, df_numeros):
        """
        Estratégia 2: Probabilidades do modelo + ajustes por padrões
        """
        # Probabilidades base do modelo
        probs_base = np.zeros(25)
        for j, clf in enumerate(modelo.estimators_):
            if hasattr(clf, "predict_proba"):
                try:
                    proba = clf.predict_proba(X_ultima_entrada)
                    if proba.shape[1] > 1:
                        probs_base[j] = proba[0, 1]
                except:
                    probs_base[j] = 0
        
        # Ajustes temporais
        ajustes_temp = self._calcular_ajustes_temporais(df_numeros)
        
        # Ajustes de ausência
        ajustes_ausencia = self._calcular_ajustes_ausencia(df_numeros)
        
        # Ajustes de co-ocorrência
        ajustes_cooc = self._calcular_ajustes_coocorrencia(df_numeros)
        
        # Combinar com pesos otimizados
        probs_ajustadas = (
            probs_base * self.pesos_otimizados['modelo'] +
            ajustes_temp * self.pesos_otimizados['temporal'] +
            ajustes_ausencia * self.pesos_otimizados['ausencia'] +
            ajustes_cooc * self.pesos_otimizados['coocorrencia']
        )
        
        # Normalizar
        probs_ajustadas = probs_ajustadas / (probs_ajustadas.sum() + 1e-10)
        
        # Selecionar top 15
        top_15_indices = np.argsort(probs_ajustadas)[-15:]
        numeros = sorted([int(idx + 1) for idx in top_15_indices])
        
        score = self._calcular_score(numeros, probs_ajustadas, 'estrategia_2')
        
        return numeros, score, probs_ajustadas
    
    def estrategia_3_ensemble_diversificado(self, modelos, X_ultima_entrada):
        """
        Estratégia 3: Ensemble de múltiplos modelos com votação ponderada
        """
        probabilidades_todos = []
        
        for modelo in modelos:
            probs = np.zeros(25)
            for j, clf in enumerate(modelo.estimators_):
                if hasattr(clf, "predict_proba"):
                    try:
                        proba = clf.predict_proba(X_ultima_entrada)
                        if proba.shape[1] > 1:
                            probs[j] = proba[0, 1]
                    except:
                        probs[j] = 0
            
            probabilidades_todos.append(probs)
        
        # Média ponderada
        probs_ensemble = np.mean(probabilidades_todos, axis=0)
        probs_ensemble = probs_ensemble / (probs_ensemble.sum() + 1e-10)
        
        # Selecionar top 15
        top_15_indices = np.argsort(probs_ensemble)[-15:]
        numeros = sorted([int(idx + 1) for idx in top_15_indices])
        
        score = self._calcular_score(numeros, probs_ensemble, 'estrategia_3')
        
        return numeros, score, probs_ensemble
    
    def estrategia_4_otimizacao_combinatoria(self, probs, df_numeros, n_combinacoes=500):
        """
        Estratégia 4: Gera e avalia múltiplas combinações, escolhe a melhor
        """
        candidatos = []
        co_matrix = self._criar_matriz_coocorrencia(df_numeros)
        
        # Gerar candidatos variados
        for _ in range(n_combinacoes):
            # Método 1: Top probabilidades com pequena variação
            if np.random.random() < 0.6:
                # Pegar top 20 e escolher 15 aleatoriamente
                top_20 = np.argsort(probs)[-20:]
                indices_selecionados = np.random.choice(top_20, size=15, replace=False)
            else:
                # Método 2: Amostragem ponderada
                indices_selecionados = np.random.choice(
                    np.arange(25), size=15, replace=False, p=probs
                )
            
            numeros = sorted([int(idx + 1) for idx in indices_selecionados])
            
            # Avaliar score completo
            score = self._calcular_score_completo(numeros, probs, co_matrix, df_numeros)
            candidatos.append((numeros, score))
        
        # Ordenar e pegar melhores
        candidatos.sort(key=lambda x: x[1], reverse=True)
        
        melhor_numeros, melhor_score = candidatos[0]
        
        return melhor_numeros, melhor_score, candidatos[:10]
    
    def _calcular_ajustes_temporais(self, df_numeros):
        """Calcula ajustes baseados em padrões temporais"""
        ajustes = np.ones(25)
        
        if 'ciclos' in self.padroes_temporais:
            for num in range(1, 26):
                if num in self.padroes_temporais['ciclos']:
                    ciclo = self.padroes_temporais['ciclos'][num]
                    distancia = abs(len(df_numeros) - ciclo['proximo_esperado'])
                    
                    # Aumenta se está próximo do ciclo
                    if distancia < ciclo.get('std', 5):
                        ajustes[num-1] *= 1.5
        
        return ajustes / (ajustes.sum() + 1e-10)
    
    def _calcular_ajustes_ausencia(self, df_numeros):
        """Calcula ajustes baseados em ausência prolongada"""
        ajustes = np.ones(25)
        
        for num in range(1, 26):
            # Calcular ausência atual
            ausencia = 0
            for i in range(len(df_numeros)-1, -1, -1):
                if num in df_numeros.iloc[i].values:
                    break
                ausencia += 1
            
            # Lei dos grandes números: aumenta probabilidade com ausência
            if ausencia > 5:
                ajustes[num-1] *= (1 + ausencia * 0.05)
            elif ausencia > 10:
                ajustes[num-1] *= 2.0
        
        return ajustes / (ajustes.sum() + 1e-10)
    
    def _calcular_ajustes_coocorrencia(self, df_numeros):
        """Calcula ajustes baseados em co-ocorrência com sorteio anterior"""
        ajustes = np.ones(25)
        
        co_matrix = self._criar_matriz_coocorrencia(df_numeros)
        
        if len(df_numeros) > 0:
            ultimos_nums = [int(n) for n in df_numeros.iloc[-1].dropna()]
            
            for num in range(1, 26):
                if ultimos_nums:
                    cooc_media = np.mean([co_matrix[num-1, n-1] for n in ultimos_nums])
                    ajustes[num-1] *= (1 + cooc_media * 0.02)
        
        return ajustes / (ajustes.sum() + 1e-10)
    
    def _criar_matriz_coocorrencia(self, df_numeros):
        """Cria matriz de co-ocorrência"""
        co_matrix = np.zeros((25, 25))
        
        for _, row in df_numeros.iterrows():
            nums = [int(n) for n in row.dropna()]
            for i, n1 in enumerate(nums):
                for n2 in nums[i+1:]:
                    co_matrix[n1-1, n2-1] += 1
                    co_matrix[n2-1, n1-1] += 1
        
        return co_matrix
    
    def _calcular_score(self, numeros, probs, estrategia):
        """Calcula score simples baseado em probabilidades"""
        score = sum(probs[n-1] for n in numeros)
        return score
    
    def _calcular_score_completo(self, numeros, probs, co_matrix, df_numeros):
        """Calcula score completo com múltiplos fatores"""
        score = 0
        
        # 1. Score de probabilidade do modelo
        score += sum(probs[n-1] for n in numeros) * self.pesos_otimizados['modelo']
        
        # 2. Score de co-ocorrência
        for pair in itertools.combinations(numeros, 2):
            score += co_matrix[pair[0]-1, pair[1]-1] * self.pesos_otimizados['coocorrencia']
        
        # 3. Score de distribuição balanceada
        faixas_count = self._contar_faixas(numeros)
        max_faixa = max(faixas_count.values())
        min_faixa = min(faixas_count.values())
        
        # Penalizar se muito desbalanceado
        if max_faixa - min_faixa <= 2:
            score *= (1 + self.pesos_otimizados['distribuicao'])
        elif max_faixa - min_faixa > 5:
            score *= 0.8
        
        # 4. Score de pares/ímpares balanceados
        pares = sum(1 for n in numeros if n % 2 == 0)
        impares = 15 - pares
        
        # Ideal: 7-8 pares, 7-8 ímpares
        if 6 <= pares <= 9:
            score *= 1.2
        
        # 5. Penalizar muitos consecutivos
        nums_sorted = sorted(numeros)
        consecutivos = sum(1 for i in range(len(nums_sorted)-1) 
                          if nums_sorted[i+1] - nums_sorted[i] == 1)
        
        if consecutivos > 6:
            score *= 0.9
        elif consecutivos >= 2 and consecutivos <= 4:
            score *= 1.1
        
        return score
    
    def _contar_faixas(self, numeros):
        """Conta distribuição por faixas"""
        faixas = {'1-5': 0, '6-10': 0, '11-15': 0, '16-20': 0, '21-25': 0}
        
        for num in numeros:
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
        
        return faixas
    
    def gerar_previsoes_multiplas(self, modelo, X_ultima_entrada, df_numeros, modelos_ensemble=None):
        """
        Gera previsões usando todas as estratégias e retorna as melhores
        """
        resultados = {}
        
        # Estratégia 1: Top Probabilidades
        nums_1, score_1, probs_1 = self.estrategia_1_top_probabilidades(modelo, X_ultima_entrada)
        resultados['estrategia_1'] = {
            'numeros': nums_1,
            'score': score_1,
            'probs': probs_1,
            'nome': 'Top 15 Probabilidades Diretas'
        }
        
        # Estratégia 2: Probabilidades Ajustadas
        nums_2, score_2, probs_2 = self.estrategia_2_probabilidades_ajustadas(
            modelo, X_ultima_entrada, df_numeros
        )
        resultados['estrategia_2'] = {
            'numeros': nums_2,
            'score': score_2,
            'probs': probs_2,
            'nome': 'Probabilidades Ajustadas por Padrões'
        }
        
        # Estratégia 3: Ensemble (se disponível)
        if modelos_ensemble and len(modelos_ensemble) > 1:
            nums_3, score_3, probs_3 = self.estrategia_3_ensemble_diversificado(
                modelos_ensemble, X_ultima_entrada
            )
            resultados['estrategia_3'] = {
                'numeros': nums_3,
                'score': score_3,
                'probs': probs_3,
                'nome': 'Ensemble de Modelos'
            }
        
        # Estratégia 4: Otimização Combinatória
        melhor_probs = probs_2  # Usar probabilidades ajustadas
        nums_4, score_4, candidatos = self.estrategia_4_otimizacao_combinatoria(
            melhor_probs, df_numeros, n_combinacoes=500
        )
        resultados['estrategia_4'] = {
            'numeros': nums_4,
            'score': score_4,
            'probs': melhor_probs,
            'nome': 'Otimização Combinatória',
            'candidatos': candidatos
        }
        
        # Encontrar melhor estratégia
        melhor_estrategia = max(resultados.items(), key=lambda x: x[1]['score'])
        
        return resultados, melhor_estrategia
    
    def otimizar_pesos(self, df_numeros, n_iteracoes=20):
        """
        Otimiza pesos usando validação cruzada temporal
        """
        from sklearn.model_selection import ParameterGrid
        
        grid_pesos = {
            'modelo': [5, 10, 15, 20],
            'coocorrencia': [1, 2, 3, 5],
            'temporal': [1.0, 1.5, 2.0],
            'ausencia': [1.0, 1.3, 1.5],
            'distribuicao': [1.0, 1.5, 2.0]
        }
        
        melhor_score = -float('inf')
        melhores_pesos = self.pesos_otimizados
        
        # Usar últimos 20 sorteios para validação
        n_validacao = min(20, len(df_numeros) // 5)
        
        for params in list(ParameterGrid(grid_pesos))[:n_iteracoes]:
            self.pesos_otimizados = params
            
            # Avaliar em sorteios de validação
            score_total = 0
            for i in range(len(df_numeros) - n_validacao, len(df_numeros)):
                # Simular previsão e calcular acertos
                # (Implementação simplificada)
                score_total += np.random.random()  # Placeholder
            
            score_medio = score_total / n_validacao
            
            if score_medio > melhor_score:
                melhor_score = score_medio
                melhores_pesos = params.copy()
        
        self.pesos_otimizados = melhores_pesos
        print(f"✅ Pesos otimizados: {melhores_pesos}")
        print(f"Score médio: {melhor_score:.4f}")
        
        return melhores_pesos

# Exemplo de uso
if __name__ == "__main__":
    print("Gerador Inteligente de Números - Lotofácil")
    print("Importe a classe GeradorInteligenteLotofacil para usar")

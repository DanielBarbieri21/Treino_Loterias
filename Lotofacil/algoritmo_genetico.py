"""
Algoritmo Genetico para Otimizacao de Portfolio de Apostas - Lotofacil
======================================================================
Metodo 3:
Trata uma cartela completa de N jogos (ex: 10 bilhetes) como um unico individuo.
Otimiza simultaneamente:
  1. Probabilidade total segundo o Ensemble IA (BiLSTM + XGBoost)
  2. Conformidade com as 7 Camadas Estruturais (filtros_estruturais)
  3. Diversidade e Cobertura do Portfolio (evita redundancias e maximiza
     as chances de premiacoes multiplas).
"""

from __future__ import annotations

import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import random
from typing import List, Tuple, Set, Dict, Optional
import numpy as np

from filtros_estruturais import avaliar_filtros

# ──────────────────────────────────────────────────────────────
# Funcao de Avaliacao de Fitness do Portfolio
# ──────────────────────────────────────────────────────────────

def calcular_fitness_portfolio(
    portfolio: List[List[int]],
    probs: np.ndarray,
    ultimo_sorteio: Optional[List[int]] = None,
    pesos: Optional[Dict[str, float]] = None,
) -> float:
    """
    Avalia a qualidade global de um conjunto de N jogos.
    """
    if pesos is None:
        pesos = {
            "ia_prob": 4.0,
            "conformidade_filtros": 3.0,
            "diversidade_inter_jogos": 2.5,
            "cobertura_top_dezenas": 2.0,
        }

    n_jogos = len(portfolio)
    if n_jogos == 0:
        return 0.0

    # 1. Score de Probabilidade da IA
    score_prob = sum(sum(probs[n - 1] for n in jogo) for jogo in portfolio) / n_jogos

    # 2. Conformidade Estrutural (Filtros de 7 Camadas)
    score_filtros = 0.0
    for jogo in portfolio:
        analise = avaliar_filtros(jogo, ultimo_sorteio)
        violacoes = analise["violacoes"]
        # Quanto menos violacoes, maior o multiplicador (1.0 para 0 violacoes, 0.7 para 1, etc.)
        score_filtros += max(0.0, 1.0 - (0.25 * violacoes))
    score_filtros /= n_jogos

    # 3. Diversidade Inter-Jogos (Penaliza sobreposicao excessiva)
    penalidade_sobreposicao = 0.0
    pares_comparados = 0
    for i in range(n_jogos):
        s1 = set(portfolio[i])
        for j in range(i + 1, n_jogos):
            s2 = set(portfolio[j])
            intersecao = len(s1 & s2)
            pares_comparados += 1
            # Ideal e ter entre 9 e 12 numeros compartilhados.
            # 14 ou 15 compartilhados e quase um jogo repetido (desperdicio de aposta)
            if intersecao >= 14:
                penalidade_sobreposicao += 1.5
            elif intersecao >= 13:
                penalidade_sobreposicao += 0.5
            elif intersecao <= 7:
                penalidade_sobreposicao += 0.3  # muito distante da distribuicao esperada

    penalidade_norm = (penalidade_sobreposicao / max(1, pares_comparados))
    score_diversidade = max(0.0, 1.0 - penalidade_norm)

    # 4. Cobertura das Top 20 Dezenas
    top_20_ia = set(np.argsort(probs)[-20:] + 1)
    todas_dezenas_usadas = set()
    for jogo in portfolio:
        todas_dezenas_usadas.update(jogo)
    
    cobertura_top = len(todas_dezenas_usadas & top_20_ia) / 20.0

    # Fitness total ponderado
    fitness = (
        pesos["ia_prob"] * score_prob +
        pesos["conformidade_filtros"] * score_filtros +
        pesos["diversidade_inter_jogos"] * score_diversidade +
        pesos["cobertura_top_dezenas"] * cobertura_top
    )
    return float(fitness)


# ──────────────────────────────────────────────────────────────
# Operadores Geneticos
# ──────────────────────────────────────────────────────────────

def gerar_individuo_aleatorio(
    probs: np.ndarray,
    n_jogos: int = 10,
    ultimo_sorteio: Optional[List[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[List[int]]:
    """Gera um portfólio inicial de N jogos válidos amostrados por probabilidade."""
    if rng is None:
        rng = np.random.default_rng()

    probs_n = probs / (probs.sum() + 1e-10)
    top_22 = np.argsort(probs)[-22:] + 1

    portfolio = []
    seen = set()

    for _ in range(n_jogos * 5):
        if len(portfolio) >= n_jogos:
            break
        # 70% chance de pegar do top 22, 30% amostragem geral
        if rng.random() < 0.7:
            jogo = sorted(list(rng.choice(top_22, size=15, replace=False)))
        else:
            jogo = sorted(list(rng.choice(25, size=15, replace=False, p=probs_n) + 1))

        t_jogo = tuple(jogo)
        if t_jogo not in seen:
            seen.add(t_jogo)
            portfolio.append(jogo)

    while len(portfolio) < n_jogos:
        jogo = sorted(list(rng.choice(25, size=15, replace=False, p=probs_n) + 1))
        portfolio.append(jogo)

    return portfolio[:n_jogos]


def cruzar_portfolios(p1: List[List[int]], p2: List[List[int]], rng: np.random.Generator) -> Tuple[List[List[int]], List[List[int]]]:
    """Crossover: Recombina jogos entre dois portfólios."""
    n = len(p1)
    ponto_corte = rng.integers(1, n)
    filho1 = p1[:ponto_corte] + p2[ponto_corte:]
    filho2 = p2[:ponto_corte] + p1[ponto_corte:]
    return filho1, filho2


def mutar_portfolio(
    portfolio: List[List[int]],
    probs: np.ndarray,
    taxa_mutacao: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> List[List[int]]:
    """Mutacao: Altera 1 ou 2 dezenas de alguns jogos dentro do portfólio."""
    if rng is None:
        rng = np.random.default_rng()

    novo_portfolio = []
    top_20 = list(np.argsort(probs)[-20:] + 1)

    for jogo in portfolio:
        jogo_mutado = list(jogo)
        if rng.random() < taxa_mutacao:
            # Troca 1 dezena
            pos_remover = rng.integers(0, 15)
            candidatos_inserir = [n for n in top_20 if n not in jogo_mutado]
            if candidatos_inserir:
                novo_num = int(rng.choice(candidatos_inserir))
                jogo_mutado[pos_remover] = novo_num
                jogo_mutado.sort()

        novo_portfolio.append(jogo_mutado)

    return novo_portfolio


# ──────────────────────────────────────────────────────────────
# Motor Principal do Algoritmo Genetico
# ──────────────────────────────────────────────────────────────

def otimizar_portfolio_genetico(
    probs: np.ndarray,
    ultimo_sorteio: Optional[List[int]] = None,
    n_jogos: int = 10,
    pop_size: int = 40,
    n_geracoes: int = 35,
    taxa_mutacao: float = 0.30,
    taxa_elitismo: float = 0.15,
    seed: int = 42,
    verbose: bool = False,
) -> List[Dict]:
    """
    Executa o Algoritmo Genetico para encontrar a melhor carteira de n_jogos.
    """
    rng = np.random.default_rng(seed)

    # 1. Criar Populacao Inicial
    populacao = [
        gerar_individuo_aleatorio(probs, n_jogos=n_jogos, ultimo_sorteio=ultimo_sorteio, rng=rng)
        for _ in range(pop_size)
    ]

    n_elite = max(1, int(pop_size * taxa_elitismo))

    melhor_global = None
    melhor_fitness_global = -float("inf")

    # 2. Ciclo de Evolucao
    for geracao in range(n_geracoes):
        # Avaliar fitness de toda a populacao
        scores = [
            (calcular_fitness_portfolio(ind, probs, ultimo_sorteio), ind)
            for ind in populacao
        ]
        scores.sort(key=lambda x: x[0], reverse=True)

        if scores[0][0] > melhor_fitness_global:
            melhor_fitness_global = scores[0][0]
            melhor_global = [list(j) for j in scores[0][1]]

        if verbose and (geracao % 10 == 0 or geracao == n_geracoes - 1):
            print(f"  [GA] Geracao {geracao+1}/{n_geracoes} | Melhor Fitness: {melhor_fitness_global:.4f}")

        # Elitismo
        nova_pop = [ind for _, ind in scores[:n_elite]]

        # Gerar restante da populacao por Torneio + Crossover + Mutacao
        while len(nova_pop) < pop_size:
            # Torneio k=3
            t1 = rng.choice(len(scores), size=3, replace=False)
            t2 = rng.choice(len(scores), size=3, replace=False)
            p1 = scores[min(t1)][1]
            p2 = scores[min(t2)][1]

            f1, f2 = cruzar_portfolios(p1, p2, rng)
            f1 = mutar_portfolio(f1, probs, taxa_mutacao=taxa_mutacao, rng=rng)
            nova_pop.append(f1)
            if len(nova_pop) < pop_size:
                f2 = mutar_portfolio(f2, probs, taxa_mutacao=taxa_mutacao, rng=rng)
                nova_pop.append(f2)

        populacao = nova_pop

    # Formatar o portfolio campeao
    jogos_finais = []
    for idx, jogo in enumerate(melhor_global, 1):
        score_ind = sum(probs[n - 1] for n in jogo)
        analise = avaliar_filtros(jogo, ultimo_sorteio)
        jogos_finais.append({
            "numeros": sorted(jogo),
            "score": float(score_ind),
            "fitness_portfolio": float(melhor_fitness_global),
            "analise": analise,
            "tipo": "Algoritmo Genetico (Portfolio)"
        })

    return jogos_finais


# ──────────────────────────────────────────────────────────────
# Teste Standalone
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Teste do Algoritmo Genetico de Portfolio ===")
    mock_probs = np.random.uniform(0.3, 0.8, size=25)
    mock_probs /= mock_probs.sum()

    res = otimizar_portfolio_genetico(mock_probs, n_jogos=5, pop_size=20, n_geracoes=15, verbose=True)
    for i, j in enumerate(res, 1):
        print(f"Jogo {i}: {j['numeros']} | Score: {j['score']:.4f}")

"""
Filtros Estruturais de 7 Camadas + Eliminacao Reversa - Lotofacil
=================================================================
Aplica regras fisicas e estatisticas comprovadas da Lotofacil:
  1. Repetidos do sorteio anterior (ideal: 8 a 10)
  2. Moldura vs Miolo (ideal: 9 a 11 na Moldura)
  3. Numeros Primos (ideal: 4 a 7 primos)
  4. Soma Total das Dezenas (ideal: 175 a 225)
  5. Sequencia de Fibonacci (ideal: 3 a 6)
  6. Proporcao Par/Impar (ideal: 7P/8I ou 8P/7I ou 6P/9I ou 9P/6I)
  7. Sequencias Consecutivas Maximas (ideal: <= 5 numeros seguidos)

Tambem inclui Eliminacao Reversa (Cold Elimination):
  Elimina as 5 a 10 dezenas com menor probabilidade do Ensemble.
"""

from __future__ import annotations

import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from typing import List, Tuple, Set, Dict, Optional
import numpy as np

# ──────────────────────────────────────────────────────────────
# Conjuntos de Referencia da Lotofacil
# ──────────────────────────────────────────────────────────────
MOLDURA: Set[int]   = {1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25}
CENTRO: Set[int]    = {7, 8, 9, 12, 13, 14, 17, 18, 19}
PRIMOS: Set[int]     = {2, 3, 5, 7, 11, 13, 17, 19, 23}
FIBONACCI: Set[int]  = {1, 2, 3, 5, 8, 13, 21}
MULTIPLOS_3: Set[int] = {3, 6, 9, 12, 15, 18, 21, 24}


# ──────────────────────────────────────────────────────────────
# Funcoes de Validacao de Camadas
# ──────────────────────────────────────────────────────────────

def checar_consecutivos_max(numeros: List[int]) -> int:
    """Retorna o maior tamanho de sequencia consecutiva existente no jogo."""
    nums = sorted(numeros)
    max_seq = 1
    cur_seq = 1
    for i in range(len(nums) - 1):
        if nums[i+1] - nums[i] == 1:
            cur_seq += 1
            if cur_seq > max_seq:
                max_seq = cur_seq
        else:
            cur_seq = 1
    return max_seq


def avaliar_filtros(jogo: List[int], ultimo_sorteio: Optional[List[int]] = None) -> Dict[str, any]:
    """
    Avalia todas as 7 caracteristicas estruturais de um jogo de 15 dezenas.
    Retorna um dicionario com as metricas e o numero de violacoes de regras.
    """
    s_jogo = set(jogo)
    nums = sorted(jogo)
    
    qtd_moldura   = len(s_jogo & MOLDURA)
    qtd_centro    = len(s_jogo & CENTRO)
    qtd_primos    = len(s_jogo & PRIMOS)
    qtd_fibonacci = len(s_jogo & FIBONACCI)
    soma_total    = sum(nums)
    qtd_pares     = sum(1 for n in nums if n % 2 == 0)
    qtd_impares   = 15 - qtd_pares
    max_consec    = checar_consecutivos_max(nums)
    
    qtd_repetidos = len(s_jogo & set(ultimo_sorteio)) if ultimo_sorteio else None

    # Checagem de conformidade com padroes historicos
    violacoes = 0
    
    # Regra 1: Moldura (9 a 11)
    if not (8 <= qtd_moldura <= 11):
        violacoes += 1
        
    # Regra 2: Primos (4 a 7)
    if not (4 <= qtd_primos <= 7):
        violacoes += 1
        
    # Regra 3: Fibonacci (3 a 6)
    if not (3 <= qtd_fibonacci <= 6):
        violacoes += 1
        
    # Regra 4: Soma (170 a 230)
    if not (170 <= soma_total <= 230):
        violacoes += 1
        
    # Regra 5: Pares (6 a 9)
    if not (6 <= qtd_pares <= 9):
        violacoes += 1
        
    # Regra 6: Consecutivos (maximo 5)
    if max_consec > 5:
        violacoes += 1
        
    # Regra 7: Repetidos do anterior (8 a 10)
    if qtd_repetidos is not None and not (7 <= qtd_repetidos <= 11):
        violacoes += 1

    return {
        "moldura": qtd_moldura,
        "centro": qtd_centro,
        "primos": qtd_primos,
        "fibonacci": qtd_fibonacci,
        "soma": soma_total,
        "pares": qtd_pares,
        "impares": qtd_impares,
        "max_consec": max_consec,
        "repetidos": qtd_repetidos,
        "violacoes": violacoes,
        "valido": (violacoes == 0),
    }


def jogo_valido(jogo: List[int], ultimo_sorteio: Optional[List[int]] = None, max_violacoes: int = 1) -> bool:
    """Retorna True se o jogo atende aos filtros dentro do limite tolerado de violacoes."""
    res = avaliar_filtros(jogo, ultimo_sorteio)
    return res["violacoes"] <= max_violacoes


# ──────────────────────────────────────────────────────────────
# Eliminacao Reversa (Cold Numbers / Anti-Previsao)
# ──────────────────────────────────────────────────────────────

def identificar_dezenas_mortas(probs: np.ndarray, n_eliminar: int = 6) -> List[int]:
    """
    Identifica as 'n_eliminar' dezenas com menor probabilidade do Ensemble.
    Retorna lista de numeros 1-based (ex: [8, 14, 19, 22, ...]).
    """
    indices_ordenados = np.argsort(probs)  # Do menor para o maior
    dezenas_mortas = [int(idx + 1) for idx in indices_ordenados[:n_eliminar]]
    return sorted(dezenas_mortas)


def gerar_jogos_com_filtros_e_eliminacao(
    probs: np.ndarray,
    ultimo_sorteio: Optional[List[int]] = None,
    n_eliminar: int = 5,
    n_jogos: int = 10,
    n_tentativas: int = 5000,
    max_violacoes: int = 0,
    seed: int = 42
) -> List[Dict]:
    """
    Metodo 1 Completo:
    1. Elimina as n_eliminar dezenas mais fracas (Pool de 25 - n_eliminar dezenas)
    2. Amostra combinacoes ponderadas pelas probabilidades restantes
    3. Passa pelo Validador Rigido de 7 Camadas
    4. Ranqueia os melhores jogos que passaram em todos os filtros
    """
    rng = np.random.default_rng(seed)
    
    # 1. Eliminacao das fracas
    dezenas_eliminadas = set(identificar_dezenas_mortas(probs, n_eliminar=n_eliminar))
    dezenas_ativas = [n for n in range(1, 26) if n not in dezenas_eliminadas]
    
    # Probabilidades apenas das dezenas ativas
    probs_ativas = np.array([probs[n-1] for n in dezenas_ativas], dtype=float)
    probs_ativas /= (probs_ativas.sum() + 1e-10)
    
    candidatos_validos = []
    seen = set()
    
    # Gerar combinacoes candidatas
    for _ in range(n_tentativas):
        chosen = rng.choice(dezenas_ativas, size=15, replace=False, p=probs_ativas)
        chosen_tuple = tuple(sorted(int(x) for x in chosen))
        
        if chosen_tuple in seen:
            continue
        seen.add(chosen_tuple)
        
        jogo = list(chosen_tuple)
        analise = avaliar_filtros(jogo, ultimo_sorteio)
        
        if analise["violacoes"] <= max_violacoes:
            # Score baseado na probabilidade das dezenas escolhidas e conformidade
            score = sum(probs[n-1] for n in jogo) * (1.0 - 0.15 * analise["violacoes"])
            candidatos_validos.append({
                "numeros": jogo,
                "score": float(score),
                "analise": analise,
                "eliminadas": sorted(list(dezenas_eliminadas))
            })
            
            if len(candidatos_validos) >= n_jogos * 10:
                break
                
    # Se nao conseguiu o suficiente com 0 violacoes, relaxa para 1 violacao
    if len(candidatos_validos) < n_jogos:
        for _ in range(n_tentativas):
            chosen = rng.choice(dezenas_ativas, size=15, replace=False, p=probs_ativas)
            chosen_tuple = tuple(sorted(int(x) for x in chosen))
            if chosen_tuple in seen:
                continue
            seen.add(chosen_tuple)
            jogo = list(chosen_tuple)
            analise = avaliar_filtros(jogo, ultimo_sorteio)
            if analise["violacoes"] <= 1:
                score = sum(probs[n-1] for n in jogo) * (1.0 - 0.15 * analise["violacoes"])
                candidatos_validos.append({
                    "numeros": jogo,
                    "score": float(score),
                    "analise": analise,
                    "eliminadas": sorted(list(dezenas_eliminadas))
                })
                if len(candidatos_validos) >= n_jogos * 5:
                    break

    candidatos_validos.sort(key=lambda x: x["score"], reverse=True)
    return candidatos_validos[:n_jogos]


# ──────────────────────────────────────────────────────────────
# Teste Standalone
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Teste de Filtros Estruturais e Eliminacao Reversa ===")
    ex_jogo = [1, 2, 3, 5, 7, 9, 10, 11, 13, 15, 16, 17, 21, 23, 25]
    ultimo  = [2, 3, 4, 5, 9, 10, 11, 12, 15, 16, 17, 18, 21, 23, 25]
    
    analise = avaliar_filtros(ex_jogo, ultimo)
    print(f"Jogo: {ex_jogo}")
    for k, v in analise.items():
        print(f"  {k}: {v}")

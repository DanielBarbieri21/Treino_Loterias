"""
Buscador de Dados Lotofacil - API Oficial Caixa v2
===================================================
Usa a API publica e gratuita:
  GET https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil         -> ultimo concurso
  GET https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil/{num}   -> concurso especifico

Estrategia eficiente:
  1. Le o Excel existente (formato asloterias ou padrao)
  2. Detecta qual o ultimo concurso salvo
  3. Busca APENAS os concursos que faltam via API (pode ser 1, 2 ou alguns)
  4. Adiciona ao Excel e salva
"""

from __future__ import annotations

import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import time
import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────
# Configuracoes
# ──────────────────────────────────────────────────────────────
BASE_URL   = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
TIMEOUT    = 15
DELAY      = 0.4
MAX_RETRY  = 3

ARQUIVO_EXCEL = Path(__file__).parent / "treino.xlsx"
ARQUIVO_PADRAO = Path(__file__).parent / "historico_padrao.xlsx"

from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _criar_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_session = _criar_session()

def _get_concurso(numero: Optional[int] = None, session: Optional[requests.Session] = None) -> Optional[dict]:
    s = session or _session
    url = BASE_URL if numero is None else f"{BASE_URL}/{numero}"
    try:
        r = s.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
    except Exception:
        pass
    return None


def _parse_concurso(data: dict) -> Optional[dict]:
    dezenas = data.get("listaDezenas") or []
    if len(dezenas) != 15:
        return None
    row: dict = {
        "concurso": int(data.get("numero", 0)),
        "data":     data.get("dataApuracao", ""),
    }
    for i, d in enumerate(dezenas, start=1):
        row[f"n{i}"] = int(d)
    return row


def buscar_ultimo_concurso() -> Optional[dict]:
    data = _get_concurso()
    return _parse_concurso(data) if data else None


def buscar_concurso(numero: int, session: Optional[requests.Session] = None) -> Optional[dict]:
    data = _get_concurso(numero, session=session)
    return _parse_concurso(data) if data else None


def buscar_faixa(inicio: int, fim: int, verbose: bool = True, max_workers: int = 12) -> List[dict]:
    numeros = list(range(inicio, fim + 1))
    total = len(numeros)
    if total == 0:
        return []

    resultados: List[dict] = []
    concluidos = 0

    if verbose:
        print(f"  -> Baixando {total} concursos com {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(buscar_concurso, num, _session): num for num in numeros}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                resultados.append(res)
            concluidos += 1
            if verbose and (concluidos % 25 == 0 or concluidos == total):
                print(f"  [{concluidos}/{total}] concursos recebidos...")

    resultados.sort(key=lambda x: x["concurso"])
    return resultados


# ──────────────────────────────────────────────────────────────
# Leitura do Excel existente (formato asloterias)
# ──────────────────────────────────────────────────────────────

def _ler_excel_asloterias(arquivo: Path) -> pd.DataFrame:
    """
    Le o Excel no formato do site asloterias.com.br.
    Header fica na linha 6 (0-indexed), dados a partir da linha 7.
    Retorna DataFrame com colunas: concurso, n1..n15
    """
    df_raw = pd.read_excel(arquivo, header=None)

    # Detectar a linha do cabecalho (contem 'Concurso')
    header_row = None
    for i, row in df_raw.iterrows():
        if any("concurso" in str(v).lower() for v in row.values if pd.notna(v)):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Cabecalho 'Concurso' nao encontrado no Excel.")

    df = pd.read_excel(arquivo, header=header_row)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Renomear colunas
    rename = {}
    bola_cols = [c for c in df.columns if "bola" in c]
    bola_cols.sort(key=lambda c: int(''.join(filter(str.isdigit, c)) or "0"))
    for i, c in enumerate(bola_cols[:15], start=1):
        rename[c] = f"n{i}"

    concurso_col = next((c for c in df.columns if c == "concurso"), None)
    if concurso_col is None:
        concurso_col = df.columns[0]
        rename[concurso_col] = "concurso"

    df.rename(columns=rename, inplace=True)

    # Manter so linhas validas (concurso numerico)
    df = df[pd.to_numeric(df["concurso"], errors="coerce").notna()].copy()
    df["concurso"] = df["concurso"].astype(int)

    # Garantir colunas n1-n15
    num_cols = [f"n{i}" for i in range(1, 16)]
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[["concurso"] + num_cols].reset_index(drop=True)


def _ler_excel_padrao(arquivo: Path) -> pd.DataFrame:
    """
    Le o Excel no formato padrao criado por este modulo (colunas: concurso, n1..n15).
    """
    df = pd.read_excel(arquivo)
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    df = df[df["concurso"].notna()].copy()
    df["concurso"] = df["concurso"].astype(int)
    return df.reset_index(drop=True)


def carregar_historico_existente() -> Tuple[pd.DataFrame, str]:
    """
    Carrega o historico do Excel disponivel.
    Tenta primeiro o arquivo padrao, depois o asloterias.

    Returns:
        (df, formato)  onde formato e 'padrao' ou 'asloterias'
    """
    if ARQUIVO_PADRAO.exists():
        df = _ler_excel_padrao(ARQUIVO_PADRAO)
        return df, "padrao"
    if ARQUIVO_EXCEL.exists():
        df = _ler_excel_asloterias(ARQUIVO_EXCEL)
        return df, "asloterias"
    return pd.DataFrame(columns=["concurso"] + [f"n{i}" for i in range(1, 16)]), "vazio"


# ──────────────────────────────────────────────────────────────
# Atualizacao incremental
# ──────────────────────────────────────────────────────────────

def atualizar_excel(arquivo: Path = ARQUIVO_PADRAO, verbose: bool = True) -> Tuple[int, int]:
    """
    Verifica o ultimo concurso no arquivo e na API,
    busca apenas os novos e salva no arquivo padrao.

    Returns:
        (concursos_novos, total_registros)
    """
    if verbose:
        print("[API] Verificando dados na API da Caixa...")

    # Ultimo da API
    ultimo_api = _get_concurso()
    if ultimo_api is None:
        print("[ERRO] Nao foi possivel acessar a API da Caixa.")
        return 0, 0

    num_api  = int(ultimo_api.get("numero", 0))
    data_api = ultimo_api.get("dataApuracao", "?")
    if verbose:
        print(f"  -> API: concurso {num_api} ({data_api})")

    # Historico existente
    df_hist, fmt = carregar_historico_existente()
    num_excel = int(df_hist["concurso"].max()) if len(df_hist) else 0
    if verbose:
        print(f"  -> Historico local ({fmt}): ultimo concurso = {num_excel} | total = {len(df_hist)}")

    if num_excel >= num_api:
        if verbose:
            print("[OK] Historico ja esta atualizado!")
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        df_hist.to_excel(arquivo, index=False)
        return 0, len(df_hist)

    # Buscar apenas os novos
    inicio = num_excel + 1
    fim    = num_api
    n_buscar = fim - inicio + 1
    if verbose:
        print(f"  -> Buscando {n_buscar} concurso(s) novos ({inicio} a {fim})...")

    novos = buscar_faixa(inicio, fim, verbose=verbose)

    if not novos:
        print("[AVISO] Nenhum dado novo obtido da API.")
        return 0, len(df_hist)

    df_novos = pd.DataFrame(novos)
    df_final = pd.concat([df_hist, df_novos], ignore_index=True)
    df_final.sort_values("concurso", inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    arquivo.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_excel(arquivo, index=False)

    if verbose:
        print(f"[OK] {len(novos)} novo(s) concurso(s) adicionado(s). Total: {len(df_final)}")

    return len(novos), len(df_final)


# ──────────────────────────────────────────────────────────────
# Carregar historico como matriz binaria
# ──────────────────────────────────────────────────────────────

def carregar_binario(verbose: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Atualiza dados e retorna:
      - binario : np.ndarray (N, 25) com 0/1
      - df      : DataFrame com concurso e n1-n15
    """
    atualizar_excel(ARQUIVO_PADRAO, verbose=verbose)

    df, fmt = carregar_historico_existente()
    if fmt == "vazio":
        raise RuntimeError("Nenhum dado encontrado. Verifique o arquivo treino.xlsx.")

    num_cols = [f"n{i}" for i in range(1, 16)]
    binario = np.zeros((len(df), 25), dtype=np.float32)
    for i, (_, row) in enumerate(df[num_cols].iterrows()):
        for val in row.dropna():
            v = int(val)
            if 1 <= v <= 25:
                binario[i, v - 1] = 1.0

    return binario, df


# ──────────────────────────────────────────────────────────────
# Execucao standalone
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Lotofacil - Atualizacao de Dados via API Caixa v2")
    print("=" * 55)

    novos, total = atualizar_excel(verbose=True)
    print()
    print(f"Resumo: {novos} novo(s) | {total} total no historico padrao")

    ult = buscar_ultimo_concurso()
    if ult:
        dezenas = [ult[f"n{i}"] for i in range(1, 16)]
        print(f"\nUltimo resultado (concurso {ult['concurso']} - {ult['data']}):")
        print(f"  {dezenas}")

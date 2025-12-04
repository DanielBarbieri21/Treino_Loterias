"""
📡 Buscador de Dados da Mega-Sena via API
==========================================
Permite buscar dados atualizados da Mega-Sena via API pública da Caixa
"""

import pandas as pd
import requests
import json
from datetime import datetime

def buscar_dados_megasena_api(ultimos_n=None):
    """
    Busca dados da Mega-Sena via API da Caixa
    
    Args:
        ultimos_n: Número de últimos sorteios a buscar (None = todos disponíveis)
    
    Returns:
        DataFrame com os dados dos sorteios
    """
    try:
        # API pública da Caixa
        url = "https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Processar dados
        sorteios = []
        if 'listaDezenas' in data or 'dezenas' in data:
            # Formato pode variar, ajustar conforme necessário
            if isinstance(data, list):
                for item in data:
                    if 'numero' in item and 'dezenas' in item:
                        dezenas = item['dezenas']
                        if len(dezenas) == 6:
                            sorteios.append({
                                'concurso': item.get('numero', len(sorteios) + 1),
                                'data': item.get('dataApuracao', ''),
                                **{f'num_{i+1}': int(d) for i, d in enumerate(dezenas)}
                            })
            else:
                # Formato alternativo
                if 'listaDezenas' in data:
                    lista = data['listaDezenas']
                    for i, item in enumerate(lista):
                        if 'dezenas' in item:
                            dezenas = item['dezenas']
                            if len(dezenas) == 6:
                                sorteios.append({
                                    'concurso': item.get('numero', i + 1),
                                    'data': item.get('dataApuracao', ''),
                                    **{f'num_{j+1}': int(d) for j, d in enumerate(dezenas)}
                                })
        
        if not sorteios:
            # Tentar formato alternativo
            if isinstance(data, dict) and 'resultado' in data:
                resultado = data['resultado']
                if isinstance(resultado, list):
                    for item in resultado:
                        dezenas = item.get('dezenasSorteadas', [])
                        if len(dezenas) == 6:
                            sorteios.append({
                                'concurso': item.get('numero', len(sorteios) + 1),
                                'data': item.get('dataApuracao', ''),
                                **{f'num_{i+1}': int(d) for i, d in enumerate(dezenas)}
                            })
        
        if sorteios:
            df = pd.DataFrame(sorteios)
            if ultimos_n:
                df = df.head(ultimos_n)
            return df
        else:
            print("⚠️ Formato de dados não reconhecido. Retornando None.")
            print(f"Dados recebidos: {json.dumps(data, indent=2)[:500]}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao buscar dados da API: {e}")
        return None
    except Exception as e:
        print(f"❌ Erro ao processar dados: {e}")
        return None

def atualizar_excel_com_api(arquivo_excel="mega_sena.xlsx", ultimos_n=100):
    """
    Atualiza o arquivo Excel com dados da API
    
    Args:
        arquivo_excel: Caminho do arquivo Excel
        ultimos_n: Número de últimos sorteios a buscar
    """
    print("🔄 Buscando dados atualizados da API...")
    df_api = buscar_dados_megasena_api(ultimos_n=ultimos_n)
    
    if df_api is None:
        print("❌ Não foi possível buscar dados da API.")
        return False
    
    print(f"✅ {len(df_api)} sorteios encontrados na API")
    
    # Tentar carregar Excel existente
    try:
        df_excel = pd.read_excel(arquivo_excel)
        print(f"📂 Arquivo Excel existente carregado: {len(df_excel)} linhas")
        
        # Verificar se há novos sorteios
        if 'concurso' in df_excel.columns and 'concurso' in df_api.columns:
            ultimo_concurso_excel = df_excel['concurso'].max()
            novos_sorteios = df_api[df_api['concurso'] > ultimo_concurso_excel]
            
            if len(novos_sorteios) > 0:
                print(f"✨ {len(novos_sorteios)} novos sorteios encontrados!")
                df_atualizado = pd.concat([df_excel, novos_sorteios], ignore_index=True)
                df_atualizado = df_atualizado.sort_values('concurso', ascending=False)
                df_atualizado.to_excel(arquivo_excel, index=False)
                print(f"✅ Arquivo atualizado: {len(df_atualizado)} sorteios totais")
                return True
            else:
                print("ℹ️ Nenhum novo sorteio encontrado.")
                return False
        else:
            # Sem coluna de concurso, adicionar todos
            df_atualizado = pd.concat([df_excel, df_api], ignore_index=True)
            df_atualizado.to_excel(arquivo_excel, index=False)
            print(f"✅ Arquivo atualizado: {len(df_atualizado)} sorteios totais")
            return True
    except FileNotFoundError:
        # Arquivo não existe, criar novo
        print("📝 Criando novo arquivo Excel...")
        df_api.to_excel(arquivo_excel, index=False)
        print(f"✅ Arquivo criado: {len(df_api)} sorteios")
        return True
    except Exception as e:
        print(f"❌ Erro ao atualizar arquivo: {e}")
        return False

if __name__ == "__main__":
    # Atualizar dados
    sucesso = atualizar_excel_com_api(ultimos_n=100)
    
    if sucesso:
        print("\n✅ Processo concluído com sucesso!")
    else:
        print("\n⚠️ Processo concluído com avisos. Verifique os dados manualmente.")


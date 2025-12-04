"""
🎯 Script Principal - Análise e Previsão Avançada Mega-Sena
=============================================================
Integra todas as análises e gera previsões inteligentes
Mega-Sena: 6 números de 1 a 60
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar o diretório atual ao path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar funções do treino_avancado
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("treino_avancado", "treino_avancado.py")
    treino_avancado = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(treino_avancado)
    
    # Extrair funções necessárias
    carregar_dados = treino_avancado.carregar_dados
    analisar_padroes_temporais = treino_avancado.analisar_padroes_temporais
    analisar_padroes_sequenciais = treino_avancado.analisar_padroes_sequenciais
    analisar_padroes_grupos = treino_avancado.analisar_padroes_grupos
    analisar_padroes_repeticao = treino_avancado.analisar_padroes_repeticao
    criar_features_avancadas = treino_avancado.criar_features_avancadas
    gerar_6_numeros_inteligentes = treino_avancado.gerar_6_numeros_inteligentes
except Exception as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

from analise_profunda_padroes import gerar_relatorio_completo, analise_estatistica_distribuicao, identificar_falhas_padroes
import joblib

st.set_page_config(page_title="Mega-Sena - Análise Avançada", layout="wide")

st.title("🎯 Sistema Avançado de Análise e Previsão - Mega-Sena")
st.markdown("---")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Opção de atualizar dados
if st.sidebar.button("🔄 Atualizar Dados da API"):
    with st.spinner("Atualizando dados..."):
        try:
            from buscar_dados_api import atualizar_excel_com_api
            sucesso = atualizar_excel_com_api()
            if sucesso:
                st.sidebar.success("✅ Dados atualizados!")
                st.rerun()
            else:
                st.sidebar.warning("⚠️ Erro ao atualizar dados")
        except Exception as e:
            st.sidebar.error(f"❌ Erro: {e}")

# Opções de análise
modo_analise = st.sidebar.selectbox(
    "Modo de Análise",
    ["Análise Completa", "Apenas Previsão", "Apenas Análise de Padrões"]
)

n_candidatos = st.sidebar.slider("Número de Candidatos para Gerar", 100, 2000, 1000)

# Carregar dados
df_numeros, y_binario = carregar_dados()

if df_numeros is None:
    st.error("❌ Erro ao carregar dados. Verifique o arquivo mega_sena.xlsx")
    st.stop()

st.info(f"📊 Dados carregados: {len(df_numeros)} sorteios históricos")

# ================================
# ANÁLISE DE PADRÕES
# ================================
if modo_analise in ["Análise Completa", "Apenas Análise de Padrões"]:
    st.header("🔍 Análise Profunda de Padrões")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Estatística", "⏰ Temporal", "🔗 Correlações", "🎯 Falhas"
    ])
    
    with tab1:
        st.subheader("Análise Estatística")
        try:
            stats_result = analise_estatistica_distribuicao(df_numeros)
        except Exception as e:
            st.error(f"Erro na análise estatística: {e}")
            st.info("Continuando com outras análises...")
            stats_result = {'frequencias': {}, 'teste_uniformidade': {'uniforme': False, 'p_value': 0}, 'numeros_anormais': []}
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Teste de Uniformidade", 
                     "Uniforme" if stats_result['teste_uniformidade']['uniforme'] else "Não Uniforme")
            st.write(f"P-value: {stats_result['teste_uniformidade']['p_value']:.4f}")
        
        with col2:
            if stats_result['numeros_anormais']:
                st.write("**Números com frequência anormal:**")
                for item in stats_result['numeros_anormais'][:5]:
                    st.write(f"- Número {item['numero']}: {item['frequencia']} ocorrências ({item['tipo']})")
        
        # Gráfico de frequências
        fig, ax = plt.subplots(figsize=(16, 5))
        nums = list(stats_result['frequencias'].keys())
        freqs = list(stats_result['frequencias'].values())
        ax.bar(nums, freqs)
        ax.axhline(y=np.mean(freqs), color='r', linestyle='--', label='Média')
        ax.set_xlabel("Número")
        ax.set_ylabel("Frequência")
        ax.set_title("Distribuição de Frequências (1-60)")
        ax.legend()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Padrões Temporais")
        with st.spinner("Analisando padrões temporais..."):
            padroes_temporais = analisar_padroes_temporais(df_numeros)
        
        if 'ciclos' in padroes_temporais:
            st.write("**Padrões Cíclicos Identificados:**")
            ciclos_df = pd.DataFrame([
                {
                    'Número': num,
                    'Intervalo Médio': info['medio'],
                    'Desvio Padrão': info['std'],
                    'Próximo Esperado': info['proximo_esperado']
                }
                for num, info in padroes_temporais['ciclos'].items()
            ])
            st.dataframe(ciclos_df.head(10))
        
        if 'tendencias' in padroes_temporais:
            st.write("**Tendências (Últimos vs Primeiros):**")
            tendencias_df = pd.DataFrame([
                {'Número': num, 'Tendência': 'Aumentando' if tend > 0 else 'Diminuindo', 'Valor': abs(tend)}
                for num, tend in padroes_temporais['tendencias'].items()
            ]).sort_values('Valor', ascending=False)
            st.dataframe(tendencias_df.head(10))
    
    with tab3:
        st.subheader("Correlações e Co-ocorrências")
        try:
            with st.spinner("Analisando correlações..."):
                padroes_grupos = analisar_padroes_grupos(df_numeros)
            
            if 'co_matrix' in padroes_grupos:
                st.write("**Matriz de Co-ocorrência (Top 30x30 para visualização):**")
                co_matrix = padroes_grupos['co_matrix']
                
                # Heatmap (mostrar apenas uma amostra para não ficar muito grande)
                fig, ax = plt.subplots(figsize=(12, 10))
                # Mostrar apenas números 1-30 para visualização
                sns.heatmap(co_matrix[1:31, 1:31], annot=False, cmap="YlOrRd", ax=ax)
                ax.set_title("Matriz de Co-ocorrência (Números 1-30)")
                st.pyplot(fig)
            else:
                st.warning("⚠️ Matriz de co-ocorrência não encontrada")
            
            if 'grupos_trios_frequentes' in padroes_grupos and padroes_grupos['grupos_trios_frequentes']:
                st.write("**Top 10 Trios Mais Frequentes:**")
                trios = list(padroes_grupos['grupos_trios_frequentes'].items())[:10]
                for i, (trio, freq) in enumerate(trios, 1):
                    st.write(f"{i}. {trio}: {freq} ocorrências")
            else:
                st.info("ℹ️ Nenhum trio frequente identificado ainda")
            
            if 'grupos_pares_frequentes' in padroes_grupos and padroes_grupos['grupos_pares_frequentes']:
                st.write("**Top 10 Pares Mais Frequentes:**")
                pares = list(padroes_grupos['grupos_pares_frequentes'].items())[:10]
                for i, (par, freq) in enumerate(pares, 1):
                    st.write(f"{i}. {par}: {freq} ocorrências")
        except Exception as e:
            st.error(f"❌ Erro ao analisar correlações: {e}")
            st.info("Tente novamente ou verifique os dados")
    
    with tab4:
        st.subheader("Falhas e Padrões Exploráveis")
        try:
            with st.spinner("Identificando falhas e padrões..."):
                falhas = identificar_falhas_padroes(df_numeros)
            
            falhas_freq = [f for f in falhas if f['tipo'] == 'frequencia_baixa']
            falhas_ausencia = [f for f in falhas if f['tipo'] == 'ausencia_prolongada']
            falhas_pares = [f for f in falhas if f['tipo'] == 'par_rarissimo']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Números com Frequência Baixa:**")
                if falhas_freq:
                    for f in falhas_freq[:10]:
                        st.write(f"- Número {f['numero']}: {f['frequencia']:.0f} ocorrências "
                               f"({f['diferenca_percentual']:.1f}% abaixo do esperado)")
                else:
                    st.info("ℹ️ Nenhum número com frequência anormalmente baixa")
            
            with col2:
                st.write("**Números com Ausência Prolongada:**")
                if falhas_ausencia:
                    for f in falhas_ausencia[:10]:
                        st.write(f"- Número {f['numero']}: {f['ausencia']} sorteios sem aparecer "
                               f"(Prob. retorno: {f['probabilidade_retorno']*100:.1f}%)")
                else:
                    st.info("ℹ️ Nenhum número com ausência prolongada")
            
            if falhas_pares:
                st.write("**Pares que Raramente Aparecem Juntos:**")
                for f in falhas_pares[:10]:
                    st.write(f"- Par {f['par']}: {f['ocorrencias']:.0f} ocorrências "
                           f"(Esperado mínimo: {f['esperado_minimo']:.0f})")
        except Exception as e:
            st.error(f"❌ Erro ao identificar falhas: {e}")
            st.info("Tente novamente ou verifique os dados")

# ================================
# PREVISÃO
# ================================
if modo_analise in ["Análise Completa", "Apenas Previsão"]:
    st.header("🎯 Geração de Previsões Inteligentes")
    
    # Verificar se modelo existe
    modelo_path = "modelo_avancado_mega.pkl"
    if not os.path.exists(modelo_path):
        st.warning("⚠️ Modelo não encontrado. Treinando novo modelo...")
        with st.spinner("Treinando modelo (isso pode levar alguns minutos)..."):
            from sklearn.multioutput import MultiOutputClassifier
            from xgboost import XGBClassifier
            
            padroes_temporais = analisar_padroes_temporais(df_numeros)
            padroes_sequenciais = analisar_padroes_sequenciais(df_numeros)
            padroes_grupos = analisar_padroes_grupos(df_numeros)
            padroes_repeticao = analisar_padroes_repeticao(df_numeros)
            
            X_avancado = criar_features_avancadas(
                df_numeros, padroes_temporais, padroes_sequenciais,
                padroes_grupos, padroes_repeticao
            )
            
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
            xgb_avancado.fit(X_avancado, y_binario)
            joblib.dump(xgb_avancado, modelo_path)
            modelo_final = xgb_avancado
            st.success("✅ Modelo treinado com sucesso!")
    else:
        modelo_final = joblib.load(modelo_path)
        st.success("✅ Modelo carregado!")
    
    # Gerar previsões
    if st.button("🎲 Gerar 6 Números Mais Prováveis", type="primary"):
        with st.spinner("Analisando padrões e gerando previsões..."):
            # Recalcular padrões para garantir dados atualizados
            padroes_temporais = analisar_padroes_temporais(df_numeros)
            padroes_sequenciais = analisar_padroes_sequenciais(df_numeros)
            padroes_grupos = analisar_padroes_grupos(df_numeros)
            padroes_repeticao = analisar_padroes_repeticao(df_numeros)
            
            X_avancado = criar_features_avancadas(
                df_numeros, padroes_temporais, padroes_sequenciais,
                padroes_grupos, padroes_repeticao
            )
            
            melhor_previsao, probs_final, top_candidatos = gerar_6_numeros_inteligentes(
                modelo_final, X_avancado, padroes_temporais, padroes_sequenciais,
                padroes_grupos, padroes_repeticao, df_numeros, n_candidatos=n_candidatos
            )
        
        # Exibir resultados
        st.success("✅ Previsões geradas com sucesso!")
        
        # ================================
        # 🎯 6 NÚMEROS MAIS PROVÁVEIS (DESTAQUE NO TOPO)
        # ================================
        st.markdown("---")
        st.markdown("## 🎯 6 NÚMEROS MAIS PROVÁVEIS")
        
        # Exibir os 6 números em destaque
        numeros_formatados = ', '.join([f"**{n:02d}**" for n in melhor_previsao])
        st.markdown(f"### {numeros_formatados}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{top_candidatos[0][1]:.2f}")
        with col2:
            st.metric("Total de Candidatos", n_candidatos)
        with col3:
            st.metric("Top 10 Gerados", len(top_candidatos))
        
        # ================================
        # 📋 LISTA COM 10 POSSÍVEIS
        # ================================
        st.markdown("---")
        st.markdown("## 📋 TOP 10 POSSÍVEIS COMBINAÇÕES")
        
        # Garantir que temos pelo menos 10 candidatos
        if len(top_candidatos) < 10:
            st.warning(f"⚠️ Apenas {len(top_candidatos)} candidatos foram gerados. Aumente o número de candidatos para ver mais opções.")
        
        # Criar tabela formatada
        dados_tabela = []
        for i, (cand, score) in enumerate(top_candidatos[:10], 1):
            numeros_str = ' - '.join([f"{n:02d}" for n in cand])
            dados_tabela.append({
                'Posição': i,
                'Números': numeros_str,
                'Score': f"{score:.2f}"
            })
        
        df_previsoes = pd.DataFrame(dados_tabela)
        st.dataframe(df_previsoes, use_container_width=True, hide_index=True)
        
        # Também exibir em formato de lista
        st.markdown("### 📝 Lista Detalhada:")
        for i, (cand, score) in enumerate(top_candidatos[:10], 1):
            numeros_str = ', '.join([f"{n:02d}" for n in cand])
            st.markdown(f"**{i}.** `{numeros_str}` - **Score:** {score:.2f}")
        
        # Gráfico de probabilidades
        st.subheader("📈 Probabilidades Finais Ajustadas")
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.bar(range(1, 61), probs_final)
        ax.set_xticks(range(1, 61, 5))  # Mostrar a cada 5 números
        ax.set_xlabel("Número")
        ax.set_ylabel("Probabilidade Ajustada")
        ax.set_title("Probabilidades Finais (Modelo + Padrões) - Mega-Sena")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Ranking completo
        st.subheader("📊 Ranking Completo de Probabilidades")
        ranking_df = pd.DataFrame({
            'Número': range(1, 61),
            'Probabilidade': probs_final
        }).sort_values('Probabilidade', ascending=False)
        st.dataframe(ranking_df.style.format({'Probabilidade': '{:.4f}'}))

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Aviso:</strong> Este sistema é para fins educacionais e de análise estatística.</p>
    <p>Não há garantia de acertos. Use com responsabilidade.</p>
</div>
""", unsafe_allow_html=True)


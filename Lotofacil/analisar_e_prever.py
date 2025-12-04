"""
🎯 Script Principal - Análise e Previsão Avançada Lotofácil
=============================================================
Integra todas as análises e gera previsões inteligentes
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

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
    gerar_15_numeros_inteligentes = treino_avancado.gerar_15_numeros_inteligentes
except Exception as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

from analise_profunda_padroes import gerar_relatorio_completo, analise_estatistica_distribuicao, identificar_falhas_padroes
import joblib

st.set_page_config(page_title="Lotofácil - Análise Avançada", layout="wide")

st.title("🎯 Sistema Avançado de Análise e Previsão - Lotofácil")
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
    st.error("❌ Erro ao carregar dados. Verifique o arquivo treino.xlsx")
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
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        nums = list(stats_result['frequencias'].keys())
        freqs = list(stats_result['frequencias'].values())
        ax.bar(nums, freqs)
        ax.axhline(y=np.mean(freqs), color='r', linestyle='--', label='Média')
        ax.set_xlabel("Número")
        ax.set_ylabel("Frequência")
        ax.set_title("Distribuição de Frequências")
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
        with st.spinner("Analisando correlações..."):
            padroes_grupos = analisar_padroes_grupos(df_numeros)
        
        if 'co_matrix' in padroes_grupos:
            st.write("**Matriz de Co-ocorrência:**")
            co_matrix = padroes_grupos['co_matrix']
            
            # Heatmap
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(co_matrix[1:26, 1:26], annot=False, cmap="YlOrRd", ax=ax)
            ax.set_title("Matriz de Co-ocorrência")
            st.pyplot(fig)
        
        if 'grupos_trios_frequentes' in padroes_grupos:
            st.write("**Top 10 Trios Mais Frequentes:**")
            trios = list(padroes_grupos['grupos_trios_frequentes'].items())[:10]
            for i, (trio, freq) in enumerate(trios, 1):
                st.write(f"{i}. {trio}: {freq} ocorrências")
    
    with tab4:
        st.subheader("Falhas e Padrões Exploráveis")
        falhas = identificar_falhas_padroes(df_numeros)
        
        falhas_freq = [f for f in falhas if f['tipo'] == 'frequencia_baixa']
        falhas_ausencia = [f for f in falhas if f['tipo'] == 'ausencia_prolongada']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Números com Frequência Baixa:**")
            if falhas_freq:
                for f in falhas_freq[:5]:
                    st.write(f"- Número {f['numero']}: {f['frequencia']:.0f} ocorrências "
                           f"({f['diferenca_percentual']:.1f}% abaixo do esperado)")
            else:
                st.write("Nenhum número com frequência anormalmente baixa")
        
        with col2:
            st.write("**Números com Ausência Prolongada:**")
            if falhas_ausencia:
                for f in falhas_ausencia[:5]:
                    st.write(f"- Número {f['numero']}: {f['ausencia']} sorteios sem aparecer "
                           f"(Prob. retorno: {f['probabilidade_retorno']*100:.1f}%)")
            else:
                st.write("Nenhum número com ausência prolongada")

# ================================
# PREVISÃO
# ================================
if modo_analise in ["Análise Completa", "Apenas Previsão"]:
    st.header("🎯 Geração de Previsões Inteligentes")
    
    # Verificar se modelo existe
    modelo_path = "modelo_avancado.pkl"
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
    if st.button("🎲 Gerar 15 Números Mais Prováveis", type="primary"):
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
            
            melhor_previsao, probs_final, top_candidatos = gerar_15_numeros_inteligentes(
                modelo_final, X_avancado, padroes_temporais, padroes_sequenciais,
                padroes_grupos, padroes_repeticao, n_candidatos=n_candidatos
            )
        
        # Exibir resultados
        st.success("✅ Previsões geradas com sucesso!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Previsão Principal")
            st.markdown(f"### **{', '.join(map(str, melhor_previsao))}**")
            st.write(f"**Score:** {top_candidatos[0][1]:.2f}")
        
        with col2:
            st.subheader("📊 Estatísticas")
            st.metric("Total de Candidatos Gerados", n_candidatos)
            st.metric("Top Score", f"{top_candidatos[0][1]:.2f}")
        
        # Top 10 candidatos
        st.subheader("🏆 Top 10 Candidatos Gerados")
        for i, (cand, score) in enumerate(top_candidatos, 1):
            st.write(f"**{i}.** `{', '.join(map(str, cand))}` - Score: {score:.2f}")
        
        # Gráfico de probabilidades
        st.subheader("📈 Probabilidades Finais Ajustadas")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(1, 26), probs_final)
        ax.set_xticks(range(1, 26))
        ax.set_xlabel("Número")
        ax.set_ylabel("Probabilidade Ajustada")
        ax.set_title("Probabilidades Finais (Modelo + Padrões)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Ranking completo
        st.subheader("📊 Ranking Completo de Probabilidades")
        ranking_df = pd.DataFrame({
            'Número': range(1, 26),
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


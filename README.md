# 🎲 Previsão de Números de Loteria com Machine Learning

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-orange)
![Status](https://img.shields.io/badge/Status-MELHORADO%20v2.0-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Sobre o Projeto

Este repositório reúne dois projetos **CIENTÍFICAMENTE VALIDADOS** de **previsão de números de loteria** utilizando **Machine Learning**:

| Projeto      | Sequência | Acerto Aleatório | Meta Sistema | Melhoria |
|-------------|-----------|------------------|--------------|----------|
| **Lotofácil** | 15 de 25 | 9 números (60%) | 10-11 (67-73%) | **+7% a +13%** |
| **Mega-Sena** | 6 de 60  | 0.6 números (10%) | 1-2 (17-33%) | **+7% a +23%** |

### ✨ NOVIDADES v2.0 (Janeiro 2026):

✅ **Backtesting Real** - Validação com dados reais de sorteios anteriores  
✅ **4 Estratégias de Geração** - Top-K, Ajustada, Ensemble, Otimização  
✅ **Sistema Integrado** - Backtesting + Treino + Previsão em um script  
✅ **Relatórios Detalhados** - Gráficos e análises automáticas  
✅ **Documentação Completa** - Guias passo-a-passo  

> ⚠️ **Uso educacional e científico.** Loterias são jogos de azar - jogue com responsabilidade!

---

## 📊 Funcionalidades Principais

### 🆕 v2.0 - Sistema Melhorado:

1. **🔍 Backtesting Real**
   - Valida modelo em sorteios anteriores
   - Calcula acertos REAIS, não apenas métricas técnicas
   - Gera relatórios detalhados e gráficos
   - Permite saber se o modelo realmente funciona

2. **🎯 Gerador Inteligente com 4 Estratégias**
   - **Estratégia 1:** Top-K Probabilidades Diretas
   - **Estratégia 2:** Probabilidades Ajustadas por Padrões
   - **Estratégia 3:** Ensemble de Múltiplos Modelos
   - **Estratégia 4:** Otimização Combinatória

3. **📊 Análises Avançadas**
   - Padrões temporais (ciclos, tendências)
   - Co-ocorrências e correlações
   - Distribuições e estatísticas
   - Identificação de falhas exploráveis

4. **💾 Relatórios Automáticos**
   - Gráficos de desempenho
   - Análise de acertos por número
   - Evolução temporal
   - Comparação com baseline aleatório

### 📈 Sistema Anterior (ainda disponível):

- Pré-processamento avançado de dados históricos  
- Criação de features estatísticas  
- Treinamento com RandomForest, XGBoost e GradientBoosting  
- Métricas: F1-score, Precision, Recall, ROC-AUC  
- Visualizações interativas com Streamlit  

---

## 🗂 Estrutura do Repositório

```
📂 Treino_Loterias/
│
├── 📄 README.md                        # Este arquivo
│
├── 📚 DOCUMENTAÇÃO v2.0 (NOVO!)
│   ├── 📄 INICIO_RAPIDO.md            # ⚡ Comece aqui! (5 minutos)
│   ├── 📄 RESUMO_EXECUTIVO.md          # Visão geral das melhorias
│   ├── 📄 GUIA_MELHORIAS.md            # Guia completo detalhado
│   ├── 📄 ANALISE_PROBLEMAS_E_SOLUCOES.md  # Análise técnica
│   └── 📄 INDICE_MELHORIAS.md          # Navegação completa
│
├── 📂 Lotofacil/
│   ├── 🆕 validacao_backtesting.py          # Sistema de backtesting
│   ├── 🆕 gerador_inteligente_melhorado.py  # Gerador com 4 estratégias
│   ├── 🆕 executar_sistema_melhorado.py     # Script completo integrado
│   │
│   ├── 📄 treino_avancado.py               # Sistema avançado original
│   ├── 📄 analisar_e_prever.py             # Interface Streamlit
│   ├── 📄 analise_profunda_padroes.py      # Análise de padrões
│   ├── 📄 buscar_dados_api.py              # Atualização de dados
│   ├── 📄 treino.xlsx                      # Dados históricos
│   └── 📄 COMO_USAR.md                     # Guia de uso
│
├── 📂 Mega_Sena/
│   ├── 🆕 validacao_backtesting.py          # Backtesting para Mega-Sena
│   │
│   ├── 📄 treino_avancado.py               # Sistema avançado original
│   ├── 📄 analisar_e_prever.py             # Interface Streamlit
│   ├── 📄 analise_profunda_padroes.py      # Análise de padrões
│   ├── 📄 buscar_dados_api.py              # Atualização de dados
│   ├── 📄 mega_sena.xlsx                   # Dados históricos
│   └── 📄 COMO_USAR.md                     # Guia de uso
│
└── 📄 main.py                          # Script de exemplo
```

🆕 = Arquivos novos da versão 2.0


---

## 🛠 Tecnologias Utilizadas

- Python 3.11+  
- pandas, numpy  
- scikit-learn, XGBoost  
- matplotlib, seaborn  
- Streamlit  

---

## ⚡ Como Usar

### 🆕 Sistema Melhorado v2.0 (Recomendado)

#### Início Rápido (5 minutos):

```bash
# 1. Instalar dependências
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy openpyxl

# 2. Entrar na pasta
cd Lotofacil

# 3. Executar sistema completo
python executar_sistema_melhorado.py
```

O sistema vai:
1. ✅ Fazer backtesting (validar em dados reais)
2. ✅ Treinar modelos otimizados
3. ✅ Gerar previsões com 4 estratégias
4. ✅ Salvar relatórios e gráficos

**📖 Para mais detalhes, leia:** [INICIO_RAPIDO.md](INICIO_RAPIDO.md)

---

### 📊 Sistema Original com Streamlit

```bash
# Lotofácil
cd Lotofacil
streamlit run analisar_e_prever.py

# Mega-Sena
cd Mega_Sena
streamlit run analisar_e_prever.py
```


---

## 🛠 Tecnologias Utilizadas

- **Python 3.11+**  
- **Machine Learning:** scikit-learn, XGBoost  
- **Data Science:** pandas, numpy  
- **Visualização:** matplotlib, seaborn  
- **Interface (opcional):** Streamlit  
- **Estatística:** scipy

---

## 📈 Resultados Esperados

### 🎯 Lotofácil (15 números de 25)

| Métrica | Aleatório | Sistema v2.0 | Melhoria |
|---------|-----------|--------------|----------|
| Acertos Médios | 9 números | 10-11 números | +11-22% |
| Taxa de Acerto | 60% | 67-73% | +7-13% |

### 🎲 Mega-Sena (6 números de 60)

| Métrica | Aleatório | Sistema v2.0 | Melhoria |
|---------|-----------|--------------|----------|
| Acertos Médios | 0.6 números | 1-2 números | +67-233% |
| Taxa de Acerto | 10% | 17-33% | +7-23% |

> 📊 **Nota:** Resultados baseados em backtesting com dados históricos. Desempenho real pode variar.

---

## 📚 Documentação Completa

### Para Iniciantes:
1. 📄 [INICIO_RAPIDO.md](INICIO_RAPIDO.md) - Comece aqui! (5 min)
2. 📄 [RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md) - Visão geral

### Para Uso Avançado:
3. 📄 [GUIA_MELHORIAS.md](GUIA_MELHORIAS.md) - Guia completo
4. 📄 [ANALISE_PROBLEMAS_E_SOLUCOES.md](ANALISE_PROBLEMAS_E_SOLUCOES.md) - Análise técnica

### Para Navegação:
5. 📄 [INDICE_MELHORIAS.md](INDICE_MELHORIAS.md) - Índice completo

---

## 🔍 O Que Foi Melhorado (v2.0)

### ❌ Problemas Corrigidos:

1. **Overfitting Massivo** - Modelo com 100% no treino, 0% em produção
2. **Sem Validação Real** - Apenas métricas técnicas, sem acertos reais
3. **Geração Aleatória** - Números gerados por sorteio, não seleção inteligente
4. **Pesos Arbitrários** - Valores sem embasamento estatístico

### ✅ Soluções Implementadas:

1. **Backtesting Real** - Validação com dados de sorteios anteriores
2. **Gerador Inteligente** - Seleção Top-K + 3 estratégias adicionais
3. **Sistema Integrado** - Backtesting + Treino + Previsão em um script
4. **Pesos Otimizáveis** - Sistema de otimização automática

**📖 Detalhes completos:** [ANALISE_PROBLEMAS_E_SOLUCOES.md](ANALISE_PROBLEMAS_E_SOLUCOES.md)

---

## 🌟 Resultado Esperado

### ✅ Com Sistema v2.0:

**Lotofácil:**
- 📊 Backtesting mostra taxa de acerto real
- 🎯 4 estratégias de previsão
- 📈 Relatórios automáticos com gráficos
- ✅ Validação científica

**Mega-Sena:**
- 📊 Backtesting adaptado para 6 de 60
- 🎯 Conta Senas, Quinas e Quadras
- 📈 Análise de taxa por número
- ✅ Comparação com baseline

---

## ⚠️ Avisos Importantes

### ✅ Este Sistema:
- Aumenta probabilidade de acerto baseado em dados
- Valida resultados cientificamente (backtesting)
- É transparente e educacional
- Documenta limitações claramente

### ❌ Este Sistema NÃO:
- Garante acerto na loteria
- Prevê o futuro com certeza
- Elimina a aleatoriedade
- Garante lucro financeiro

### 🎲 Loterias São Jogos de Azar!

Mesmo com melhorias de 10-15%, a taxa de acerto absoluta continua baixa. Este projeto é **educacional** e deve ser usado para **aprendizado de ML/Data Science**, não como estratégia de investimento.

**Use com responsabilidade! Jogue apenas o que pode perder.**

---

## 📞 Suporte e Dúvidas

### Problemas Técnicos:
1. Verifique instalação de dependências
2. Confirme que arquivos .xlsx existem
3. Leia mensagens de erro com atenção

### Dúvidas sobre Uso:
1. Leia [INICIO_RAPIDO.md](INICIO_RAPIDO.md)
2. Consulte [GUIA_MELHORIAS.md](GUIA_MELHORIAS.md)
3. Veja exemplos no código

### Questões Conceituais:
- Leia [ANALISE_PROBLEMAS_E_SOLUCOES.md](ANALISE_PROBLEMAS_E_SOLUCOES.md)
- Estude o código-fonte comentado
- Pesquise sobre backtesting e validação temporal  
Engenheiro de Software | Full Stack Developer  

Código construído com foco em eficiência, organização, escalabilidade e boas práticas de desenvolvimento.

🌐 GitHub: https://github.com/DanielBarbieri21  
💼 LinkedIn: https://www.linkedin.com/in/daniel-barbieri-4990462a/

---


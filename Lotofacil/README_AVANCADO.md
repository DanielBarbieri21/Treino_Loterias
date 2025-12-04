# 🎯 Sistema Avançado de Análise e Previsão - Lotofácil

## 📋 Visão Geral

Este sistema avançado foi desenvolvido para **identificar padrões profundos** e **aumentar a probabilidade de acerto** na Lotofácil através de análises estatísticas robustas e machine learning.

### 🎯 Objetivos

- ✅ Identificar **padrões temporais** (ciclos, tendências, sazonalidade)
- ✅ Analisar **padrões sequenciais** (consecutivos, intervalos, gaps)
- ✅ Descobrir **padrões de grupos** (co-ocorrências, clusters)
- ✅ Detectar **padrões de repetição** (retorno após X sorteios)
- ✅ Encontrar **falhas exploráveis** no histórico
- ✅ Gerar **15 números mais prováveis** baseado em múltiplos fatores

---

## 📁 Estrutura de Arquivos

```
Lotofacil/
├── treino_avancado.py          # Sistema principal de análise e ML
├── analise_profunda_padroes.py # Análise estatística avançada
├── buscar_dados_api.py         # Busca dados atualizados via API
├── analisar_e_prever.py       # Script principal integrado (Streamlit)
├── treino.xlsx                 # Base de dados históricos
└── modelo_avancado.pkl         # Modelo treinado (gerado automaticamente)
```

---

## 🚀 Como Usar

### 1. Instalação de Dependências

```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn scipy requests openpyxl
```

### 2. Preparar Dados

#### Opção A: Usar Excel Existente
Coloque seu arquivo `treino.xlsx` na pasta `Lotofacil/` com os sorteios históricos.

#### Opção B: Atualizar via API
```python
from buscar_dados_api import atualizar_excel_com_api
atualizar_excel_com_api(ultimos_n=100)
```

### 3. Executar Análise Completa

#### Via Streamlit (Recomendado - Interface Visual)
```bash
cd Lotofacil
streamlit run analisar_e_prever.py
```

#### Via Python (Análise de Padrões)
```bash
cd Lotofacil
python analise_profunda_padroes.py
```

---

## 🔍 Tipos de Análises Realizadas

### 1. **Padrões Temporais**
- **Ciclos**: Identifica períodos de repetição para cada número
- **Tendências**: Detecta se números estão aumentando ou diminuindo em frequência
- **Frequência Recente**: Analisa padrões nas últimas N jogadas

### 2. **Padrões Sequenciais**
- **Consecutivos**: Números que aparecem juntos em sequência
- **Intervalos**: Distâncias médias entre números
- **Gaps**: Lacunas grandes entre números
- **Distribuição por Faixas**: Análise de números por grupos (1-5, 6-10, etc.)

### 3. **Padrões de Grupos**
- **Co-ocorrências**: Pares, trios e quartetos que aparecem frequentemente juntos
- **Anti-co-ocorrências**: Números que raramente aparecem juntos
- **Clusters**: Grupos de números próximos que tendem a aparecer juntos

### 4. **Padrões de Repetição**
- **Retorno após N sorteios**: Probabilidade de um número retornar após X sorteios
- **Ausência Atual**: Quantos sorteios um número está sem aparecer
- **Alternâncias**: Padrões de aparecer/não aparecer

### 5. **Análise Estatística**
- **Teste de Uniformidade**: Verifica se a distribuição é aleatória
- **Z-scores**: Identifica números com frequência anormal
- **Correlações**: Análise de correlação entre números

---

## 🎲 Geração de Previsões

O sistema gera **15 números mais prováveis** usando:

1. **Probabilidades do Modelo ML**: XGBoost, RandomForest, GradientBoosting
2. **Ajustes Temporais**: Baseado em ciclos e ausências
3. **Ajustes de Co-ocorrência**: Baseado em padrões históricos
4. **Validação de Distribuição**: Evita concentrações anormais
5. **Validação de Intervalos**: Evita muitos números consecutivos

### Score de Candidatos

Cada candidato recebe um score baseado em:
- Probabilidade do modelo ML (peso: 10x)
- Co-ocorrências históricas (peso: 5x)
- Distribuição balanceada (penalização se desbalanceado)
- Intervalos adequados (penalização se muitos consecutivos)

---

## 📊 Interpretação dos Resultados

### Probabilidades Ajustadas
- Valores entre 0 e 1 indicam a probabilidade de cada número aparecer
- Números com probabilidade > 0.06 são considerados mais prováveis
- Ajustes temporais e de co-ocorrência podem aumentar/diminuir probabilidades

### Padrões Identificados
- **Ciclos**: Se um número tem ciclo de 5 sorteios, ele tende a aparecer a cada 5 sorteios
- **Ausência Prolongada**: Números que não aparecem há muito tempo têm maior probabilidade de retorno
- **Co-ocorrências**: Se números A e B aparecem juntos frequentemente, incluir A aumenta probabilidade de B

### Falhas Exploráveis
- **Frequência Baixa**: Números que aparecem menos que o esperado podem estar "atrasados"
- **Ausência Prolongada**: Números sem aparecer há muitos sorteios têm maior chance de retorno
- **Pares Raríssimos**: Pares que nunca aparecem juntos podem indicar padrões ocultos

---

## ⚙️ Configurações Avançadas

### Ajustar Número de Candidatos
No script `analisar_e_prever.py`, você pode ajustar:
```python
n_candidatos = st.sidebar.slider("Número de Candidatos", 100, 2000, 1000)
```
- Mais candidatos = mais tempo de processamento, mas melhor seleção
- Recomendado: 500-1000 para balance entre velocidade e qualidade

### Modificar Pesos de Score
No arquivo `treino_avancado.py`, função `gerar_15_numeros_inteligentes`:
```python
score += np.sum([probs_modelo[n-1] for n in numeros]) * 10  # Peso do modelo
score += co_matrix[pair[0], pair[1]] * 5  # Peso de co-ocorrência
```

---

## 🔄 Atualização de Dados

### Atualizar Manualmente
1. Adicione novos sorteios ao arquivo `treino.xlsx`
2. Execute novamente o script

### Atualizar via API
```python
from buscar_dados_api import atualizar_excel_com_api
atualizar_excel_com_api(ultimos_n=100)
```

---

## 📈 Melhorias em Relação ao Sistema Anterior

### ✅ Análises Mais Profundas
- Análise de ciclos e tendências temporais
- Identificação de padrões de repetição
- Análise estatística avançada (chi-quadrado, z-scores)

### ✅ Features Mais Ricas
- Features baseadas em múltiplos padrões
- Ajustes dinâmicos baseados em ausências e ciclos
- Validação de distribuições e intervalos

### ✅ Geração Mais Inteligente
- Score combinado de múltiplos fatores
- Geração de múltiplos candidatos e seleção do melhor
- Validação de padrões históricos

### ✅ Interface Mais Completa
- Visualizações interativas
- Análise detalhada de padrões
- Relatórios estatísticos

---

## ⚠️ Avisos Importantes

1. **Não há garantia de acertos**: Este sistema é para análise estatística e educacional
2. **Use com responsabilidade**: Não aposte mais do que pode perder
3. **Padrões podem mudar**: O sistema identifica padrões históricos, mas loterias são aleatórias
4. **Valide resultados**: Compare previsões com sorteios reais para avaliar eficácia

---

## 🐛 Solução de Problemas

### Erro ao carregar dados
- Verifique se o arquivo `treino.xlsx` existe
- Verifique se o formato está correto (15 números por linha)

### Erro ao buscar API
- A API pode estar temporariamente indisponível
- Use dados do Excel manualmente

### Modelo não encontrado
- O modelo será treinado automaticamente na primeira execução
- Isso pode levar alguns minutos

### Performance lenta
- Reduza o número de candidatos gerados
- Use menos splits na validação temporal

---

## 📞 Suporte

Para dúvidas ou melhorias, analise o código e ajuste conforme necessário.

---

## 📝 Licença

Uso educacional e de análise estatística. Use com responsabilidade.


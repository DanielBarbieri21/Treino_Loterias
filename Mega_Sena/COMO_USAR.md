# 🚀 Guia Rápido de Uso - Sistema Avançado Mega-Sena

## ⚡ Início Rápido

### 1. Instalar Dependências
```bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn scipy requests openpyxl
```

### 2. Executar Sistema Completo
```bash
cd Mega_Sena
streamlit run analisar_e_prever.py
```

O sistema abrirá no navegador automaticamente!

---

## 📋 Passo a Passo Detalhado

### Passo 1: Preparar Dados

Você precisa do arquivo `mega_sena.xlsx` com os sorteios históricos.

**Formato do Excel:**
- Cada linha = um sorteio
- 6 colunas com números de 1 a 60
- Exemplo:
```
num_1  num_2  num_3  num_4  num_5  num_6
  1      15     23     35     47     59
  2      18     25     38     42     55
```

### Passo 2: Atualizar Dados (Opcional)

Na interface Streamlit, clique em **"🔄 Atualizar Dados da API"** no menu lateral.

Ou execute:
```python
python buscar_dados_api.py
```

### Passo 3: Escolher Modo de Análise

No menu lateral, escolha:
- **Análise Completa**: Tudo (recomendado)
- **Apenas Previsão**: Só gera números
- **Apenas Análise de Padrões**: Só análises

### Passo 4: Gerar Previsões

1. Clique em **"🎲 Gerar 6 Números Mais Prováveis"**
2. Aguarde o processamento (pode levar 1-3 minutos)
3. Veja os resultados:
   - **Previsão Principal**: Os 6 números recomendados
   - **Top 10 Candidatos**: Outras opções geradas
   - **Probabilidades**: Gráfico e ranking completo

---

## 🔍 Entendendo os Resultados

### Previsão Principal
Os **6 números mais prováveis** baseados em:
- ✅ Modelo de Machine Learning
- ✅ Padrões temporais (ciclos, ausências)
- ✅ Co-ocorrências históricas
- ✅ Distribuição balanceada

### Score
Quanto **maior o score**, melhor a combinação segundo os padrões identificados.

### Probabilidades
- **> 0.02**: Número muito provável (ajustado para 60 números)
- **0.015 - 0.02**: Número provável
- **< 0.015**: Número menos provável

---

## 📊 Análises Disponíveis

### 1. Estatística
- Distribuição de frequências (1-60)
- Teste de uniformidade
- Números anormais

### 2. Temporal
- Padrões cíclicos
- Tendências (aumentando/diminuindo)
- Frequência recente

### 3. Correlações
- Matriz de co-ocorrência
- Pares e trios frequentes
- Grupos que aparecem juntos

### 4. Falhas
- Números com frequência baixa
- Números com ausência prolongada
- Padrões exploráveis

---

## 💡 Dicas de Uso

### Para Melhor Precisão:
1. ✅ Use dados atualizados (últimos 100+ sorteios)
2. ✅ Analise os padrões antes de gerar previsões
3. ✅ Compare múltiplos candidatos
4. ✅ Verifique números com ausência prolongada (>15 sorteios)

### Para Análise Profunda:
1. Execute `python analise_profunda_padroes.py` para relatório completo
2. Analise os padrões temporais identificados
3. Verifique as correlações entre números
4. Identifique falhas exploráveis

### Para Atualização Contínua:
1. Configure atualização automática via API
2. Execute análises regularmente
3. Compare previsões com resultados reais
4. Ajuste estratégia baseado em resultados

---

## ⚙️ Configurações Avançadas

### Ajustar Número de Candidatos
No menu lateral, ajuste o slider:
- **100-500**: Rápido, menos preciso
- **500-1000**: Balanceado (recomendado)
- **1000-2000**: Mais lento, mais preciso

### Modificar Pesos (Código)
Edite `treino_avancado.py`, função `gerar_6_numeros_inteligentes`:
```python
score += np.sum([probs_modelo[n-1] for n in numeros]) * 10  # Peso modelo
score += co_matrix[pair[0], pair[1]] * 5  # Peso co-ocorrência
```

---

## 🐛 Problemas Comuns

### Erro: "Arquivo não encontrado"
- Verifique se `mega_sena.xlsx` está na pasta `Mega_Sena/`
- Verifique o formato do arquivo

### Erro: "Modelo não encontrado"
- Normal na primeira execução
- O modelo será treinado automaticamente
- Aguarde 2-5 minutos (mais tempo que Lotofácil)

### Performance Lenta
- Reduza número de candidatos
- Use menos análises simultâneas
- Feche outras aplicações

### API não funciona
- Use dados do Excel manualmente
- Verifique conexão com internet
- API pode estar temporariamente indisponível

---

## 📈 Próximos Passos

1. **Teste as Previsões**: Compare com sorteios reais
2. **Analise Padrões**: Identifique padrões que funcionam melhor
3. **Ajuste Estratégia**: Modifique pesos conforme resultados
4. **Atualize Regularmente**: Mantenha dados atualizados

---

## ⚠️ Lembrete Importante

Este sistema é para **análise estatística e educacional**. 

- ❌ Não há garantia de acertos
- ❌ Loterias são aleatórias por natureza
- ✅ Use com responsabilidade
- ✅ Não aposte mais do que pode perder

---

## 📞 Suporte

Para dúvidas:
1. Leia o `README_AVANCADO.md` para documentação completa
2. Analise o código para entender o funcionamento
3. Ajuste conforme suas necessidades

---

**Boa sorte com suas análises! 🍀**


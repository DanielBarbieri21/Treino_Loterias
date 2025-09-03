# 🎲 Previsão de Números de Loteria com Machine Learning

![GitHub repo size](https://img.shields.io/github/repo-size/seuusuario/loterias-ml)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Sobre o Projeto

Este repositório reúne dois projetos interativos de **previsão de números de loteria** utilizando **Machine Learning**:

| Projeto      | Sequência de Números | Descrição |
|-------------|-------------------|-----------|
| Lotofácil   | 15                | Analisa histórico, cria features estatísticas e gera 10 sequências prováveis. |
| Mega-Sena   | 6                 | Analisa histórico, cria features estatísticas e gera 10 sequências prováveis. |

Ambos os projetos utilizam **RandomForest** e **XGBoost** para treinar modelos multi-output, com **features avançadas** como frequência recente, quantidade de pares/impares e soma total dos números.  
O objetivo é **explorar padrões históricos e gerar previsões com base em probabilidades**, de forma educativa e interativa.

> ⚠️ Uso educativo: não há garantia de acertos na loteria.

---

## 📊 Funcionalidades Principais

- **Pré-processamento avançado**: dados históricos, filtragem de números, remoção de duplicatas.  
- **Criação de features**: pares, ímpares, soma e frequência recente.  
- **Treinamento e avaliação de modelos**:
  - DummyClassifier (baseline)
  - RandomForest e XGBoost
  - Métricas: F1-score, Precision, Recall, Matriz de Confusão, ROC-AUC
  - Validação cruzada  
- **Geração de sequências prováveis**:
  - Lotofácil: 10 sequências de 15 números  
  - Mega-Sena: 10 sequências de 6 números  
- **Visualizações interativas com Streamlit**:
  - Distribuição histórica dos números  
  - Frequência e heatmap  
  - Padrões temporais  
  - Gráficos de barras com probabilidades

![Exemplo Streamlit](docs/streamlit_demo.gif)

---

## 🗂 Estrutura do Repositório

Loterias/
│
├── Lotofacil/
│ ├── treino.py # Script principal da Lotofácil
│ ├── treino.xlsx # Base de dados da Lotofácil
│ └── modelo_lotofacil.pkl
│
├── Mega_Sena/
│ ├── treino.py # Script principal da Mega-Sena
│ ├── mega_sena.xlsx # Base de dados da Mega-Sena
│ └── modelo_xgb.pkl
│
├── docs/ # Screenshots, GIFs, imagens de demonstração
├── requirements.txt # Bibliotecas necessárias
└── README.md # Este arquivo


---

## 🛠 Tecnologias Utilizadas

- Python 3.11+  
- pandas, numpy  
- scikit-learn, XGBoost  
- matplotlib, seaborn  
- Streamlit  

---

## ⚡ Como Usar

1. Clone o repositório:
```bash
git clone https://github.com/seuusuario/loterias-ml.git
cd loterias-ml


🌟 Resultado Esperado

✅ 10 sequências exclusivas para cada loteria

✅ Visualizações interativas de frequência, heatmaps e padrões

✅ Métricas detalhadas de performance dos modelos

✅ Interface moderna e intuitiva via Streamlit

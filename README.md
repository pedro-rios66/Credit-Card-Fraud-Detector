# 💳 Credit Card Fraud Detection

Este projeto implementa um modelo de aprendizado de máquina para detectar transações fraudulentas com cartões de crédito. Ele utiliza o algoritmo **Random Forest**, técnicas de pré-processamento de dados, validação cruzada, ajuste de threshold e visualizações úteis para análise de desempenho.

---

## 📂 Estrutura do Projeto

```
credit-card-fraud-detector/
├── data/
│   └── creditcard.csv         # Dataset original (Kaggle)
├── data_exploration.ipynb     # Código principal do projeto
├── outputs/
│   ├── rf_model.joblib        # Modelo treinado (opcional)
│   └── scaler.joblib          # Scaler treinado (opcional)
├── README.md
└── requirements.txt
```

---

## ⚙️ Tecnologias e Bibliotecas

- `pandas`, `numpy`  
- `scikit-learn`  
- `matplotlib`, `seaborn`  
- `joblib`  

---

## 🚀 Passo a Passo

1. **Instale as dependências**:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Certifique-se de que o arquivo `creditcard.csv` esteja no diretório `data/`.**  
   Você pode baixar o dataset em:  
   [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

3. **Execute o script principal**:  
   *(No caso, o notebook)*  
   ```bash
   jupyter notebook data_exploration.ipynb
   ```

---

## 📊 Descrição do Pipeline

### 🔹 Carregamento e pré-processamento:
- Leitura do dataset `.csv`
- Remoção da coluna `id` (se existir)
- Separação entre variáveis explicativas (features) e variável alvo (`Class`)
- Normalização das features com `StandardScaler`

### 🔹 Modelagem:
- Treinamento de um modelo `RandomForestClassifier` com hiperparâmetros ajustados
- Aplicação de validação cruzada (5-fold) com métrica `f1-score`
- Ajuste do limiar de decisão (`threshold = 0.4`)

### 🔹 Avaliação:
- Exibição do relatório de classificação
- Matriz de confusão com `seaborn`
- Gráfico **Precision-Recall vs Threshold**

---

## 📈 Resultados Esperados

- **F1 médio (validação cruzada)**: ~0.83

Na base de teste com `threshold = 0.4`:
- **Precision** ≈ 0.90  
- **Recall** ≈ 0.84  
- **F1-score** ≈ 0.87  

O projeto está preparado para lidar com desbalanceamento de classes, utilizando `class_weight='balanced'`.

---

## 💾 Salvando o Modelo

Para salvar o modelo e o scaler treinado, adicione ao final do script:

```python
joblib.dump(rf_model, 'outputs/rf_model.joblib')
joblib.dump(scaler, 'outputs/scaler.joblib')
```

---

## 🔮 Possíveis Melhorias Futuras

- Implementar comparação com outros modelos (XGBoost, LightGBM, etc.)
- Calibração das probabilidades preditivas
- Automatização do tuning de hiperparâmetros
- Criação de API com Flask ou FastAPI para deploy
- Dashboard com Streamlit para análise interativa

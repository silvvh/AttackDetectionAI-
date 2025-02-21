<a href="https://colab.research.google.com/github/silvvh/AttackDetectionAI-/blob/main/Projeto_Final.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
#bilbiotecas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, make_scorer, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#IAs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
```


```python
df = pd.read_csv("cybersecurity_intrusion_data.csv")
# Remover os espaços em branco do começo e do final dos nomes das colunas
df.columns = [i.strip() for i in df.columns]

# Mostrar um resumo da base de dados
df
```





  <div id="df-6c659af1-9716-4260-bfba-e1d334b0b5ce" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session_id</th>
      <th>network_packet_size</th>
      <th>protocol_type</th>
      <th>login_attempts</th>
      <th>session_duration</th>
      <th>encryption_used</th>
      <th>ip_reputation_score</th>
      <th>failed_logins</th>
      <th>browser_type</th>
      <th>unusual_time_access</th>
      <th>attack_detected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SID_00001</td>
      <td>599</td>
      <td>TCP</td>
      <td>4</td>
      <td>492.983263</td>
      <td>DES</td>
      <td>0.606818</td>
      <td>1</td>
      <td>Edge</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SID_00002</td>
      <td>472</td>
      <td>TCP</td>
      <td>3</td>
      <td>1557.996461</td>
      <td>DES</td>
      <td>0.301569</td>
      <td>0</td>
      <td>Firefox</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SID_00003</td>
      <td>629</td>
      <td>TCP</td>
      <td>3</td>
      <td>75.044262</td>
      <td>DES</td>
      <td>0.739164</td>
      <td>2</td>
      <td>Chrome</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SID_00004</td>
      <td>804</td>
      <td>UDP</td>
      <td>4</td>
      <td>601.248835</td>
      <td>DES</td>
      <td>0.123267</td>
      <td>0</td>
      <td>Unknown</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SID_00005</td>
      <td>453</td>
      <td>TCP</td>
      <td>5</td>
      <td>532.540888</td>
      <td>AES</td>
      <td>0.054874</td>
      <td>1</td>
      <td>Firefox</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9532</th>
      <td>SID_09533</td>
      <td>194</td>
      <td>ICMP</td>
      <td>3</td>
      <td>226.049889</td>
      <td>AES</td>
      <td>0.517737</td>
      <td>3</td>
      <td>Chrome</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9533</th>
      <td>SID_09534</td>
      <td>380</td>
      <td>TCP</td>
      <td>3</td>
      <td>182.848475</td>
      <td>NaN</td>
      <td>0.408485</td>
      <td>0</td>
      <td>Chrome</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9534</th>
      <td>SID_09535</td>
      <td>664</td>
      <td>TCP</td>
      <td>5</td>
      <td>35.170248</td>
      <td>AES</td>
      <td>0.359200</td>
      <td>1</td>
      <td>Firefox</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9535</th>
      <td>SID_09536</td>
      <td>406</td>
      <td>TCP</td>
      <td>4</td>
      <td>86.664703</td>
      <td>AES</td>
      <td>0.537417</td>
      <td>1</td>
      <td>Chrome</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9536</th>
      <td>SID_09537</td>
      <td>340</td>
      <td>TCP</td>
      <td>6</td>
      <td>86.876744</td>
      <td>NaN</td>
      <td>0.277069</td>
      <td>4</td>
      <td>Chrome</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>9537 rows × 11 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6c659af1-9716-4260-bfba-e1d334b0b5ce')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6c659af1-9716-4260-bfba-e1d334b0b5ce button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6c659af1-9716-4260-bfba-e1d334b0b5ce');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5e817b50-20ed-42a5-abcc-86d0c99e6aa9">
  <button class="colab-df-quickchart" onclick="quickchart('df-5e817b50-20ed-42a5-abcc-86d0c99e6aa9')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5e817b50-20ed-42a5-abcc-86d0c99e6aa9 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_49fe7a8f-da0c-48e0-858c-a557d9965c6e">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_49fe7a8f-da0c-48e0-858c-a557d9965c6e button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
#OneHotEncoder
transformers = [
    ('oh_protocol_type', OneHotEncoder(sparse_output=False), ['protocol_type']),
    ('oh_encryption_used', OneHotEncoder(sparse_output=False), ['encryption_used']),
    ('oh_browser_type',OneHotEncoder(sparse_output=False), ['browser_type'] )
]
ct_oh = ColumnTransformer(
    transformers, remainder='passthrough'
)

 # as características: todas as colunas exceto "attack_detected"
X = df.drop(["attack_detected", "session_id"], axis=1)
# y são os rótulos: apenas a coluna "attack_detected"
y = df["attack_detected"]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, stratify=y, test_size=0.2, random_state=99)
X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=100)

```


```python
def get_model_and_params(model_type):
    if model_type == 'svm':
        model = SVC()
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf']
        }
    elif model_type == 'knn':
        model = KNeighborsClassifier()
        param_grid = {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan']
        }
    elif model_type == 'tree':
        model = DecisionTreeClassifier()
        param_grid = {
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        }
    elif model_type == 'mlp':
        model = MLPClassifier(max_iter=2000, solver='adam', learning_rate_init=0.001, early_stopping=True)
        param_grid = {
           'model__hidden_layer_sizes': [(50,), (100, 50), (100, 100, 100)],
            'model__activation': ['relu', 'tanh']
        }
    else:
        raise ValueError("Modelo desconhecido")
    return model, param_grid

# Escolher modelo (altere aqui)
model_type = 'svm'  # Opções: 'svm', 'knn', 'tree', 'mlp'
model, param_grid = get_model_and_params(model_type)

# Criar pipeline
pipeline = Pipeline([
    ('preprocessing', ct_oh),
    ('scaler', StandardScaler()),
    ('model', model)
])

# GridSearchCV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')

grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring=f1_scorer, n_jobs=-1)
grid_search.fit(X_treino, y_treino)

# Avaliação no conjunto de validação
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
f1_val = f1_score(y_val, y_val_pred, average='weighted')

print("Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)
print(f"F1-score no conjunto de validação: {f1_val:.4f}")

# Avaliação final no conjunto de teste
y_test_pred = best_model.predict(X_teste)
f1_test = f1_score(y_teste, y_test_pred, average='weighted')

print(f"F1-score no conjunto de teste: {f1_test:.4f}")
```

    Melhores hiperparâmetros encontrados:
    {'model__C': 10, 'model__kernel': 'rbf'}
    F1-score no conjunto de validação: 0.8645
    F1-score no conjunto de teste: 0.8607
    


```python
#MÉTRICAS
# Usando F1-score como métrica
f1_scorer = make_scorer(f1_score, average='weighted')  # 'weighted' para lidar com classes desbalanceadas

# Avaliação com cross_val_score usando F1-score
scores = cross_val_score(pipeline, X, y, cv=kf, scoring=f1_scorer)

# Resultados
print(f"F1-score médio (cross-validation): {scores.mean():.4f} ± {scores.std():.4f}")

# Treinando o pipeline com os dados de treino
pipeline.fit(X_treino, y_treino)

# Fazendo previsões no conjunto de teste
y_pred = pipeline.predict(X_teste)

# Acurácia
accuracy = accuracy_score(y_teste, y_pred)
print(f"Acurácia: {accuracy:.4f}")

# Métricas de classificação: F1-Score, Precision, Recall
print("Métricas de Classificação:")
print(classification_report(y_teste, y_pred))
```

    F1-score médio (cross-validation): 0.8590 ± 0.0094
    Acurácia: 0.8443
    Métricas de Classificação:
                  precision    recall  f1-score   support
    
               0       0.81      0.94      0.87      1055
               1       0.91      0.72      0.81       853
    
        accuracy                           0.84      1908
       macro avg       0.86      0.83      0.84      1908
    weighted avg       0.85      0.84      0.84      1908
    
    


```python
# Gerando a matriz de confusão
cm = confusion_matrix(y_teste, y_pred)

# Plotando a matriz de confusão com os valores
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.title('Matriz de Confusão com Valores')
plt.xlabel('Classes Preditas')
plt.ylabel('Classes Reais')
plt.show()
```


    
![png](output_6_0.png)
    


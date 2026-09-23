# 📊 Analise Score

Projeto de **credit scoring com Machine Learning** para estimar o risco de inadimplência a partir de dados cadastrais e financeiros de solicitações de crédito. O repositório reúne análise exploratória, treinamento e comparação de modelos e uma interface interativa em Streamlit.

## 🎯 Objetivo

O projeto usa a variável `loan_status` como alvo para classificar solicitações de crédito. O fluxo prepara os dados, treina três classificadores, compara as métricas e seleciona o modelo com maior ROC-AUC para uso na aplicação demonstrativa.

## ✨ Funcionalidades

- Leitura e filtragem básica do dataset, removendo registros com idade acima de 100 anos ou tempo de emprego acima de 80 anos.
- Imputação de ausências, padronização numérica e One-Hot Encoding de variáveis categóricas.
- Treinamento e comparação de Logistic Regression, Random Forest e Gradient Boosting.
- Avaliação por accuracy, precision, recall, F1-score, ROC-AUC, matriz de confusão e classification report.
- Seleção do melhor modelo por ROC-AUC e persistência com Joblib.
- Geração de relatórios em CSV e JSON.
- Interface Streamlit para estimar a probabilidade e uma faixa de risco.
- Notebook de análise exploratória e estatística.

## 🛠️ Tecnologias

- Python `>=3.14`
- Pandas, NumPy, Scikit-learn e Joblib
- Streamlit
- Matplotlib, Seaborn, Plotly e Missingno
- uv para gerenciamento do ambiente e do lockfile

## 🏗️ Organização do projeto

```text
analise-score/
├── analise-estatistica/
│   └── analise-dados.ipynb       # Análise exploratória e estatística
├── app/
│   └── app.py                    # Interface Streamlit
├── data/
│   ├── 01-raw/credit_risk_dataset.csv
│   └── 02-processed/credit_risk_tratado.csv
├── reports/
│   ├── metrics_full.json
│   └── metrics_summary.csv
├── models/
│   └── .gitkeep                    # Mantém a pasta; o modelo é gerado localmente
├── src/train_model.py            # Treinamento e avaliação
├── main.py                       # Exemplo mínimo do projeto
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

O diretório `models/` é mantido no repositório por meio de `.gitkeep`, mas o arquivo `credit_scoring_model.joblib` é ignorado pelo Git. O treinamento cria esse artefato localmente; por isso, deve ser executado antes da aplicação.

## 📋 Pré-requisitos

- Python 3.14 ou superior, conforme `pyproject.toml` e `uv.lock`.
- Git.
- uv (recomendado) ou pip.

## 📦 Instalação

### ⚡ Opção 1 — uv (recomendada)

Instale o `uv` seguindo a documentação oficial ou, no Windows PowerShell, execute:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

No Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Depois, clone o repositório e sincronize o ambiente:

```bash
git clone https://github.com/Robert-FM/analise-score.git
cd analise-score
uv sync
```

O `uv sync` cria o ambiente virtual `.venv`, instala as dependências declaradas em `pyproject.toml` e respeita as versões resolvidas em `uv.lock`. O projeto não define grupos ou extras de desenvolvimento. Para atualizar o ambiente após alterações nas dependências, execute novamente `uv sync`.

Os comandos podem ser executados diretamente com `uv run`, sem ativar manualmente o ambiente:

```bash
uv run python --version
uv run python src/train_model.py
uv run streamlit run app/app.py
```

Se quiser ativar o ambiente explicitamente:

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Para adicionar uma dependência ao projeto durante o desenvolvimento, use:

```bash
uv add nome-do-pacote
```

O treinamento não baixa nem versiona o modelo no Git. Ele cria ou atualiza localmente `models/credit_scoring_model.joblib`, que é necessário para executar a aplicação Streamlit. O arquivo é excluído pelo padrão `models/*.joblib` em `.gitignore`.

### 🐍 Opção 2 — pip

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Como não há `requirements.txt`, instale as dependências declaradas no `pyproject.toml`:

```bash
python -m pip install joblib matplotlib missingno numpy pandas plotly scikit-learn seaborn streamlit
```

## ▶️ Treinamento

```bash
uv run python src/train_model.py
```

Ou, com o ambiente ativado:

```bash
python src/train_model.py
```

O script lê `data/01-raw/credit_risk_dataset.csv`, usa divisão estratificada (`test_size=0.2`, `random_state=42`), treina os três modelos e cria o modelo persistido em `models/` e os relatórios em `reports/`.

## 💻 Aplicação interativa

Depois do treinamento:

```bash
uv run streamlit run app/app.py
```

Ou:

```bash
streamlit run app/app.py
```

A interface coleta idade, renda, moradia, tempo de emprego, finalidade e grade do empréstimo, valor, taxa de juros e histórico de crédito. Ao clicar em **Calcular risco**, exibe a probabilidade estimada e usa os limiares implementados: baixo risco (`<25%`), risco moderado (`25%–<50%`) e alto risco (`>=50%`).

## 🔬 Análise exploratória

O notebook [`analise-estatistica/analise-dados.ipynb`](analise-estatistica/analise-dados.ipynb) lê o dataset bruto e documenta exploração, tipos, valores ausentes, estatísticas descritivas, correlações e visualizações.

## 📈 Resultados registrados

Os relatórios versionados indicam o `random_forest` como melhor modelo pela ROC-AUC:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9303 | 0.9773 | 0.6967 | 0.8135 | 0.9308 |
| Gradient Boosting | 0.9231 | 0.9440 | 0.6882 | 0.7961 | 0.9206 |
| Logistic Regression | 0.8107 | 0.5478 | 0.7579 | 0.6360 | 0.8636 |

Os valores correspondem à execução registrada em `reports/metrics_summary.csv`.

## ⚠️ Limitações e próximos passos

- A aplicação depende de `models/credit_scoring_model.joblib`, criado localmente pelo treinamento.
- Não foram identificados testes automatizados.
- `main.py` contém apenas uma mensagem de exemplo e não é o ponto de entrada do treinamento ou da aplicação.
- Melhorias possíveis: validação cruzada, análise de limiar e custo de erros, explicabilidade, testes e deploy.

## 👨‍💻 Autor

**Robert Melo**

🔗 LinkedIn: [linkedin.com/in/robertdemelo](https://www.linkedin.com/in/robertdemelo/)

🐍 Python | Scikit-learn | Pandas | Streamlit | Machine Learning | Credit Scoring

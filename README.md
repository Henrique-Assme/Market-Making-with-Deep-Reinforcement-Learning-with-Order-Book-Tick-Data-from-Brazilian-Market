# Market Making com Deep RL e Livro de Ofertas da B3

Repositório do Trabalho de Conclusão de Curso (TCC) “Market Making with Deep Reinforcement Learning Based on Signals and Order Book
Tick Data from Brazilian Market”. O documento completo está disponível em `./TCC_EPUSP_Henrique_Assme.pdf` e detalha o racional teórico, o desenho do ambiente e os experimentos realizados.

---

- [Visão Geral](#visão-geral)
- [Componentes Principais](#componentes-principais)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Pré-requisitos e Setup](#pré-requisitos-e-setup)
- [Preparação dos Dados](#preparação-dos-dados)
- [Treinamento](#treinamento)
- [Avaliação e Métricas](#avaliação-e-métricas)
- [Demonstração Web](#demonstração-web)
- [Monitoramento e Notebooks](#monitoramento-e-notebooks)
- [Próximos Passos](#próximos-passos)

---

## Visão Geral
- Agente de market making passivo treinado com PPO (Stable-Baselines3) usando dados de livro de ofertas e negócios da B3 disponibilizados via BTG Pactual Data Services.
- Ambiente customizado (`TradeEngineEnv`) que simula cotações persistentes em ambos os lados do book, controle de inventário, ordem com TTL e múltiplos componentes de recompensa (mark-to-market, pnl realizado positivo/negativo, penalização de inventário/no-quote).
- Pipeline de preparação de dados em nível de tick (book + trades) e rotina de avaliação determinística com geração de gráficos, métricas por dia/ticker/categoria de liquidez e app Streamlit para exploração interativa.

## Componentes Principais
- **Ambiente RL (`trade_env.py`)**: observa best bid/ask (e tamanho quando disponível), normaliza inventário máximo, aplica spreads mínimos por tick e mantém fila FIFO de ordens vivas por lado.
- **Política customizada (`policy.py` + `backbone.py`)**: extrator simples em PyTorch com largura configurável que alimenta a política/valor da PPO.
- **Callbacks e wrappers (`ActionLoggerCallback`, `LoggingVecNormalize`)**: enviam distribuição de ações ao TensorBoard e incorporam recompensas normalizadas aos logs do ambiente.
- **Pipeline de dados (`tick_data_prep.ipynb`)**: baixa dados via API Bulk do BTG (book incremental + trades), faz filtro de sessão (13h–20h50 UTC), remove duplicidades, alinha eventos por `rpt_seq` e salva CSVs em `train_data/` e `test_data/`.
- **Ferramentas de avaliação (`test_model.py`, `aggregate_metrics.py`)**: executam o modelo treinado de maneira determinística, salvam métricas/plots/logs por dia e consolidam resultados por ticker e por “faixa de liquidez”.
- **Demonstração (`demo_app.py`)**: interface Streamlit que permite escolher dias/tickers, rodar o modelo, visualizar métricas e baixar relatórios.

## Estrutura do Repositório
```
.
├── DRL-TCC/
│   ├── train.py                  # loop de treino PPO com SubprocVecEnv
│   ├── test_model.py             # avaliação determinística e geração de métricas
│   ├── trade_env.py              # ambiente Gymnasium customizado
│   ├── policy.py / backbone.py   # arquitetura da política
│   ├── aggregate_metrics.py      # consolidação por ticker/categoria
│   ├── demo_app.py               # app Streamlit
│   ├── tick_data_prep.ipynb      # notebook de coleta/tratamento de dados
│   ├── train_data/ | test_data/  # dados em CSV (não versionados originalmente)
│   ├── evaluation_results/       # saídas da avaliação
│   ├── tb_logs/                  # logs para TensorBoard
│   ├── MM_model_trained.zip      # modelo PPO pré-treinado (quando disponível)
│   └── pyproject.toml / uv.lock  # dependências gerenciadas via uv
├── README.md
└── TCC_EPUSP_Henrique_Assme.pdf
```

## Pré-requisitos e Setup
1. **Python 3.10+** e [uv](https://github.com/astral-sh/uv) instalados na máquina.
2. Clonar o repositório e acessar `DRL-TCC/`.
3. Instalar dependências sem criar env manualmente:
   ```bash
   cd DRL-TCC
   uv run -- python -V          # dispara resolução de dependências conforme pyproject/uv.lock
   ```
   (Caso prefira ambiente dedicado: `uv venv .venv && source .venv/bin/activate && uv pip sync`.)
4. Para a etapa de coleta de dados é preciso um `.env` com `BTG_API_KEY` válido (BTG Solutions Data Services).

## Preparação dos Dados
1. Configure o arquivo `.env` no diretório `DRL-TCC/`:
   ```
   BTG_API_KEY=seu_token_aqui
   ```
2. Execute/abra `tick_data_prep.ipynb` para:
   - Baixar dados de **livro incremental** e **trades** via `btgsolutions-dataservices-python-client`.
   - Filtrar a janela intraday (13:00–20:50 UTC) e remover duplicidades.
   - Realizar forward-fill do melhor bid/ask antes de unir trades + book.
   - Salvar CSVs de treino (`train_data/`) e teste (`test_data/`), seguindo o padrão `RANK_TICKER_YYYYMMDD.csv` para testes.
3. A lista de tickers de teste pode ser montada automaticamente a partir do endpoint “Top N Liquidity” (vide notebook).

> **Observação**: o repositório inclui os dados de treino e teste utilizados na monografia em `train_data/` e `test_data/` copm a devida autorização do BTG para tal.

## Treinamento
1. Certifique-se de que a pasta `train_data/` contém os dias desejados.
2. Parâmetros importantes em `train.py`:
   - `N_DAYS`: quantidade de dias utilizados por época (default 19 mais recentes).
   - `PASSES`: número de passagens sobre o conjunto (timesteps totais = médias de steps × dias × passes).
   - `VEC_MODE`: `subproc` para paralelização com SubprocVecEnv (default).
   - `quote_size`, `spread`, `inventory` etc. são definidos ao instanciar `TradeEngineEnv`.
3. Execute:
   ```bash
   cd DRL-TCC
   uv run train.py
   ```
4. Artefatos gerados:
   - `MM_model_trained.zip`: pesos da PPO.
   - `vecnormalize_stats.pkl`: estatísticas de normalização para reuso na inferência.
   - TensorBoard em `tb_logs/` com métricas de recompensa e distribuição de ações.

## Avaliação e Métricas
1. Garanta `test_data/` preenchido e o modelo `.zip` disponível em `DRL-TCC/`.
2. Rode a avaliação determinística:
   ```bash
   cd DRL-TCC
   uv run test_model.py
   ```
   São criados diretórios em `evaluation_results/deterministic/` com:
   - `metrics/`: JSON por dia (PnL final, média, inventário, drawdown, contagem de ações, probabilidade média de cada ação).
   - `plots/`: PNG com inventário, PnL, melhores preços e ações.
   - `logs/`: CSV com o log passo a passo do ambiente.
   - `daily_metrics.csv`: consolidação por dia.
3. Para agrupar os resultados:
   ```bash
   cd DRL-TCC
   uv run aggregate_metrics.py
   ```
   Gera:
   - `aggregated_by_ticker.csv`
   - `aggregated_by_liquidity.csv` (categorias liq_01_05, liq_06_10, liq_11_15, liq_outros).

## Demonstração Web
1. Aplique o modelo desejado em `MM_model_trained.zip` e garanta `test_data/` presente.
2. Inicie o Streamlit:
   ```bash
   cd DRL-TCC
   uv run streamlit run demo_app.py
   ```
3. Funcionalidades:
   - Seleção de tickers/dias (com agrupamento automático por ticker).
   - Execução determinística do modelo e exibição de métricas por dia.
   - Tabelas agregadas por dia, ticker e categoria de liquidez com opção de download CSV.
   - Visualização dos gráficos gerados durante a execução.
4. As saídas ficam armazenadas em `evaluation_results/webapp/` para consulta posterior.

## Monitoramento e Notebooks
- **TensorBoard**:
  ```bash
  cd DRL-TCC
  uv run tensorboard --logdir ./tb_logs --port 6006
  ```
- **Jupyter/Notebook** (com kernel isolado pelo uv):
  ```bash
  cd DRL-TCC
  uv run --with ipykernel python -m ipykernel install --user --name drl-uv --display-name "Python (uv)"
  uv run --with notebook jupyter notebook
  ```

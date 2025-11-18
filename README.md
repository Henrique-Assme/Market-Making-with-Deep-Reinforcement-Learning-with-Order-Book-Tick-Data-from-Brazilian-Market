# Market-Making-with-Deep-Reinforcement-Learning-with-Order-Book-Tick-Data-from-Brazilian-Market

Inicializar repo:
```
cd DRL-TCC && uv run
```

treinamento:
```
cd DRL-TCC && uv run train.py
```

teste:
```
cd DRL-TCC && uv run test_model.py
```

Agregar métricas e resultados:
```
cd DRL-TCC && uv run aggregate_metrics.py
```

Página de demonstração (Streamlit):
```
cd DRL-TCC && uv run streamlit run demo_app.py
```

TensorBoard:
```
cd DRL-TCC && uv run tensorboard --logdir ./tb_logs --port 6006
```

Run notebooks with uv kernel:
Setup the kernel
```
uv run --with ipykernel python -m ipykernel install --user --name drl-uv --display-name "Python (uv)" &&
uv run --with notebook jupyter notebook
``` 

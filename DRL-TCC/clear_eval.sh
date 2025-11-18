#!/bin/bash
# Limpa os resultados de avaliação sem remover as pastas

# Caminhos das pastas
dirs=(
  "evaluation_results/deterministic/logs"
  "evaluation_results/deterministic/metrics"
  "evaluation_results/deterministic/plots"
)

for dir in "${dirs[@]}"; do
  if [ -d "$dir" ]; then
    echo "Limpando $dir..."
    rm -rf "${dir:?}/"*
  else
    echo "Pasta não encontrada: $dir"
  fi
done

echo "Limpeza concluída."

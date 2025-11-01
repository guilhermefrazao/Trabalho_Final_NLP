import time
import math

# Tamanho da carga de trabalho — quanto maior, mais visível a diferença
N = 200_000_000  

print("Iniciando processamento pesado...")
inicio = time.time()

resultado = 0
for i in range(1, N):
    resultado += math.sqrt(i) * math.sin(i) * math.cos(i)

fim = time.time()

print(f"Resultado final: {resultado:.2f}")
print(f"Tempo total: {fim - inicio:.2f} segundos")

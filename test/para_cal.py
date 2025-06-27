import numpy as np

n = 100
N_values = np.arange(1000, 10001, 1000)

results = []
for N in N_values:
    N_u = int((1 / (n + 1)) * N + 2 * (n + 1))
    N_f = int((n / (n + 1)) * N)
    results.append((N, N_u, N_f))

print("N\t\tN_u\t\tN_f")
print("-" * 35)
for N, N_u, N_f in results:
    print(f"{N}\t\t{N_u}\t\t{N_f}")

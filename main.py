import numpy as np
import matplotlib.pyplot as plt

t = np.array([0, 2, 4, 6, 8, 10])         # Velocidade (100 ft/s)
f = np.array([0, 2.90, 14.8, 39.6, 74.3, 119])  # Força (100 lb)

# Monta matriz de Vandermonde para grau 5
A = np.vander(t, 6, increasing=True)

# Resolve o sistema A @ a = f
a = np.linalg.solve(A, f)

print("Coeficientes do polinômio de grau 5 (a0 até a5):")
print(a)

# Estimar força para t = 7.5 (750 ft/s)
t_val = 7.5
p_val = np.polyval(a[::-1], t_val)
print(f"\nForça estimada em 750 ft/s: {p_val:.4f} (em centenas de lb)")
print(f"Força estimada real: {p_val * 100:.4f} lb")

# Comparação com polinômio de grau 3
A3 = np.vander(t, 4, increasing=True)
a3 = np.linalg.lstsq(A3, f, rcond=None)[0]

# Geração de pontos para plotar
tt = np.linspace(0, 10, 200)
y_poly5 = np.polyval(a[::-1], tt)
y_poly3 = np.polyval(a3[::-1], tt)

# Plotagem
plt.figure(figsize=(8, 5))
plt.plot(t, f, 'ro', label='Pontos originais')
plt.plot(tt, y_poly5, 'b-', label='Polinômio grau 5')
plt.plot(tt, y_poly3, 'g--', label='Polinômio grau 3')
plt.xlabel('Velocidade (100 ft/s)')
plt.ylabel('Força (100 lb)')
plt.title('Interpolação Polinomial')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

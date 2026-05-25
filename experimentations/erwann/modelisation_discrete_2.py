import numpy as np
import matplotlib.pyplot as plt

# =========================
# Modèle
# =========================
N = 20
lv = 5.0
vmax = 15.0
dt = 0.01
T = 20.0
n_steps = int(T / dt)


def speed(d):
    """v(d) = vmax * (1 - lv/d), tronquée à 0."""
    d = max(d, lv)
    return max(0.0, vmax * (1.0 - lv / d))


def simulate(x0):
    """
    Simule le système :
      x_i' = vmax * (1 - lv / (x_{i+1} - x_i)),  i=1,...,N-1
      x_N' = vmax
    """
    x = x0.copy().astype(float)
    X_hist = np.zeros((n_steps + 1, N))
    X_hist[0] = x

    for n in range(n_steps):
        x_new = x.copy()

        for i in range(N - 1):
            d = x[i + 1] - x[i]
            v_i = speed(d)
            x_new[i] = x[i] + dt * v_i

        # véhicule de tête
        x_new[N - 1] = x[N - 1] + dt * vmax

        x = x_new
        X_hist[n + 1] = x

    return X_hist


# =========================
# Configurations initiales
# =========================

# 1) uniforme
x0_uniform = np.array([12.0 * i for i in range(N)], dtype=float)

# 2) petit bouchon au milieu
x0_bottleneck = np.array([12.0 * i for i in range(N)], dtype=float)
x0_bottleneck[7:13] -= np.linspace(0, 18, 6)  # resserre quelques voitures
x0_bottleneck = np.maximum.accumulate(x0_bottleneck)  # garde l'ordre

# 3) voitures plus serrées au départ
x0_dense = np.array([8.0 * i for i in range(N)], dtype=float)

# 4) perturbation sinusoïdale
x0_wave = np.array([12.0 * i for i in range(N)], dtype=float)
x0_wave += 3.0 * np.sin(2 * np.pi * np.arange(N) / N)

configs = {
    "Uniforme": x0_uniform,
    "Bouchon initial": x0_bottleneck,
    "Dense": x0_dense,
    "Onde sinusoïdale": x0_wave,
}

# =========================
# Tracé des trajectoires
# =========================
times = np.arange(n_steps + 1) * dt

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.ravel()

for ax, (title, x0) in zip(axes, configs.items()):
    X_hist = simulate(x0)

    for i in range(N):
        ax.plot(X_hist[:, i], times, lw=1)

    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Temps")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)

plt.suptitle(
    "Trajectoires des véhicules pour différentes positions initiales", fontsize=14
)
plt.tight_layout()
plt.show()

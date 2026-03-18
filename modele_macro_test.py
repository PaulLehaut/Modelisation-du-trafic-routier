import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paramètres du modèle
# =========================
L = 1.0               # longueur de la route
Nx = 200              # nombre de cellules
dx = L / Nx

T = 0.4               # temps final
vmax = 1.0            # vitesse maximale
rho_max = 1.0         # densité maximale

# CFL : dt <= dx / max|f'(rho)|
# ici f'(rho) = vmax * (1 - 2 rho/rho_max), donc max|f'| = vmax
CFL = 0.8
dt = CFL * dx / vmax
Nt = int(T / dt)

x = np.linspace(0, L, Nx, endpoint=False)

# =========================
# Flux de LWR
# =========================
def flux(rho):
    return rho * vmax * (1.0 - rho / rho_max)

# =========================
# Flux numérique de Godunov
# =========================
def godunov_flux(rhoL, rhoR):
    """
    Flux numérique de Godunov pour le flux concave f(rho)=rho*vmax*(1-rho/rho_max)
    """
    sigma = rho_max / 2.0  # densité critique

    if rhoL <= rhoR:
        # cas "rarefaction"
        if rhoL >= sigma:
            return flux(rhoL)
        elif rhoR <= sigma:
            return flux(rhoR)
        else:
            return flux(sigma)
    else:
        # cas "shock"
        s = (flux(rhoR) - flux(rhoL)) / (rhoR - rhoL)
        if s >= 0:
            return flux(rhoL)
        else:
            return flux(rhoR)

# version vectorisée
godunov_flux_vec = np.vectorize(godunov_flux)

# =========================
# Condition initiale
# =========================
rho = np.where((x > 0.2) & (x < 0.5), 0.8, 0.2)

# stockage pour carte espace-temps
rho_hist = np.zeros((Nt + 1, Nx))
rho_hist[0] = rho.copy()

# =========================
# Boucle en temps
# =========================
for n in range(Nt):
    # bords périodiques
    rhoL = rho
    rhoR = np.roll(rho, -1)

    # flux aux interfaces j+1/2
    F = godunov_flux_vec(rhoL, rhoR)

    # mise à jour volumes finis
    rho = rho - (dt / dx) * (F - np.roll(F, 1))

    rho_hist[n + 1] = rho.copy()

# =========================
# Tracé densité finale
# =========================
plt.figure(figsize=(8, 4))
plt.plot(x, rho_hist[0], label="t = 0")
plt.plot(x, rho_hist[Nt//3], label=f"t ≈ {Nt//3 * dt:.2f}")
plt.plot(x, rho_hist[2*Nt//3], label=f"t ≈ {2*Nt//3 * dt:.2f}")
plt.plot(x, rho_hist[-1], label=f"t = {Nt * dt:.2f}")
plt.xlabel("Position x")
plt.ylabel("Densité ρ")
plt.title("Évolution de la densité - modèle LWR")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# Carte espace-temps
# =========================
plt.figure(figsize=(8, 5))
plt.imshow(
    rho_hist,
    aspect="auto",
    origin="lower",
    extent=[0, L, 0, Nt * dt]
)
plt.colorbar(label="Densité ρ")
plt.xlabel("Position x")
plt.ylabel("Temps t")
plt.title("Carte espace-temps de la densité")
plt.tight_layout()
plt.show()
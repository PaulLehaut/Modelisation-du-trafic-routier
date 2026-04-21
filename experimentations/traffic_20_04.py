import numpy as np
import matplotlib.pyplot as plt

#######################################################################
#                       Constants
#######################################################################
T = 0.1               # hours
V_MAX = 50.0          # km/h
L_VEHICLE = 0.005     # km = 5 meters
RHO_MAX = 1.0 / L_VEHICLE

# Domaine utilisé pour définir la donnée initiale continue
X_DOMAIN = [0.0, 10.0]   # km

#######################################################################
#                       Initial continuous density
#######################################################################
def rho_initial(x, profile="shock_wave"):
    """
    Retourne la densité initiale continue rho_0(x).

    @n: les deux modèles partent de la même donnée initiale continue.
    Le micro ne doit pas être construit à partir du macro, ni l'inverse.
    """
    x = np.asarray(x)
    rho = np.zeros_like(x, dtype=float)

    support = (x >= X_DOMAIN[0]) & (x <= X_DOMAIN[1])

    if profile == "shock_wave":
        shock_pos = 4.0
        rho[support & (x < shock_pos)] = 0.75 * RHO_MAX
        rho[support & (x >= shock_pos)] = 0.15 * RHO_MAX

    elif profile == "rarefaction_wave":
        # densité lisse décroissante sur le support
        rho[support] = 0.15 * RHO_MAX + 0.55 * RHO_MAX * np.exp(-(x[support] - X_DOMAIN[0]) / 2.0)

    elif profile == "stop_and_go_wave":
        # oscillation positive sur le support
        base = 0.45 * RHO_MAX
        amp = 0.30 * RHO_MAX
        rho[support] = base + amp * np.sin(3.0 * np.pi * (x[support] - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0]))

    else:
        raise ValueError(f"Profil inconnu: {profile}")

    return rho

#######################################################################
#                       Greenshields
#######################################################################
def compute_speed(rho):
    """
    Greenshields:
        v(rho) = V_MAX * (1 - rho / RHO_MAX)
    """
    return V_MAX * (1.0 - rho / RHO_MAX)


def flux(rho):
    return rho * compute_speed(rho)

#######################################################################
#                       Godunov flux
#######################################################################
def compute_godunov_flux(rho_l, rho_r):
    """
    Godunov flux for the concave Greenshields flux.
    """
    rho_c = RHO_MAX / 2.0
    q_max = flux(rho_c)

    demand = flux(rho_l) if rho_l <= rho_c else q_max
    supply = q_max if rho_r <= rho_c else flux(rho_r)

    return min(demand, supply)

#######################################################################
#               Initial positions from the same continuous density
#######################################################################
def build_initial_positions_from_density(N, profile="shock_wave", n_ref=20001):
    """
    @n: positions initiales construites par quantiles de masse.

    On impose :
        ∫_{x_i}^{x_{i+1}} rho_0(x) dx = M / N

    Ce choix est plus cohérent avec FtL que d'échantillonner arbitrairement
    la densité ou de reconstruire une autre donnée.
    """
    x_ref = np.linspace(X_DOMAIN[0], X_DOMAIN[1], n_ref)
    rho_ref = rho_initial(x_ref, profile)

    dx_ref = x_ref[1] - x_ref[0]
    cdf_ref = np.zeros_like(x_ref)
    cdf_ref[1:] = np.cumsum(0.5 * (rho_ref[:-1] + rho_ref[1:]) * dx_ref)

    total_mass = cdf_ref[-1]
    mass_particle = total_mass / N

    # x_0,...,x_N : N intervalles de même masse
    mass_levels = np.linspace(0.0, total_mass, N + 1)
    X_0 = np.interp(mass_levels, cdf_ref, x_ref)

    return X_0, mass_particle, total_mass

#######################################################################
#               FtL microscopic simulation
#######################################################################
def simulate_ftl(X_0, mass_particle, dt, nt):
    """
    FtL explicite Euler.

    X_tab.shape = (N+1, nt)
    - les N premiers points sont les followers
    - le dernier point est le leader, à vitesse V_MAX

    @n: pas de sécurité numérique artificielle ici.
    On suppose que le pas de temps choisi respecte la stabilité
    et que l'ordre des véhicules est conservé.
    """
    n_veh = len(X_0)
    X_tab = np.zeros((n_veh, nt))
    X_tab[:, 0] = X_0.copy()

    for t in range(nt - 1):
        v = np.zeros(n_veh)

        for i in range(n_veh - 1):
            gap = X_tab[i + 1, t] - X_tab[i, t]
            rho_loc = mass_particle / gap
            v[i] = compute_speed(rho_loc)

        v[-1] = V_MAX

        X_tab[:, t + 1] = X_tab[:, t] + dt * v

    return X_tab

#######################################################################
#               Eulerian density reconstruction from micro
#######################################################################
def reconstruct_eulerian_density(X_tab, x_edges, dx, mass_particle):
    """
    Reconstruction conservative de la densité eulérienne à partir des positions FtL.

    @n: on reconstruit une densité par moyennes de cellules
    en répartissant la masse de chaque intervalle [x_i, x_{i+1}]
    sur les cellules qu'il recouvre.
    """
    n_veh, nt = X_tab.shape
    nx = len(x_edges) - 1
    rho_euler = np.zeros((nt, nx))

    for t in range(nt):
        for i in range(n_veh - 1):
            left = X_tab[i, t]
            right = X_tab[i + 1, t]

            if right <= left:
                continue

            rho_i = mass_particle / (right - left)

            j_left = np.searchsorted(x_edges, left, side="right") - 1
            j_right = np.searchsorted(x_edges, right, side="left")

            j_left = max(0, j_left)
            j_right = min(nx - 1, j_right)

            for j in range(j_left, j_right + 1):
                cell_left = x_edges[j]
                cell_right = x_edges[j + 1]
                overlap = max(0.0, min(cell_right, right) - max(cell_left, left))

                if overlap > 0:
                    rho_euler[t, j] += rho_i * overlap / dx

    return rho_euler

#######################################################################
#               LWR macroscopic simulation
#######################################################################
def simulate_godunov(rho_0_grid, dt, nt, dx):
    """
    Schéma de Godunov pour:
        rho_t + f(rho)_x = 0
    """
    nx = len(rho_0_grid)
    rho_tab = np.zeros((nt, nx))
    rho = rho_0_grid.copy()
    rho_tab[0, :] = rho

    for ts in range(1, nt):
        fluxes = np.zeros(nx + 1)

        # @n: bords pris comme extérieur vide
        fluxes[0] = compute_godunov_flux(0.0, rho[0])
        fluxes[-1] = compute_godunov_flux(rho[-1], 0.0)

        for j in range(1, nx):
            fluxes[j] = compute_godunov_flux(rho[j - 1], rho[j])

        rho_new = np.zeros(nx)
        for j in range(nx):
            rho_new[j] = rho[j] - (dt / dx) * (fluxes[j + 1] - fluxes[j])

        rho = rho_new
        rho_tab[ts, :] = rho

    return rho_tab

#######################################################################
#               Metrics
#######################################################################
def compute_metrics(rho_micro, rho_macro, dx, dt):
    """
    Erreur L1 espace-temps et erreur relative L1.
    """
    diff = np.abs(rho_micro - rho_macro)
    l1 = np.sum(diff) * dx * dt

    ref = np.sum(np.abs(rho_macro)) * dx * dt
    rel_l1 = l1 / ref if ref > 0 else 0.0

    return l1, rel_l1

#######################################################################
#               Plots
#######################################################################
def plot_trajectories(X_tab, t_values, title):
    """
    @n: toutes les trajectoires sont tracées finement,
    seul le leader est mis en évidence pour éviter une légende lourde.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(X_tab.shape[0] - 1):
        ax.plot(X_tab[i, :], t_values, color="0.6", linewidth=0.4, alpha=0.25)

    ax.plot(X_tab[-1, :], t_values, color="black", linewidth=2.0, label="Leader")
    ax.set_xlabel("Position (km)")
    ax.set_ylabel("Temps (h)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_density_comparison(rho_micro, rho_macro, x_edges, t_values, profile, N, l1, rel_l1):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    x_min = x_edges[0]
    x_max = x_edges[-1]

    im0 = axes[0].imshow(
        rho_micro,
        aspect="auto",
        origin="lower",
        extent=[x_min, x_max, t_values[0], t_values[-1]],
        cmap="jet",
        vmin=0.0,
        vmax=RHO_MAX,
    )
    fig.colorbar(im0, ax=axes[0], label="Density (vehicles/km)")
    axes[0].set_title(f"FtL reconstructed density, N={N}")
    axes[0].set_ylabel("Temps (h)")

    im1 = axes[1].imshow(
        rho_macro,
        aspect="auto",
        origin="lower",
        extent=[x_min, x_max, t_values[0], t_values[-1]],
        cmap="jet",
        vmin=0.0,
        vmax=RHO_MAX,
    )
    fig.colorbar(im1, ax=axes[1], label="Density (vehicles/km)")
    axes[1].set_title(f"LWR Godunov density, N={N}")
    axes[1].set_xlabel("Position (km)")
    axes[1].set_ylabel("Temps (h)")

    plt.suptitle(f"{profile} | L1={l1:.3e} | rel-L1={rel_l1:.3e}", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_convergence(n_tab, err_tab):
    plt.figure(figsize=(8, 5))
    plt.plot(n_tab, err_tab, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Nombre de véhicules N")
    plt.ylabel("Erreur L1 espace-temps")
    plt.title("Convergence FtL → LWR")
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.tight_layout()
    plt.show()

#######################################################################
#               One experiment
#######################################################################
def run_experiment(N, profile="shock_wave", nx=1000, cfl_factor=0.9):
    """
    Exécute une comparaison FtL / LWR pour un N donné.

    @n: comparaison honnête =
    - même donnée initiale continue,
    - positions micro dérivées de cette donnée,
    - même domaine d'observation,
    - même grille spatiale pour les deux modèles,
    - même pas de temps.
    """
    # micro initial state
    X_0, mass_particle, total_mass = build_initial_positions_from_density(N, profile=profile)

    # domaine naturel d'observation
    x_min = X_0[0]
    x_max = X_0[-1] + V_MAX * T   # leader initial + propagation maximale
    road_length = x_max - x_min

    # grille eulérienne
    dx = road_length / nx
    x_centers = x_min + (np.arange(nx) + 0.5) * dx
    x_edges = x_min + np.arange(nx + 1) * dx

    # pas de temps CFL
    min_gap0 = np.min(np.diff(X_0))
    dt = cfl_factor * min(dx / V_MAX, min_gap0 / V_MAX)
    nt = int(np.ceil(T / dt)) + 1
    dt = T / (nt - 1)
    t_values = np.arange(nt) * dt

    # micro simulation
    X_tab = simulate_ftl(X_0, mass_particle, dt, nt)

    # micro -> eulerian density
    rho_micro = reconstruct_eulerian_density(X_tab, x_edges, dx, mass_particle)

    # macro initial data on the same grid
    rho_0_grid = rho_initial(x_centers, profile)

    # macro simulation
    rho_macro = simulate_godunov(rho_0_grid, dt, nt, dx)

    # metrics
    l1, rel_l1 = compute_metrics(rho_micro, rho_macro, dx, dt)

    return {
        "N": N,
        "profile": profile,
        "X_0": X_0,
        "X_tab": X_tab,
        "x_edges": x_edges,
        "x_centers": x_centers,
        "t_values": t_values,
        "rho_micro": rho_micro,
        "rho_macro": rho_macro,
        "dt": dt,
        "dx": dx,
        "nt": nt,
        "l1": l1,
        "rel_l1": rel_l1,
    }

#######################################################################
#               Main
#######################################################################
if __name__ == "__main__":

    profile = "stop_and_go_wave"  # "shock_wave", "rarefaction_wave", "stop_and_go_wave"

    # Tests de convergence
    n_tab = [10, 50, 100, 200, 500, 1000, 2000]
    err_tab = []

    # Cas représentatif pour l'affichage
    n_show = 1000
    show_data = None

    for N in n_tab:
        data = run_experiment(N, profile=profile, nx=1000, cfl_factor=0.9)

        err_tab.append(data["l1"])
        print(f"N = {N:4d} | L1 space-time error = {data['l1']:.6e} | rel-L1 = {data['rel_l1']:.6e}")

        if N == n_show:
            show_data = data

    # Trajectoires FtL
    if show_data is not None:
        plot_trajectories(
            show_data["X_tab"],
            show_data["t_values"],
            title=f"Trajectoires FtL, N = {n_show}"
        )

        # Densités FtL / LWR
        plot_density_comparison(
            show_data["rho_micro"],
            show_data["rho_macro"],
            show_data["x_edges"],
            show_data["t_values"],
            profile,
            n_show,
            show_data["l1"],
            show_data["rel_l1"]
        )

    # Convergence
    plot_convergence(n_tab, err_tab)
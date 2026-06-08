# flake8: noqa

import matplotlib.pyplot as plt
import numpy as np
import random

#######################################################################
#                       PARAMÈTRES GLOBAUX
#######################################################################

"""
- T is a time in hour 
- v_max is a velocity in km/h 
- l is a length in km
- rho_max is a density in vehicle/km
"""
N = 20
T = 0.1  # 6 minutes
v_max = 120
l = 0.005  # 5 meters
l_init = 10 * l  # distance de réfénce pour initialisation des positions
rho_max = 1 / l

#######################################################################
#                       BLOC 1 : MODÈLES DE VITESSE
#######################################################################


def speed_greenshields(v_max, rho_max, rho):
    """Modèle linéaire classique"""
    speed = v_max * (1 - rho / rho_max)
    return max(0, min(speed, v_max))


## AUTRES VERSIONS POSSIBLES (VITESSE) :
# def speed_quadratic(v_max, rho_max, rho): ...

#######################################################################
#     BLOC 2 : MODÈLES DE DEFINITION DE RHO (Dans modèle discret)
#######################################################################


def discrete_rho_func(l, distance, rho_max):
    return l / distance if distance > 0 else rho_max


## AUTRES VERSIONS POSSIBLES

#######################################################################
#                       BLOC 3 : MODÈLES DE FLUX (POUR CONTINU)
#######################################################################


def flux_standard(rho, speed_func, v_max, rho_max):
    """Flux classique = Densité * Vitesse locale"""
    return rho * speed_func(v_max, rho_max, rho)


## AUTRES VERSIONS POSSIBLES (FLUX) :
# def flux_specifique(rho, speed_func, v_max, rho_max): ...


#######################################################################
#                       BLOC 4 : CONDITIONS INITIALES
#######################################################################


# --- POUR LE MODÈLE DISCRET ---
def init_pos_uniform(N, l_ref):
    """1) Uniforme"""
    return np.array([l_ref * i for i in range(N)], dtype=float)


def init_pos_bottleneck(N, l_ref, start_ratio=0.35, end_ratio=0.65, comp_factor=1.5):
    """2) Petit bouchon au milieu"""
    x0 = np.array([l_ref * i for i in range(N)], dtype=float)
    start = int(N * start_ratio)
    end = int(N * end_ratio)
    n_bouchon = end - start
    max_comp = l_ref * comp_factor
    x0[start:end] -= np.linspace(0, max_comp, n_bouchon)
    return np.maximum.accumulate(x0)


def init_pos_dense(N, l_ref, density_factor=0.5):
    """3) Voitures plus serrées au départ"""
    return np.array([(l_ref * density_factor) * i for i in range(N)], dtype=float)


def init_pos_wave(N, l_ref, amplitude_factor=0.25):
    """4) Perturbation sinusoïdale"""
    x0 = np.array([l_ref * i for i in range(N)], dtype=float)
    amplitude = l_ref * amplitude_factor
    x0 += amplitude * np.sin(2 * np.pi * np.arange(N) / N)
    return x0


## AUTRES VERSIONS POSSIBLES (POSITIONS DISCRÈTES) :
# def init_pos_bouchon_localise(N, l): ...


# --- Pour le modèle continu ---
def init_rho_two_jams(nx, rho_max, x_tab):
    """Deux bouchons distincts"""
    rho = np.ones(nx) * (0.2 * rho_max)
    rho[(x_tab >= 1) & (x_tab <= 2)] = 0.8 * rho_max
    rho[(x_tab >= 3) & (x_tab <= 4)] = 0.8 * rho_max
    return rho


def init_rho_single_jam(nx, rho_max, x_tab):
    """Un seul gros bouchon au centre"""
    rho = np.ones(nx) * (0.1 * rho_max)
    rho[(x_tab >= 2) & (x_tab <= 3)] = 0.9 * rho_max
    return rho


# équivalent au bottleneck discret
def init_rho_bottleneck(nx, rho_max, x_tab):
    rho = np.zeros(nx)
    L_init = N * l_init
    rho[x_tab <= L_init] = 1 / l_init
    start_x, end_x = L_init * 0.35, L_init * 0.65
    rho[(x_tab >= start_x) & (x_tab <= end_x)] = rho_max * 0.8
    return rho


## AUTRES VERSIONS POSSIBLES (DENSITÉS CONTINUES) :
# def init_rho_empty_road(nx, rho_max, x_tab): ...


#######################################################################
#                       BLOC 5 : MODÈLE DISCRET
#######################################################################


def discrete_model(
    N,
    time_actualisation,
    l,
    rho_func=discrete_rho_func,
    speed_func=speed_greenshields,
    init_pos_func=init_pos_uniform,
    ax=None,
):
    time_steps = int(T / time_actualisation)
    t_tab = np.linspace(0, T, time_steps)
    x_tab = np.zeros((N, time_steps))
    v_tab = np.zeros((N, time_steps))

    x_tab[:, 0] = init_pos_func(N, l_init)
    rho_tab = np.zeros((N - 1, time_steps))

    for i in range(N):
        if i == N - 1:
            v_tab[i][0] = v_max
        else:
            distance = x_tab[i + 1][0] - x_tab[i][0]
            density = rho_func(l, distance, rho_max)
            v_tab[i][0] = speed_func(v_max, rho_max, density)
            rho_tab[i][0] = density

    t = 1
    while t < time_steps:
        for i in range(N):
            x_tab[i][t] = x_tab[i][t - 1] + v_tab[i][t - 1] * time_actualisation

        for i in range(N):
            if i == N - 1:
                v_tab[i][t] = v_max
            else:
                distance = x_tab[i + 1][t] - x_tab[i][t]
                density = rho_func(l, distance, rho_max)
                v_tab[i][t] = speed_func(v_max, rho_max, density)
                rho_tab[i][t] = density
        t += 1

    # --- GESTION DE L'AFFICHAGE ---
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True

    # 2. Affichage du tableau rho en arrière-plan via un scatter plot
    x_mid = (x_tab[:-1, :] + x_tab[1:, :]) / 2
    T_mat = np.tile(t_tab, (N - 1, 1))

    # On ajoute vmin et vmax pour forcer l'échelle de couleurs
    sc = ax.scatter(
        x_mid,
        T_mat,
        c=rho_tab,
        cmap="jet",
        vmin=0,
        vmax=rho_max,
        s=40,
        alpha=0.9,
        edgecolors="none",
        marker="s",
    )

    # AJOUT DE LA COLORBAR SPÉCIFIQUE AU GRAPHE DISCRET
    # On utilise la figure associée à l'axe pour la placer correctement
    fig = ax.figure
    fig.colorbar(sc, ax=ax, label="Densité (veh/km)")

    ax.set_title(
        f"Discret | Init: {init_pos_func.__name__} | Vit: {speed_func.__name__}"
    )
    ax.set_xlabel("Position (km)")
    ax.set_ylabel("Temps (h)")
    ax.grid(True, alpha=0.5)

    if show_plot:
        plt.show()


#######################################################################
#                       BLOC 6 : MODÈLE CONTINU
#######################################################################


def continuous_model(
    L=5,
    nx=200,
    speed_func=speed_greenshields,
    flux_func=flux_standard,
    init_cond_func=init_rho_two_jams,
    ax=None,
):
    dx = L / nx
    dt = 0.9 * dx / v_max
    nt = int(T / dt)
    x_tab = np.linspace(0, L, nx)

    rho = init_cond_func(nx, rho_max, x_tab)
    rho_tab = np.zeros((nt, nx))
    rho_tab[0, :] = rho

    for ts in range(1, nt):
        rho_steps = np.zeros(nx)
        rho_steps[0] = rho[0]
        rho_steps[-1] = rho[-1]

        for xs in range(1, nx - 1):
            flux_avant = flux_func(rho[xs - 1], speed_func, v_max, rho_max)
            flux_apres = flux_func(rho[xs + 1], speed_func, v_max, rho_max)
            rho_steps[xs] = 0.5 * (rho[xs + 1] + rho[xs - 1]) - dt / (2 * dx) * (
                flux_apres - flux_avant
            )

        rho = rho_steps.copy()
        rho_tab[ts, :] = rho

    # --- GESTION DE L'AFFICHAGE ---
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True
    else:
        fig = ax.figure

    im = ax.imshow(
        rho_tab,
        aspect="auto",
        origin="lower",
        extent=[0, L, 0, T],
        cmap="jet",
        vmin=0,
        vmax=rho_max,
    )

    if show_plot:
        fig.colorbar(im, ax=ax, label="Densité (veh/km)")

    ax.set_title(
        f"Continu | Init: {init_cond_func.__name__} | Vit: {speed_func.__name__}"
    )
    ax.set_xlabel("Position (km)")
    ax.set_ylabel("Temps (h)")

    if show_plot:
        plt.show()


#######################################################################
#                       BLOC 6 : COMPARATEURS (RUNNERS)
#######################################################################


def run_rho_comparison():
    """Compare directement la densité (rho) entre le modèle discret et continu"""
    print("\n--- Lancement du comparateur DE CONVERGENCE RHO ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Comparaison de la Densité : Discret vs Continu", fontsize=16, fontweight="bold"
    )

    # L_total prend en compte la position initiale ET la distance parcourue pendant T
    L_total = (N * l_init) + (v_max * T)

    # Lancement du modèle discret
    discrete_model(
        N=N,
        time_actualisation=5 / 3600,
        l=l,
        rho_func=discrete_rho_func,
        speed_func=speed_greenshields,
        init_pos_func=init_pos_bottleneck,
        ax=axes[0],
    )

    # Lancement du modèle continu
    continuous_model(
        L=L_total,
        nx=100,
        speed_func=speed_greenshields,
        flux_func=flux_standard,
        init_cond_func=init_rho_bottleneck,
        ax=axes[1],
    )

    # Synchronisation des colorbars
    im = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=rho_max))
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        label="Densité (veh/km)",
        orientation="vertical",
        fraction=0.02,
        pad=0.04,
    )

    plt.show()


def run_continuous_comparison():
    """Lance une grille de comparaison pour le modèle continu"""
    print("\n--- Lancement du comparateur CONTINU ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Comparaison des Modèles Continus", fontsize=16, fontweight="bold")

    continuous_model(
        speed_func=speed_greenshields, init_cond_func=init_rho_two_jams, ax=axes[0, 0]
    )
    plt.tight_layout(pad=3.0)
    plt.show()


def run_discrete_comparison():
    """Lance une grille de comparaison pour le modèle discret"""
    print("\n--- Lancement du comparateur DISCRET ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 6))
    fig.suptitle("Comparaison des Modèles Discrets", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    discrete_model(
        N=N,
        time_actualisation=5 / 3600,
        l=l,
        speed_func=speed_greenshields,
        init_pos_func=init_pos_uniform,
        ax=axes[0],
    )
    discrete_model(
        N=N,
        time_actualisation=5 / 3600,
        l=l,
        speed_func=speed_greenshields,
        init_pos_func=init_pos_bottleneck,
        ax=axes[1],
    )
    discrete_model(
        N=N,
        time_actualisation=5 / 3600,
        l=l,
        speed_func=speed_greenshields,
        init_pos_func=init_pos_dense,
        ax=axes[2],
    )
    discrete_model(
        N=N,
        time_actualisation=5 / 3600,
        l=l,
        speed_func=speed_greenshields,
        init_pos_func=init_pos_wave,
        ax=axes[3],
    )

    plt.tight_layout(pad=3.0)
    plt.show()


if __name__ == "__main__":
    # Comparaison des modèles
    run_discrete_comparison()
    # run_continuous_comparison()
    # run_rho_comparison()

    # Voir un seul modèle à la fois
    # continuous_model(speed_func=speed_greenshields, init_cond_func=init_rho_single_jam)

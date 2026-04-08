# flake8: noqa

import matplotlib.pyplot as plt
import numpy as np
import random

#######################################################################
#                       PARAMÈTRES GLOBAUX               
#######################################################################

'''
- N is a number of véhicles
- T is a time in hour 
- v_max is a velocity in km/h 
- l is a length in km
- rho_max is a density in vehicle/km
'''
N = 20
T = 0.2
v_max = 50
l = 0.005
l_init = 10*l
rho_max = 1 / l
L = 5
nx=200
time_actualisation = 5/3600

#######################################################################
#                       BLOC 1 : MODÈLES DE VITESSE               
#######################################################################

def speed_greenshields(v_max, rho_max, rho):
    """ Modèle linéaire classique """
    speed = v_max * (1 - rho / rho_max)
    return max(0, min(speed, v_max))

## AUTRES VERSIONS POSSIBLES (VITESSE) :
# def speed_quadratic(v_max, rho_max, rho): ...

#######################################################################
#     BLOC 2 : MODÈLES DE DEFINITION DE RHO (Dans modèle discret)               
#######################################################################

def discrete_rho_func(l, distance, rho_max):
    return 1 / distance if distance > 0 else rho_max

## AUTRES VERSIONS POSSIBLES

#######################################################################
#                       BLOC 3 : MODÈLES DE FLUX (POUR CONTINU)               
#######################################################################

def flux_lax_friedrichs(rho_L, rho_R, dx, dt, speed_func, v_max, rho_max):
    """ Flux numérique de Lax-Friedrichs évalué à l'interface """
    flux_L = rho_L * speed_func(v_max, rho_max, rho_L)
    flux_R = rho_R * speed_func(v_max, rho_max, rho_R)
    # Moyenne des flux + terme de viscosité numérique (diffusion)
    return 0.5 * (flux_L + flux_R) - (dx / (2 * dt)) * (rho_R - rho_L)

def flux_godunov(rho_L, rho_R, dx, dt, speed_func, v_max, rho_max):
    """ Flux numérique de Godunov (Offre / Demande) évalué à l'interface """
    rho_c = rho_max / 2.0  # Densité critique pour Greenshields
    
    # Calcul du flux physique local
    def f(rho):
        return rho * speed_func(v_max, rho_max, rho)
    
    # 1. Calcul de la Demande (amont)
    demande = f(rho_L) if rho_L <= rho_c else f(rho_c)
        
    # 2. Calcul de l'Offre (aval)
    offre = f(rho_c) if rho_R <= rho_c else f(rho_R)
        
    # Le flux traversant est le minimum des deux
    return min(demande, offre)

## AUTRES VERSIONS POSSIBLES (FLUX) :
# def flux_roe(rho_L, rho_R, dx, dt, speed_func, v_max, rho_max): ...


#######################################################################
#                       BLOC 4 : CONDITIONS INITIALES               
#######################################################################

# --- POUR LE MODÈLE DISCRET ---
def init_pos_uniform(N, l_ref):
    """ 1) Uniforme """
    return np.array([l_ref * i for i in range(N)], dtype=float)

def init_pos_bottleneck(N, l_ref, start_ratio=0.35, end_ratio=0.65, comp_factor=1.5):
    """ 2) Petit bouchon au milieu """
    x0 = np.array([l_ref * i for i in range(N)], dtype=float)
    start = int(N * start_ratio)
    end = int(N * end_ratio)
    n_bouchon = end - start
    max_comp = l_ref * comp_factor
    x0[start:end] -= np.linspace(0, max_comp, n_bouchon) 
    return np.maximum.accumulate(x0)

def init_pos_dense(N, l_ref, density_factor=0.5):
    """ 3) Voitures plus serrées au départ """
    return np.array([(l_ref * density_factor) * i for i in range(N)], dtype=float)

def init_pos_wave(N, l_ref, amplitude_factor=0.25):
    """ 4) Perturbation sinusoïdale """
    x0 = np.array([l_ref * i for i in range(N)], dtype=float)
    amplitude = l_ref * amplitude_factor
    x0 += amplitude * np.sin(2 * np.pi * np.arange(N) / N)
    return x0

## AUTRES VERSIONS POSSIBLES (POSITIONS DISCRÈTES) :
# def init_pos_bouchon_localise(N, l): ...


# --- Pour le modèle continu ---
def init_rho_two_jams(nx, rho_max, x_tab):
    """ Deux bouchons distincts """
    rho = np.ones(nx) * (0.2 * rho_max)
    rho[(x_tab >= 1) & (x_tab <= 2)] = 0.8 * rho_max
    rho[(x_tab >= 3) & (x_tab <= 4)] = 0.8 * rho_max
    return rho

def init_rho_single_jam(nx, rho_max, x_tab):
    """ Un seul gros bouchon au centre """
    rho = np.ones(nx) * (0.1 * rho_max)
    rho[(x_tab >= 2) & (x_tab <= 3)] = 0.9 * rho_max
    return rho

# équivalent au bottleneck discret
def init_rho_bottleneck(nx, rho_max, x_tab):
    rho = np.zeros(nx)
    L_init = N * l_init
    rho[x_tab <= L_init] = 1 / l_init 
    start_x, end_x = L_init * 0.35, L_init * 0.65 
    
    # On aligne la densité continue sur le bouchon discret (distance min = 0.035 km)
    rho_bouchon = 1 / 0.035 
    rho[(x_tab >= start_x) & (x_tab <= end_x)] = rho_bouchon
    
    return rho

## AUTRES VERSIONS POSSIBLES (DENSITÉS CONTINUES) :
# def init_rho_empty_road(nx, rho_max, x_tab): ...


#######################################################################
#                       BLOC 5 : MODÈLE DISCRET               
#######################################################################

def discrete_model(N, time_actualisation, l, rho_func=discrete_rho_func, speed_func=speed_greenshields, init_pos_func=init_pos_uniform, ax=None, plot_type='trajectories'):
    time_steps = int(T / time_actualisation)
    t_tab = np.linspace(0, T, time_steps)
    x_tab = np.zeros((N, time_steps))
    v_tab = np.zeros((N, time_steps))

    x_tab[:, 0] = init_pos_func(N, l_init)
    rho_tab = np.zeros((N-1, time_steps))
    
    for i in range(N):
        if i == N - 1:
            v_tab[i][0] = v_max
        else:
            distance = x_tab[i+1][0] - x_tab[i][0]
            density = rho_func(l, distance, rho_max)
            v_tab[i][0] = speed_func(v_max, rho_max, density)
            rho_tab[i][0] = density

    t = 1
    while t < time_steps :
        for i in range(N):
            x_tab[i][t] = x_tab[i][t-1] + v_tab[i][t-1] * time_actualisation

        for i in range(N):
            if i == N - 1:  
                v_tab[i][t] = v_max
            else:
                distance = x_tab[i+1][t] - x_tab[i][t]
                density = rho_func(l, distance, rho_max)
                v_tab[i][t] = speed_func(v_max, rho_max, density)
                rho_tab[i][t] = density
        t += 1

    # --- GESTION DE L'AFFICHAGE ---
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True

    if plot_type == 'density':
        # Affichage du tableau rho en arrière-plan via un scatter plot
        x_mid = (x_tab[:-1, :] + x_tab[1:, :]) / 2
        T_mat = np.tile(t_tab, (N-1, 1))
        
        # On ajoute vmin et vmax pour forcer l'échelle de couleurs
        sc = ax.scatter(x_mid, T_mat, c=rho_tab, cmap='jet', vmin=0, vmax=rho_max, s=40, alpha=0.9, edgecolors='none', marker='s')

        # AJOUT DE LA COLORBAR SPÉCIFIQUE AU GRAPHE DISCRET
        fig = ax.figure 
        fig.colorbar(sc, ax=ax, label='Densité (veh/km)')
    else:
        # Affichage des trajectoires (Diagramme x-t) avec couleurs différentes
        # Utilisation de la palette 'tab20' qui génère 20 couleurs distinctes très lisibles
        cmap = plt.get_cmap('tab20') 
        for i in range(N):
            # x_tab en abscisse (Position) et t_tab en ordonnée (Temps) pour matcher le continu
            ax.plot(x_tab[i, :], t_tab, color=cmap(i % 20), linewidth=1.5, label=f'Véhicule {i+1}')
            
        # Ajout de la légende (taille réduite et sur 2 colonnes pour ne pas masquer les courbes)
        # ax.legend(fontsize='x-small', loc='best', ncol=2)

    ax.set_title(f'Discret | Init: {init_pos_func.__name__} | Vit: {speed_func.__name__}')
    ax.set_xlabel('Position (km)')
    ax.set_ylabel('Temps (h)')
    ax.grid(True, alpha=0.5) 

    if show_plot:
        plt.show()

    return t_tab, x_tab

#######################################################################
#                       BLOC 6 : MODÈLE CONTINU               
#######################################################################

def continuous_model(L=5, nx=200, speed_func=speed_greenshields, flux_func=flux_lax_friedrichs, init_cond_func=init_rho_two_jams, ax=None):
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

        # BOUCLE ULTRA MODULAIRE : On interroge simplement le flux aux interfaces
        for xs in range(1, nx - 1):
            flux_avant = flux_func(rho[xs - 1], rho[xs], dx, dt, speed_func, v_max, rho_max)
            flux_apres = flux_func(rho[xs], rho[xs + 1], dx, dt, speed_func, v_max, rho_max)
            
            # Forme conservative standard pour les volumes finis
            rho_steps[xs] = rho[xs] - (dt / dx) * (flux_apres - flux_avant)

        rho = rho_steps.copy()
        rho_tab[ts, :] = rho
        
    # --- GESTION DE L'AFFICHAGE ---
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True
    else:
        fig = ax.figure

    im = ax.imshow(rho_tab, aspect='auto', origin='lower', extent=[0, L, 0, T], cmap='jet', vmin=0, vmax=rho_max)
    
    if show_plot:
        fig.colorbar(im, ax=ax, label='Densité (veh/km)')
        
    ax.set_title(f'Continu | Init: {init_cond_func.__name__} | Vit: {speed_func.__name__}')
    ax.set_xlabel('Position (km)')
    ax.set_ylabel('Temps (h)')

    if show_plot:
        plt.show()

    t_tab = np.linspace(0, T, nt)
    return t_tab, rho_tab


#######################################################################
#                       BLOC 7 : COMPARATEURS (RUNNERS)               
#######################################################################

def run_rho_comparison():
    """ Compare directement la densité (rho) entre le modèle discret et continu """
    print("\n--- Lancement du comparateur DE CONVERGENCE RHO ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Comparaison de la Densité : Discret vs Continu", fontsize=16, fontweight='bold')

    # L_total prend en compte la position initiale ET la distance parcourue pendant T
    L_total = (N * l_init) + (v_max * T)

    # Lancement du modèle discret AVEC LE PLOT TYPE DENSITY
    discrete_model(N=N, time_actualisation=time_actualisation,l=l,rho_func=discrete_rho_func, speed_func=speed_greenshields, init_pos_func=init_pos_bottleneck, ax=axes[0], plot_type='density')

    # Lancement du modèle continu 
    continuous_model(L=L_total, nx=nx, speed_func=speed_greenshields, flux_func=flux_lax_friedrichs, init_cond_func=init_rho_bottleneck, ax=axes[1])

    # Synchronisation des colorbars
    im = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=rho_max))
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Densité (veh/km)', orientation='vertical', fraction=0.02, pad=0.04)

    plt.show()

def run_bound_comparison():
    """ 
    Évalue et trace la distance inter-véhicules minimale au cours du temps 
    pour les modèles discret et continu, et la compare à la borne inférieure théorique l/M.
    """
    print("\n--- Lancement du comparateur de BORNES (Distances) ---")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # --- 1. Récupération des données du modèle DISCRET ---
    time_actualisation = 5/3600
    
    # On génère la simulation (sans l'afficher immédiatement pour ne garder que la comparaison)
    dummy_fig, dummy_ax1 = plt.subplots()
    t_tab_disc, x_tab_disc = discrete_model(
        N=N, time_actualisation=time_actualisation, l=l, 
        speed_func=speed_greenshields, init_pos_func=init_pos_bottleneck, ax=dummy_ax1
    )
    plt.close(dummy_fig) 
    
    # Calcul des distances inter-véhicules (x_{i+1} - x_i) sur tout l'historique
    distances_disc = np.diff(x_tab_disc, axis=0)
    
    # On prend la distance minimale à chaque instant t
    min_distances_disc = np.min(distances_disc, axis=0)
    
    # Calcul de la borne inf théorique : M = sup(l / (x_{i+1}(0) - x_i(0)))
    M = np.max(l / distances_disc[:, 0])
    lower_bound = l / M

    # Calcul de la borne sup théorique : 
    m = np.min(l/distances_disc[:, 0])
    upper_bound = l/m
    
    # --- 2. Récupération des données du modèle CONTINU ---
    L_total = (N * l_init) + (v_max * T)
    
    dummy_fig, dummy_ax2 = plt.subplots()
    t_tab_cont, rho_tab_cont = continuous_model(
        L=L_total, nx=200, flux_func=flux_godunov, 
        init_cond_func=init_rho_bottleneck, ax=dummy_ax2
    )
    plt.close(dummy_fig)
    
    # Dans le modèle continu, la distance inter-véhicules équivalente est l / rho
    # On évite la division par zéro en masquant les rho très petits (ex: route vide en aval)
    rho_tab_safe = np.where(rho_tab_cont > 1e-5, rho_tab_cont, 1e-5)
    distances_cont = 1 / rho_tab_safe
    
    # On cherche la distance minimale globale sur la route à chaque pas de temps
    min_distances_cont = np.min(distances_cont, axis=1)
    
    # --- 3. Tracé des courbes de comparaison ---
    ax.plot(t_tab_disc, min_distances_disc, label="Distance min. (Discret)", color="blue", linewidth=2)
    ax.plot(t_tab_cont, min_distances_cont, label="Distance min. équivalente (Continu)", color="green", linestyle="--", linewidth=2)
    
    # Ligne horizontale pour la borne inférieure théorique l/M
    ax.axhline(y=lower_bound, color="red", linestyle=":", linewidth=2.5, 
               label=f"Borne inf. théorique (l/M) = {lower_bound:.5f} km")
    ax.axhline(y=upper_bound, color="orange", linestyle="-.", linewidth=2, 
               label=f"Borne sup. théorique fermée (l/m) = {upper_bound:.5f} km")
    
    ax.set_title("Évolution de la distance inter-véhicules minimale vs Borne théorique", fontsize=14, fontweight='bold')
    ax.set_xlabel("Temps (h)", fontsize=12)
    ax.set_ylabel("Distance inter-véhicules minimale (km)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.6)
    
    ax.set_ylim(0.02, upper_bound * 1.5)
    plt.tight_layout()
    plt.show()


def run_continuous_comparison():
    """ Lance une grille de comparaison pour le modèle continu """
    print("\n--- Lancement du comparateur CONTINU ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Comparaison des Modèles Continus", fontsize=16, fontweight='bold')

    continuous_model(speed_func=speed_greenshields,flux_func=flux_lax_friedrichs, init_cond_func=init_rho_two_jams, ax=axes[0, 0])
    continuous_model(speed_func=speed_greenshields,flux_func=flux_godunov, init_cond_func=init_rho_two_jams, ax=axes[0, 1])
    plt.tight_layout(pad=3.0)
    plt.show()

def run_discrete_comparison():
    """ Lance une grille de comparaison pour le modèle discret """
    print("\n--- Lancement du comparateur DISCRET ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 6))
    fig.suptitle("Comparaison des Modèles Discrets", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    discrete_model(N=N, time_actualisation=time_actualisation, l=l, speed_func=speed_greenshields, init_pos_func=init_pos_uniform, ax=axes[0])
    discrete_model(N=N, time_actualisation=time_actualisation, l=l, speed_func=speed_greenshields, init_pos_func=init_pos_bottleneck, ax=axes[1])
    discrete_model(N=N, time_actualisation=time_actualisation, l=l, speed_func=speed_greenshields, init_pos_func=init_pos_dense, ax=axes[2])
    discrete_model(N=N, time_actualisation=time_actualisation, l=l, speed_func=speed_greenshields, init_pos_func=init_pos_wave, ax=axes[3])

    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == '__main__':
    # Comparaison des modèles
    # run_discrete_comparison()
    # run_continuous_comparison()
    # run_rho_comparison()
    run_bound_comparison()

    # Voir un seul modèle à la fois
    # continuous_model(speed_func=speed_greenshields, init_cond_func=init_rho_single_jam)
import matplotlib.pyplot as plt
import numpy as np
import random

#######################################################################
#                       Discrete Modelisation               
#######################################################################

'''
- T is a time in hour 
- v_max is a velocity in km/h 
- l is a length in km
- rho_max is a density in vehicle/km
'''
T = 0.1 # 6 minutes 
v_max = 50 
l = 0.005 # 5 meters
rho_max = 1 / l

def compute_speed(v_max, rho_max, rho, debug=False):
    '''
    Compute vehicle speed
    '''
    speed = v_max * (1 - rho / rho_max)
    clamped_speed = max(0, min(speed, v_max))
    
    if debug:
        print(f"  [DEBUG] Formule de Greenshields : vitesse = {v_max} * (1 - {rho:.2f} / {rho_max:.2f})")
        print(f"  [DEBUG] Résultat brut : {speed:.2f} km/h | Résultat borné : {clamped_speed:.2f} km/h")
        
    return clamped_speed

def discrete_model(N, time_actualisation):
    print(f"\n[DEBUG] === DÉMARRAGE DU MODÈLE DISCRET ===")
    time_steps = int(T / time_actualisation)
    print(f"[DEBUG] Temps total T={T}h. Pas de temps temporel (dt)={time_actualisation:.4f}h. Nombre d'itérations={time_steps}")
    
    t_tab = np.linspace(0, T, time_steps)
    x_tab = np.zeros((N, time_steps))
    v_tab = np.zeros((N, time_steps))

    # Initialisation des positions
    for i in range(N):
        if i == 0:
            x_tab[i][0] = 0
            print(f"[DEBUG] Position initiale Voiture {i} (Leader) : {x_tab[i][0]} km")
        else:
            espacement = random.uniform(50 * l, 100 * l)
            x_tab[i][0] = x_tab[i - 1][0] - espacement
            if i < 3: # On affiche juste les 3 premières pour comprendre
                print(f"[DEBUG] Position initiale Voiture {i} : placée à {espacement:.4f} km derrière la Voiture {i-1} -> {x_tab[i][0]:.4f} km")
    
    # Initialisation des vitesses
    for i in range(N):
        if i == N - 1:
            v_tab[i][0] = v_max
        else:
            distance = x_tab[i + 1][0] - x_tab[i][0]
            density = 1 / distance
            if i == 0:
                print(f"\n[DEBUG] Calcul de la vitesse initiale pour la Voiture 0 :")
                print(f"  [DEBUG] Distance avec la voiture devant : {distance:.4f} km -> Densité locale : {density:.2f} veh/km")
                v_tab[i][0] = compute_speed(v_max, rho_max, density, debug=True)
            else:
                v_tab[i][0] = compute_speed(v_max, rho_max, density)

    # Boucle d'évolution temporelle
    t = 1
    print(f"\n[DEBUG] --- Début de la simulation temporelle (boucle while) ---")
    while t < time_steps :
        for i in range(N):
            # Formule d'Euler explicite : x(t) = x(t-1) + v(t-1) * dt
            x_tab[i][t] = x_tab[i][t-1] + v_tab[i][t-1] * time_actualisation
            
            if t == 1 and i == 0: # Debug uniquement pour t=1 et voiture 0
                print(f"[DEBUG] t={t} | Voiture 0 avance : Nouvelle position = {x_tab[i][t-1]:.4f} + ({v_tab[i][t-1]:.2f} * {time_actualisation:.4f}) = {x_tab[i][t]:.4f} km")
        
        for i in range(N):
            if i == N - 1:  
                v_tab[i][t] = v_max
            else:
                distance = x_tab[i + 1][t] - x_tab[i][t]
                v_tab[i][t] = compute_speed(v_max, rho_max, 1 / distance)
        t += 1
    print(f"[DEBUG] Fin du calcul discret. Génération du graphique...\n")

    plt.figure(figsize=(10,6))
    for i in range(N):
        plt.plot(x_tab[i], t_tab, label = f'Position of vehicle {i+1}')
    plt.title('Evolution of vehicle positions over time')
    plt.xlabel('Position (in km)')
    plt.ylabel('Time (in hours)')
    plt.grid(True)
    plt.legend()
    plt.show()


#######################################################################
#                       Continuous Modelisation               
#######################################################################

L = 5
nx = 200
dx = L / nx
dt = 0.9 * dx / v_max
nt = int(T / dt)

def compute_rho_Greenshields(rho):
    return rho * compute_speed(v_max, rho_max, rho)

x_tab = np.linspace(0, L, nx)
rho = np.ones(nx) * (0.2 * rho_max)
rho[(x_tab >= 1) & (x_tab <= 2)] = 0.8 * rho_max
rho[(x_tab >= 3) & (x_tab <= 4)] = 0.8 * rho_max

def continuous_model(rho = rho):
    print(f"\n[DEBUG] === DÉMARRAGE DU MODÈLE CONTINU ===")
    print(f"[DEBUG] Longueur (L)={L}km, Nombres de cellules spatiales (nx)={nx} -> Pas spatial (dx)={dx:.4f} km")
    
    # Vérification de la condition CFL
    cfl_limit = dx / v_max
    print(f"[DEBUG] Vérification CFL : dt ({dt:.6f}h) doit être <= dx/v_max ({cfl_limit:.6f}h). Condition respectée : {dt <= cfl_limit}")
    
    rho_tab = np.zeros((nt, nx))
    rho_tab[0, :] = rho

    for ts in range(1, nt):
        rho_steps = np.zeros(nx)

        # Conditions aux limites
        rho_steps[0] = rho[0]
        rho_steps[-1] = rho[-1]

        for xs in range(1, nx - 1):
            # Schéma de discrétisation de type Lax-Friedrichs
            flux_avant = compute_rho_Greenshields(rho[xs - 1])
            flux_apres = compute_rho_Greenshields(rho[xs + 1])
            
            rho_steps[xs] = 0.5 * (rho[xs + 1] + rho[xs - 1]) - dt / (2 * dx) * (flux_apres - flux_avant)
            
            # Debug au milieu de la route lors de la première itération temporelle
            if ts == 1 and xs == int(nx/2):
                print(f"\n[DEBUG] --- Itération ts=1 | Cellule spatiale xs={xs} (milieu de la route) ---")
                print(f"  [DEBUG] Densité(xs-1)={rho[xs-1]:.2f} | Densité(xs+1)={rho[xs+1]:.2f}")
                print(f"  [DEBUG] Flux entrant(xs-1)={flux_avant:.2f} | Flux sortant(xs+1)={flux_apres:.2f}")
                print(f"  [DEBUG] Calcul de la nouvelle densité locale :")
                print(f"  [DEBUG] 0.5 * ({rho[xs+1]:.2f} + {rho[xs-1]:.2f}) - {dt:.6f}/(2*{dx:.4f}) * ({flux_apres:.2f} - {flux_avant:.2f}) = {rho_steps[xs]:.2f}")

        rho = rho_steps.copy()
        rho_tab[ts, :] = rho
        
    print(f"\n[DEBUG] Fin du calcul continu. Génération de la Heatmap...\n")
    plt.figure(figsize=(10, 6))
    plt.imshow(rho_tab, aspect='auto', origin='lower', extent=[0, L, 0, T], cmap='jet', vmin=0, vmax=rho_max)
    plt.colorbar(label='Density (in vehicles/km)')
    plt.title('Evolution of density over time')
    plt.xlabel('Position on the road (in km)')
    plt.ylabel('Time (in hours)')
    plt.show()

if __name__ == '__main__':
    discrete_model(N = 10, time_actualisation = 5 / 3600)
    continuous_model()
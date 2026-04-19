import matplotlib.pyplot as plt
import numpy as np
import random

#######################################################################
#                       Constants for modelisation
#######################################################################
'''
- T is a time in hour 
- v_max is a velocity in km/h 
- l is a length in km
- rho_max is a density in vehicle/km
- N is the number of vehicle
'''
T = 0.1 # 6 minutes
# @n: l'idéal est le plus juste est de définir t_span=[0.0, 0.1] et ensuite T = t_span[1]
v_max = 120 # @n: on fait un choix réaliste de vitesse maximale sur autoroute
l = 0.005 # 5 meters
rho_max = 1 / l #@n: correct, attention aux dimensions, on fait le choix km et h (1km / 0.005km)
N = 500
L = N * l # @n: on définit la norme L1 de la densité c'est-à dire la longuers des N véhicules bumper-to-bumper

'''
-rho_o is the initial density !!! initialisé avec N, on ne connait pas la position initiale des véhicules !!!
@n: il faut justement initialiser la position des véhicles à partir de la densité "ground-truth" 
'''
rho_0 = np.random.normal()

# @n: ici il ne faut pas tirer un scalaire aléatoire.
# On veut définir un vrai profil rho_0(x), puis reconstruire les positions
# initiales des véhicules à partir de cette densité.
# Pour rester simple : d'abord prendre une densité constante 
# ensuite varier e.g. densité feu rouge/vert ou inversement, ou demander LLM de vous en fournir

'''
#Définition des positions à partir de la répartition
# @n: il faut le faire, je rapelle que 
# la position des N véhicules se définit à partir de
#  x_{i+1} - x_i = (rho_max / rho_i) * (L / N)   (espacement inter-véhicule)
# ou précisément en Latex $\bar{x}^N_{i+1}\coloneqq \sup\left\{x\in\mathbb{R}:\int_{\bar{x}^N_i}^{x}\bar{\rho}(y)dy=\frac{L}{N}\right\},\quad i=0,\cdots, N-1$
# et du x_0 (que l'on choisira nul par exemple x_0=0 wlog), attention aux dimensions (vérifier pertinence de la présence de rho_max)
'''

'''
- L is the length of the road in km # @n: c'est justement là qu'il y mécompréhension,
# nous ne fixons pas une longeur de route initiale, celle-ci sera dictée par le données du problème, 
# notamment la vitesse maximale, la durée de simulation et la position initiale du véhicule leader
@n: j'insiste sur cette remarque importante:
La route n'est pas fixée à l'avance. Le domaine spatial sera choisi après avoir construit les positions.

# @n: Vous aurez quelque chose come:
x_min = float(X_0[0])  # Dernier follower àt=0 (a priori x_0=0)
x_max = float(X_0[-1]) + float(v_max) * float(t_span[1] - t_span[0])  # leader à t=T

- nx is the number of spatial discretization points
- dx is the spatial step in km
- dt is the time step in hours, to satisfy the CFL condition we need dt <= dx / v_max 
- nt is the number of time steps
'''
# L = x_tab[-1][0]+ v_max * T - x_tab[0][0] @n: on supprime donc cette ligne
nx = 1000
dx = L / nx

'''
@n: CFL :
dt <= dx / v_max
'''

dt = 0.9 * dx / v_max
nt = int(T / dt) + 1

x_init = x_tab[:, 0].copy() #@n; corriger en fonction des positions initiales
x_min = x_init[0] #@n: il s'agira bien de x_init[0] (à savoir x_0)
x_cell_edges = np.linspace(x_min, x_min + L, nx + 1)
#######################################################################
#                       Discrete Modelisation               
#######################################################################



def compute_speed(v_max, rho_max, rho):
    speed = v_max * (1 - rho / rho_max)
    return max(0, min(speed, v_max))

def discrete_model(N, time_actualisation, x_tab, time_steps):
    # @n: schéma Euler explicite correct dans l'esprit à vérifier et adapter aux modifications
    t_tab = np.linspace(0, T, time_steps)
    v_tab = np.zeros((N, time_steps))
    for i in range(N):
        if i == N - 1:
            v_tab[i][0] = v_max
        else:
            distance = x_tab[i + 1][0] - x_tab[i][0]
            v_tab[i][0] = compute_speed(v_max, rho_max, 1 / distance)
    t = 1
    while t < time_steps :
        for i in range(N):
            x_tab[i][t] = x_tab[i][t-1] + v_tab[i][t-1] * time_actualisation
        
        for i in range(N):
            if i == N - 1:  
                v_tab[i][t] = v_max
            else:
                distance = x_tab[i + 1][t] - x_tab[i][t]
                v_tab[i][t] = compute_speed(v_max, rho_max, 1 / distance)
        t += 1
    '''
    plt.figure(figsize=(10,6))
    for i in range(N):
        plt.plot(x_tab[i], t_tab, label = f'Position of vehicle {i+1}')
    plt.title('Evolution of vehicle positions over time')
    plt.xlabel('Position (in km)')
    plt.ylabel('Time (in hours)')
    plt.grid(True)
    plt.legend()
    plt.show()
    '''
    return x_tab


#######################################################################
#                       Continuous Modelisation               
#######################################################################

def compute_mesh(x, nx, dx, x_min):
    '''
    Compute rho from a set of position x using geometric projection
    '''

# @n: la projection géométrique cellule par cellule est possible,
# mais ici elle complique inutilement le problème.

# L'idée naturelle du modèle FtL est :
# sur chaque intervalle [x_i(t), x_{i+1}(t)),
# on pose rho = 1 / (x_{i+1}(t)-x_i(t))

# On obtient ainsi une densité en escalier,
# beaucoup plus simple à coder et directement liée au modèle micro.

# C'est cette reconstruction qu'on compare ensuite au modèle LWR.
# voir fonction suivante: compute_mesh_simple

    rho = np.zeros(nx)
    for i in range(N - 1):
        interval_left = x[i]
        interval_right = x[i + 1]
        rho_loc = 1.0 / (interval_right - interval_left)

        start_cell = max(0, min(nx - 1, int((interval_left - x_min) / dx)))
        end_cell = max(0, min(nx - 1, int((interval_right - x_min - 1e-12) / dx)))

        for j in range(start_cell, end_cell + 1):
            cell_left = x_cell_edges[j]
            cell_right = x_cell_edges[j + 1]
            overlap = max(0.0, min(cell_right, interval_right) - max(cell_left, interval_left))
            if overlap > 0:
                rho[j] += rho_loc * overlap / dx
    return rho

def compute_mesh_simple(x, x_grid):
    rho = np.zeros(len(x_grid))

    for i in range(len(x)-1):
        rho_i = 1 / (x[i+1] - x[i])

        mask = (x_grid >= x[i]) & (x_grid < x[i+1])
        rho[mask] = rho_i

    return rho

rho = compute_mesh(x_init, nx, dx, x_min) # @nb: attention il va falloir changer appel


def compute_rho_Greenshields(rho):
    return rho * compute_speed(v_max, rho_max, rho)

'''
- rho_c is the critical density on the road
- q_max is the maximum flux at rho_c
'''
rho_c = rho_max / 2
q_max = compute_rho_Greenshields(rho_c)


def compute_godunov_flux(rho_l, rho_r, rho_c, q_max):
    d_rho = compute_rho_Greenshields(rho_l) if rho_l <= rho_c else q_max
    s_rho = q_max if rho_r <= rho_c else compute_rho_Greenshields(rho_r)
    return min(d_rho, s_rho)


def continuous_model(rho = rho):
    rho_tab = np.zeros((nt, nx))
    rho_tab[0, :] = rho

    for ts in range(1, nt):
        rho_steps = np.zeros(nx)
        flux = np.zeros(nx + 1)

        flux[0] = compute_godunov_flux(rho[0], rho[0], rho_c, q_max)
        flux[-1] = compute_godunov_flux(rho[-1], rho[-1], rho_c, q_max)
        # @nb: il faudra probablement clarifier les conditions de bord, le reste me semble bon

        for xs in range(1, nx):
            flux[xs] = compute_godunov_flux(rho[xs - 1], rho[xs], rho_c, q_max)

        for xs in range(nx):
            rho_steps[xs] = rho[xs] - dt / dx * (flux[xs + 1] - flux[xs])

        rho_steps = np.clip(rho_steps, 0.0, rho_max)

        rho = rho_steps.copy()
        rho_tab[ts, :] = rho
    plt.figure(figsize=(10, 6))
    plt.imshow(rho_tab, aspect='auto', origin='lower', extent=[0, L, 0, T], cmap='jet', vmin=0, vmax=rho_max)
    plt.colorbar(label='Density (in vehicles/km)')
    plt.title('Evolution of density over time')
    plt.xlabel('Position on the road (in km)')
    plt.ylabel('Time (in hours)')
    plt.show()



#######################################################################
#                              Computation               
#######################################################################
if __name__ == '__main__':

    x_tab = discrete_model(N, time_actualisation, x_tab, time_steps)
    #@ n: il va falloir définir les objets ci-dessus, 
    # gross modo quelque chose comme ce qui suit à placer au bon endroit:
    # time_steps = nt
    # time_actualisation = dt
    # x_tab = np.zeros((N, nt))
    # x_tab[:, 0] = x_init

    rho_discrete = np.zeros((nt, nx))
    rho_discrete[0, :] = rho
    for ts in range(1, nt):
        rho_discrete[ts, :] = compute_mesh(x_tab[:, ts], nx, dx, x_min)

    plt.figure(figsize=(10, 6))
    plt.imshow(rho_discrete, aspect='auto', origin='lower', extent=[0, L, 0, T], cmap='jet', vmin=0, vmax=rho_max)
    plt.colorbar(label='Density (in vehicles/km)')
    plt.title('Evolution of density over time')
    plt.xlabel('Position on the road (in km)')
    plt.ylabel('Time (in hours)')
    plt.show()

    continuous_model()

#######################################################################
# @n: remarques pour la suite
#######################################################################
'''
1. Faire varier N = 100, 200, 500, 1000,2000, ...
   et comparer FTL reconstruit avec LWR. Bien sûr pas de temps et d'espace va jouer un rôle.

2. Calculer une erreur :
   ||rho_FTL - rho_LWR||_{L1}

3. Vérifier expérimentalement la convergence quand N augmente.

4. Rester simple :
   pas besoin d'outils compliqués, le but est de comprendre
   le lien micro (FTL) -> macro (LWR).
'''
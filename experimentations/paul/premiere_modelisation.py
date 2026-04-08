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
- N is the number of vehicle
'''
T = 0.1 # 6 minutes 
v_max = 50 
l = 0.005 # 5 meters
rho_max = 1 / l
N = 500
time_actualisation = 5 / 3600
time_steps = int(T / time_actualisation) + 1
x_tab = np.zeros((N, time_steps))
for i in range(N):
    if i == 0:
         x_tab[i][0] = 0
    else:
        x_tab[i][0] = x_tab[i - 1][0] + random.uniform(5 * l, 10 * l)

def compute_speed(v_max, rho_max, rho):
    speed = v_max * (1 - rho / rho_max)
    return max(0, min(speed, v_max))

def discrete_model(N, time_actualisation, x_tab, time_steps):
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

'''
- L is the length of the road in km
- nx is the number of spatial discretization points
- dx is the spatial step in km
- dt is the time step in hours, to satisfy the CFL condition we need dt <= dx / v_max 
- nt is the number of time steps
'''
L = x_tab[-1][0]+ v_max * T - x_tab[0][0]
nx = 200
dx = L / nx
dt = 0.9 * dx / v_max
nt = int(T / dt) + 1
x_init = x_tab[:, 0].copy()
x_min = x_init[0]
x_cell_edges = np.linspace(x_min, x_min + L, nx + 1)

def compute_mesh(x, nx, dx, x_min):
    '''
    Compute rho from a set of position x using geometric projection
    '''
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

rho = compute_mesh(x_init, nx, dx, x_min)

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


if __name__ == '__main__':

    x_tab = discrete_model(N, time_actualisation, x_tab, time_steps)
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
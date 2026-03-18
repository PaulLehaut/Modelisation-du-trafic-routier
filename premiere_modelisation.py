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

def compute_speed(v_max, rho_max, rho):
    '''
    Compute vehicle speed
    args:
    - v_max: int -> maximal velocity of our model
    - rho_max: int -> maximal density of vehicle on the road
    - rho: int -> actual density of vehicle on the road
    '''
    speed = v_max * (1 - rho / rho_max)
    return max(0, min(speed, v_max))

def discrete_model(N, time_actualisation):
    '''
    Print vehicle evolution over time
    args:
    - N: int -> number of vehicles
    - time_actualisation: int -> time between each actualisation
    '''
    time_steps = int(T / time_actualisation)
    t_tab = np.linspace(0, T, time_steps)
    x_tab = np.zeros((N, time_steps))
    v_tab = np.zeros((N, time_steps))

    for i in range(N):
        if i == 0:
            x_tab[i][0] = 0
        else:
            x_tab[i][0] = x_tab[i - 1][0] + random.uniform(5 * l, 10 * l)
    
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

'''
- L is the length of the road in km
- nx is the number of spatial discretization points
- dx is the spatial step in km
- dt is the time step in hours, to satisfy the CFL condition we need dt <= dx / v_max 
- nt is the number of time steps
'''
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
    rho_tab = np.zeros((nt, nx))
    rho_tab[0, :] = rho

    for ts in range(1, nt):
        rho_steps = np.zeros(nx)

        # We assume that the density is constant on the edges of the road
        rho_steps[0] = rho[0]
        rho_steps[-1] = rho[-1]

        for xs in range(1, nx - 1):
            rho_steps[xs] = 0.5 * (rho[xs + 1] + rho[xs - 1]) - dt / (2 * dx) * (compute_rho_Greenshields(rho[xs + 1]) - compute_rho_Greenshields(rho[xs - 1]))

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
    discrete_model(N = 500, time_actualisation = 5 / 3600)
    continuous_model()

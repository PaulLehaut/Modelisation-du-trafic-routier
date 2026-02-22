import matplotlib.pyplot as plt
import numpy as np
import random

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

def model(N, time_actualisation):
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
    plt.xlabel('Position')
    plt.ylabel('Time')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model(N = 5, time_actualisation = 5 / 3600)

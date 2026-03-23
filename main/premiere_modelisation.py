import matplotlib.pyplot as plt
import numpy as np
import random

def compute_speed(v_max, d, d_min):
    '''
    Compute vehicle speed
    args:
    - v_max: int -> maximal velocity of our model
    - d: int -> distance between the vehicle and the vehicle in front
    - d_min: int -> minimal distance between two vehicle
    '''
    speed = v_max * (d / d_min - 1)
    return max(0, min(speed, v_max))

def model(N, l, v_max, T, first_vehicle_speed):
    '''
    Print vehicle evolution over time
    args:
    - N: int -> number of vehicles
    - l: int -> length of the vehicles
    - v_max: int -> maximal velocity of our model
    - T: int -> time of study
    - first_vehicle_speed: int -> speed of the first vehicle 
    '''
    if T <= 1:
        raise ValueError('Period must be > 1')

    if first_vehicle_speed < 0 or first_vehicle_speed > v_max:
        raise ValueError('First vehicle speed must be between 0 and v_max.')
    
    x_tab = np.array([[0 for i in range(T)] for i in range(N)])
    v_tab = np.array([[0 for i in range(T)] for i in range(N)])
    for i in range(N):
        x_tab[i][0] = random.randint(0 if i == 0 else x_tab[i - 1][0] + l, (i+1) * l)
    v_tab[N-1][0] = v_max
    t_tab = np.linspace(0, T, T)
    v_tab[N-1] = first_vehicle_speed
    t = 1
    while t < T :
        for i in range(N):
            x_tab[N - 1 - i][t] = x_tab[N - 1 - i][t-1] + v_tab[N - 1 - i][t-1]
            if i == 0:
                v_tab[N - 1 - i][t] = first_vehicle_speed
            else:
                v_tab[N - 1 - i][t] = compute_speed(v_max, x_tab[N - i][t] - x_tab[N - 1 - i][t], l)
        t += 1
    plt.figure(figsize=(10,6))
    for i in range(N):
        plt.plot(x_tab[i], t_tab, label = f'Position of vehicle {i+1}')
    plt.xlabel('Position')
    plt.ylabel('Time')
    plt.grid(True)
    plt.legend()
    plt.show()


# Test
if __name__ == "__main__":
    model(N=5, l=5, v_max=50, T=50, first_vehicle_speed=42)


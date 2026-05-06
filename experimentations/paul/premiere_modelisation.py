import matplotlib.pyplot as plt
import numpy as np
import random

#######################################################################
#                       Constants for modelisation
#######################################################################
'''
- T is a time in hour 
- V_MAX is a velocity in km/h 
- L_VEHICLE is a vehicle length in km
- RHO_MAX is a density in vehicle/km
'''
T = 0.1 # 6 minutes 
V_MAX = 50
L_VEHICLE = 0.005 # 5 meters
RHO_MAX = 1 / L_VEHICLE

'''
- RHO_C is the critical density on the road (for continuous modelisation)
- Q_MAX is the maximum flux at rho_c (computed later)
'''
RHO_C = RHO_MAX / 2
# Q_MAX = compute_rho_Greenshields(RHO_C)


#######################################################################
#                   Initial density and position         
#######################################################################
DENSITY_PROFILES = {
    'shock_wave': lambda x : 
        }

#def rho_initial(x, profile):


#######################################################################
#                       Discrete Modelisation               
#######################################################################
def compute_speed(rho):
    speed = V_MAX * (1 - rho / RHO_MAX)
    return max(0, min(speed, V_MAX))

def discrete_model(N, time_actualisation, x_tab, time_steps):
    v_tab = np.zeros((N, time_steps))
    for i in range(N):
        if i == N - 1:
            v_tab[i][0] = V_MAX
        else:
            distance = max(x_tab[i + 1][0] - x_tab[i][0], L_VEHICLE)
            v_tab[i][0] = compute_speed(1 / distance)
    t = 1
    while t < time_steps :
        for i in range(N):
            x_tab[i][t] = x_tab[i][t-1] + v_tab[i][t-1] * time_actualisation
        
        for i in range(N):
            if i == N - 1:  
                v_tab[i][t] = V_MAX
            else:
                distance = x_tab[i + 1][t] - x_tab[i][t]
                v_tab[i][t] = compute_speed(1 / distance)
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
    rho = np.zeros(nx)
    for i in range(n - 1):
        interval_left = x[i]
        interval_right = x[i + 1]
        rho_loc = 1.0 / (interval_right - interval_left + 1e-12)

        start_cell = max(0, min(nx - 1, int((interval_left - x_min) / dx)))
        end_cell = max(0, min(nx - 1, int((interval_right - x_min - 1e-12) / dx)))

        for j in range(start_cell, end_cell + 1):
            cell_left = x_cell_edges[j]
            cell_right = x_cell_edges[j + 1]
            overlap = max(0.0, min(cell_right, interval_right) - max(cell_left, interval_left))
            if overlap > 0:
                rho[j] += rho_loc * overlap / dx
    return rho

def compute_rho_Greenshields(rho):
    return rho * compute_speed(rho)

Q_MAX = compute_rho_Greenshields(RHO_C)


def compute_godunov_flux(rho_l, rho_r):
    d_rho = compute_rho_Greenshields(rho_l) if rho_l <= RHO_C else Q_MAX
    s_rho = Q_MAX if rho_r <= RHO_C else compute_rho_Greenshields(rho_r)
    return min(d_rho, s_rho)


def continuous_model(rho, dt, nt):
    rho_tab = np.zeros((nt, nx))
    rho_tab[0, :] = rho

    for ts in range(1, nt):
        rho_steps = np.zeros(nx)
        flux = np.zeros(nx + 1)

        flux[0] = compute_godunov_flux(rho[0], rho[0])
        flux[-1] = compute_godunov_flux(rho[-1], rho[-1])

        for xs in range(1, nx):
            flux[xs] = compute_godunov_flux(rho[xs - 1], rho[xs])

        for xs in range(nx):
            rho_steps[xs] = rho[xs] - dt / dx * (flux[xs + 1] - flux[xs])

        rho_steps = np.clip(rho_steps, 0.0, RHO_MAX)

        rho = rho_steps.copy()
        rho_tab[ts, :] = rho
    return rho_tab



#######################################################################
#                              Computation               
#######################################################################
if __name__ == '__main__':
    '''
    - n_tab is an array containing the number of vehicles for a modelisation
    - err_tab compute the L1 distance between rho_discrete and rho_continuous
    '''
    n_tab = [10, 50, 100, 500, 1000, 2000]
    err_tab = []
    n_ref=20001
    for n in n_tab:
        l = n * L_VEHICLE # Length for N vehicles bumper-to-bumper
        x_domain = [0, l]
        options = ', '.join(DENSITY_PROFILES.keys())
        print(f"Chose an initial density profile: {options}.")
        choice = input()
        assert choice in DENSITY_PROFILES, 'Invalid choice of initial density.'

        ########################################################################
        #           Definition of the original position with rho_0             #
        ########################################################################
        x_ref = np.linspace(x_domain[0], x_domain[1], n_ref)
        rho_ref = DENSITY_PROFILES[choice](x_ref)
        x_0 = np.zeros(n)
        for i in range(1, n):
            x_0[i] = x_0[i-1] + 1 / rho_0[i-1]

        '''
        - l_road is the length of the road in km
        - nx is the number of spatial discretization points
        - dx is the spatial step in km
        - dt is the time step in hours, to satisfy the CFL condition we need dt <= dx / V_MAX 
        - nt is the number of time steps
        '''
        l_road = x_0[-1] + V_MAX * T - x_0[0]
        nx = 1000
        dx = l_road / nx
        dt_discrete = 0.5 * (L_VEHICLE / V_MAX)
        # CFM condition requires dt <= dx / V_MAX
        dt_continuous = 0.9 * dx / V_MAX

        nt_discrete = int(T / dt_discrete) + 1
        nt_continuous = int(T / dt_continuous) + 1

        plt.figure()
        plt.plot(rho_0 / RHO_MAX * 100)
        plt.xlabel('Position on the road')
        plt.ylabel('Density (percentage of RHO_MAX)')
        plt.title(f'Initial density {n} vehicles')
        plt.show()

        ########################################################################
        #                        Discrete modelisation                         #
        ########################################################################
        # x_tab for discrete modelisation
        x_tab = np.zeros((n, nt_discrete))
        x_tab[:,0] = x_0.copy()
        x_min = x_0[0]
        x_cell_edges = np.linspace(x_min, x_min + l_road, nx + 1)

        # rho for discrete modelisation
        rho_discrete_0 = compute_mesh(x_0, nx, dx, x_min)

        discrete_model(n, dt_discrete, x_tab, nt_discrete)
        rho_discrete = np.zeros((nt_continuous, nx))
        rho_discrete[0, :] = rho_discrete_0
        for ts in range(1, nt_continuous):
            t_continuous = ts * dt_continuous
            ts_discret = int(t_continuous / dt_discrete)
            ts_discret = min(ts_discret, nt_discrete - 1)
            rho_discrete[ts, :] = compute_mesh(x_tab[:, ts_discret], nx, dx, x_min)

        fig, ax = plt.subplots(2, 1, figsize = (15,15))
        im0 = ax[0].imshow(rho_discrete, aspect='auto', origin='lower', extent=[0, l_road, 0, T], cmap='jet', vmin=0, vmax=RHO_MAX)
        fig.colorbar(im0, ax=ax[0], label='Density (in vehicles/km)')
        ax[0].set_title(f'Evolution of density over time: discrete model for {n} vehicles')
        ax[0].set_xlabel('Position on the road (in km)')
        ax[0].set_ylabel('Time (in hours)')

        ########################################################################
        #                     Continuous modelisation                          #
        ########################################################################
        rho_continuous = continuous_model(rho_discrete_0, dt_continuous, nt_continuous)
        im1 = ax[1].imshow(rho_continuous, aspect='auto', origin='lower', extent=[0, l_road, 0, T], cmap='jet', vmin=0, vmax=RHO_MAX)
        fig.colorbar(im0, ax=ax[1], label='Density (in vehicles/km)')
        ax[1].set_title(f'Evolution of density over time: continuous model for {n} vehicles')
        ax[1].set_xlabel('Position on the road (in km)')
        ax[1].set_ylabel('Time (in hours)')
        plt.show()


        ########################################################################
        #                        Error computation                             #
        ########################################################################
        err = np.abs(rho_continuous - rho_discrete).sum()
        err_tab.append(err)
    
    plt.plot(n_tab, err_tab)
    plt.xlabel('Number of vehicles')
    plt.ylabel('L1 distance of discret and continuous model')
    plt.xscale('log')
    plt.title('Evolution of L1 distance with n')
    plt.show()
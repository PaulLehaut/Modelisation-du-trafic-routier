import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

#######################################################################
#                       Constants for modelisation                    #
#######################################################################
'''
- T is a time in hour 
- V_MAX is a velocity in km/h 
- L_VEHICLE is a vehicle length in km
- RHO_MAX is the maximal density (vehicle/km)
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
#                       Initial density                               #
#######################################################################
def shock_wave(r_pos):
    '''
    A light traffic at the start and heavy at the end
    '''
    return np.where(r_pos <= 0.5, 0.2, 0.9)
    
def rarefaction_wave(r_pos):
    '''
    An heavy traffic at the start which get smoother at the end
    '''
    return np.where(r_pos <= 0.75, 0.80, 0.20).astype(float)

def stop_and_go_wave(r_pos):
    '''
    Three zones of huge density split by smoother zones
    '''
    w     = 0.012
    rho   = 0.10 * np.ones_like(r_pos)
    delta = 0.80   
    for start, end in [(0.05, 0.22), (0.38, 0.50), (0.65, 0.82)]:
        rho += delta * (
            0.5 * (1 + np.tanh((r_pos - start) / w)) *
            0.5 * (1 - np.tanh((r_pos - end)   / w))
        )
    return np.clip(rho, 0.05, 0.95)

DENSITY_PROFILES = {
    'shock_wave': shock_wave ,
    'rarefaction_wave': rarefaction_wave,
    'stop_and_go_wave': stop_and_go_wave
    }


#######################################################################
#                       Discrete Modelisation                         #
#######################################################################
'''
- x is the absolute position of a vehicle on the road
- r_pos is the relative position of a vehcile given by r_pos[i] = i / number_of_vehicle
'''
def x_from_relative_position(choice, l_ref, n_ref = 10001):
    r_pos    = np.linspace(0.0, 1.0, n_ref)
    dr       = r_pos[1] - r_pos[0]
    inv_rho  = 1.0 / DENSITY_PROFILES[choice](r_pos)

    cum      = np.zeros(n_ref)
    cum[1:]  = np.cumsum(0.5 * (inv_rho[:-1] + inv_rho[1:]) * dr)
    total    = cum[-1]

    x_ref    = l_ref * cum / total  
    return r_pos, x_ref


def relative_positions_from_x(x, x_ref, r_pos):
    return np.interp(x, x_ref, r_pos)

def compute_initial_positions(N, r_pos, rho_0_norm):
    gaps = L_VEHICLE / rho_0_norm

    x_0 =  np.zeros(N + 1)
    x_0[1:] = np.cumsum(gaps)

    l_0 = x_0[N]
    return x_0, l_0

def compute_speed(rho):
    speed = V_MAX * (1 - rho / RHO_MAX)
    return np.clip(speed, 0, V_MAX)

def discrete_model(N, time_actualisation, x_tab, nb_time_steps):
    v_tab = np.zeros((N + 1, nb_time_steps))

    distance = x_tab[1:, 0] - x_tab[:N, 0]
    v_tab[:N, 0] = compute_speed(1 / distance)
    v_tab[N, 0] = V_MAX

    t = 1
    while t < nb_time_steps :
        x_tab[:, t] = x_tab[:, t-1] + v_tab[:, t-1] * time_actualisation
        
        distance = np.maximum(x_tab[1:, t] - x_tab[:N, t], L_VEHICLE)
        v_tab[:N, t] = compute_speed(1 / distance)
        v_tab[N, t] = V_MAX
        t += 1

def compute_rho_discrete_from_position(x_tab, x_cells, t_discrete, t_continuous):
    '''
    Interpolate vehicles position then compute density at times studied
    in the discrete model (both discrete and continuous temporal grids are not equals)
    
    - x_tab[N, T] is the array of position given by the discrete model
    - t_discrete is the temporal grid of discrete model
    - t_continuous is the temporal grid of continuous model
    '''
    n_seg = x_tab.shape[0] - 1 
    interpolation_x = interp1d(t_discrete, x_tab, axis=1, kind='linear', bounds_error=False, fill_value=(x_tab[:,0], x_tab[:,-1]))

    nt_cont = len(t_continuous)
    x_seg = np.zeros((n_seg + 1, nt_cont))
    rho_seg = np.zeros((n_seg, nt_cont))

    for k, t in enumerate(t_continuous):
        x_k = interpolation_x(t)
        gap = np.maximum(np.diff(x_k), L_VEHICLE)
        x_seg[:, k] = x_k
        rho_seg[:, k] = np.minimum(1 / gap, RHO_MAX)
    
    rho_discrete = np.zeros((nt_cont, len(x_cells)))
    for k in range(nt_cont):
        x_k = x_seg[:, k]
        idx = np.searchsorted(x_k, x_cells, side='right') - 1
        in_fleet = (idx >= 0) & (idx < n_seg)
        rho_discrete[k, in_fleet] = rho_seg[idx[in_fleet], k]
    
    return rho_discrete


#######################################################################
#                       Continuous Modelisation                       #     
#######################################################################
def compute_initial_density(x_cells, r_pos, x_ref, L_ref, profile):
    '''
    rho_0_continuous(x_i) = rho_0_norm(r_pos(x_i)) * RHO_MAX
                          = rho_0_norm(i / N) * RHO_MAX
                          = rho_0_discrete(x_i)
    when N -> infty
    '''
    rho_0 = np.zeros(len(x_cells))

    between_boundaries = (x_cells >= 0) & (x_cells <= L_ref)
    r_pos_vals = relative_positions_from_x(x_cells[between_boundaries], x_ref, r_pos)
    rho_0[between_boundaries] = DENSITY_PROFILES[profile](r_pos_vals) * RHO_MAX
    return rho_0

def compute_rho_Greenshields(rho):
    return rho * compute_speed(rho)

Q_MAX = compute_rho_Greenshields(RHO_C)

def compute_godunov_flux(rho_l, rho_r):
    rho_l = np.asarray(rho_l)
    rho_r = np.asarray(rho_r)
    
    d_rho = np.where(rho_l <= RHO_C, compute_rho_Greenshields(rho_l), Q_MAX)
    s_rho = np.where(rho_r <= RHO_C, Q_MAX, compute_rho_Greenshields(rho_r))
    return np.minimum(d_rho, s_rho)


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
#                         Erreurs L1 et L2                            #   
#######################################################################
def l1_error(rho_discrete, rho_continuous, dx, dt):
    T_tot = dt * (rho_discrete.shape[0] - 1)
    L_domain = dx * rho_discrete.shape[1]
    return np.sum(np.abs(rho_continuous - rho_discrete) * dx * dt / (T_tot * L_domain))

def l2_error(rho_discrete, rho_continuous, dx, dt):
    T_tot = dt * (rho_discrete.shape[0] - 1)
    L_domain = dx * rho_discrete.shape[1]
    return np.sqrt(np.sum(np.abs(rho_continuous - rho_discrete)**2)* dx * dt / (T_tot * L_domain))


#######################################################################
#                           Computation                               #   
#######################################################################
if __name__ == '__main__':
    '''
    - n_tab is an array containing the number of vehicles for a modelisation
    - err_tab compute the L1 distance between rho_discrete and rho_continuous
    '''
    n_tab = [20, 50, 100, 200, 500, 1000, 2000]
    err1 = []
    err2 = []
    options = ', '.join(DENSITY_PROFILES.keys())
    print(f"Chose an initial density profile: {options}.")
    choice = input()
    assert choice in DENSITY_PROFILES, 'Invalid choice of initial density.'

    for n in n_tab:
        ########################################################################
        #                        Discrete modelisation                         #
        ########################################################################
        r_pos = np.arange(n) / n
        rho_0_norm = DENSITY_PROFILES[choice](r_pos)
        x_0, l_0 = compute_initial_positions(n, r_pos, rho_0_norm)
        rho_0 = rho_0_norm * RHO_MAX

        '''
        - dt is the time step in hours, to satisfy the CFL condition we need dt <= L_VEHICLE / V_MAX 
        - nt is the number of time steps
        '''
        dt_discrete = 0.9 * (L_VEHICLE / V_MAX)
        nt_discrete = int(T / dt_discrete) + 1

        plt.figure()
        plt.plot(rho_0_norm * 100)
        plt.xlabel('Position on the road')
        plt.ylabel('Density (percentage of RHO_MAX) for discrete model')
        plt.title(f'Initial density {n} vehicles')
        plt.show()

        # x_tab for discrete modelisation
        x_tab = np.zeros((n + 1, nt_discrete))
        x_tab[:,0] = x_0.copy()
        x_max = x_0[n] + V_MAX * T

        discrete_model(n, dt_discrete, x_tab, nt_discrete)
        t_discrete = np.linspace(0, T, nt_discrete)


        ########################################################################
        #                     Continuous modelisation                          #
        ########################################################################
        nx = max(100, n)  
        dx = x_max / nx
        # CFL condition requires dt <= dx / V_MAX
        x_cells = np.linspace(dx / 2, x_max - dx / 2, nx)
        dt_continuous = 0.9 * dx / V_MAX
        nt_continuous = int(T / dt_continuous) + 1

        r_pos_continuous, x_ref_continuous = x_from_relative_position(choice, l_0)
        rho_0_continuous = compute_initial_density(x_cells, r_pos_continuous, x_ref_continuous, l_0, choice)

        rho_continuous = continuous_model(rho_0_continuous, dt_continuous, nt_continuous)
        t_continuous = np.linspace(0, T, nt_continuous)

        ########################################################################
        #                              Plots                                   #
        ########################################################################
        '''
        Computation of the density from discrete model
        '''
        rho_discrete = compute_rho_discrete_from_position(x_tab, x_cells, t_discrete, t_continuous)

        fig, ax = plt.subplots(2, 1, figsize = (15,15))
        im0 = ax[0].imshow(rho_discrete, aspect='auto', origin='lower', extent=[0, x_max, 0, T], cmap='jet', vmin=0, vmax=RHO_MAX)
        fig.colorbar(im0, ax=ax[0], label='Density (in vehicles/km)')
        ax[0].set_title(f'Evolution of density over time: discrete model for {n} vehicles')
        ax[0].set_xlabel('Position on the road (in km)')
        ax[0].set_ylabel('Time (in hours)')

        im1 = ax[1].imshow(rho_continuous, aspect='auto', origin='lower', extent=[0, x_max, 0, T], cmap='jet', vmin=0, vmax=RHO_MAX)
        fig.colorbar(im1, ax=ax[1], label='Density (in vehicles/km)')
        ax[1].set_title(f'Evolution of density over time: continuous model for {n} vehicles')
        ax[1].set_xlabel('Position on the road (in km)')
        ax[1].set_ylabel('Time (in hours)')
        plt.show()


        ########################################################################
        #                        Error computation                             #
        ########################################################################
        e1 = l1_error(rho_discrete, rho_continuous, dx, dt_continuous)
        e2 = l2_error(rho_discrete, rho_continuous, dx, dt_continuous)
        err1.append(e1)
        err2.append(e2)

    fig, ax = plt.subplots(1, 1, figsize = (10, 6))
    ax.loglog(n_tab, err1, 'o-', label='Erreur L1', linewidth=2, markersize=8)
    ax.loglog(n_tab, err2, 's-', label='Erreur L2', linewidth=2, markersize=8)
    ax.set_xlabel('Number of vehicles N', fontsize=12)
    ax.set_ylabel('Error (normalized) [veh/km]', fontsize=12)
    ax.set_title('Convergence Discrete → Continuous', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
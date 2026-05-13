import numpy as np
from scipy.interpolate import interp1d

import os
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
#  Paramètres
# ══════════════════════════════════════════════════════════════════════

V_MAX   = 50.0    # [km/h]  vitesse maximale
L_VEH   = 0.005   # [km]    longueur d'un véhicule (5 m)
RHO_MAX = 1.0 / L_VEH   # [veh/km]  densité maximale physique

PROFILE = 'rarefaction'   # choisir le profil de densité 'shock' | 'rarefaction' | 'stop_and_go'

N_LIST  = [20, 50, 100, 200, 500, 1000, 2000, 5000, 6000, 8000, 10000]
T_LIST  = [0.05, 0.05, 0.02, 0.04, 0.1, 0.2, 0.25, 0.5, 0.6, 0.8, 1.00]

# T_LIST[k] correspond à N_LIST[k].
# T s'adapte avec N car L_ref proportionnel à N — il faut que les ondes aient le temps de traverser.

NX_PAR_VEH = 1   # cellules Godunov par véhicule
                   # dx ≈ L_VEH/(rho_mean * NX_PAR_VEH) — constant avec N
                   # 1 suffit pour la convergence, augmenter pour plus de précision

# on peut également décorréler le choix de la résolution spatiale Nx (Godunov)
# en le préservant constant et indépendant de N 

CFL = 0.9    # facteur CFL ∈ (0,1]

# ══════════════════════════════════════════════════════════════════════
#  Modèle de vitesse, flux — Greenshields
#  Modifier uniquement ces fonctions pour changer de modèle.
# ══════════════════════════════════════════════════════════════════════

def velocity(rho: np.ndarray) -> np.ndarray:
    """V(rho) = V_MAX*(1-rho/RHO_MAX) [km/h]. V(0)=V_MAX, V(RHO_MAX)=0."""
    return np.clip(V_MAX * (1.0 - rho / RHO_MAX), 0.0, V_MAX)

def flux(rho: np.ndarray) -> np.ndarray:
    """f(rho) = rho*V(rho) [veh/h]. Parabolique concave, f(0)=f(RHO_MAX)=0."""
    return rho * velocity(rho)

def flux_derivative(rho: np.ndarray) -> np.ndarray:
    """f'(rho) = V_MAX*(1-2*rho/RHO_MAX) [km/h]. Vitesse des ondes LWR."""
    return V_MAX * (1.0 - 2.0 * rho / RHO_MAX)

def critical_density() -> float:
    """
    rho_c = argmax f, solution de f'(rho_c) = 0.
    Greenshields : rho_c = RHO_MAX/2.
    Q_MAX = f(rho_c) est la capacité de la route [veh/h].
    """
    rho_c = RHO_MAX / 2.0
    assert abs(flux_derivative(np.array([rho_c]))[0]) < 1e-10
    return rho_c

# ══════════════════════════════════════════════════════════════════════
#  Profil de densité initiale normalisé
# ══════════════════════════════════════════════════════════════════════

def rho_0_norm(s: np.ndarray, profile: str) -> np.ndarray:
    """
    Profil normalisé rho_0_norm(s) ∈ ]0,1] pour s ∈ [0,1].

    s = i/N est l'indice normalisé du véhicule — PAS sa position spatiale.
    La position physique x_i est obtenue par cumul des gaps.

    Entrée : s ∈ [0,1]  (0=arrière de la flotte, 1=avant proche du leader)
    Sortie : rho_0_norm ∈ ]0,1]  sans unité
    Densité physique : rho_0_phys = rho_0_norm * RHO_MAX [veh/km]

    shock :
      Fluide à gauche, dense à droite.
      rho_l=0.20, rho_r=0.90 → rho_l+rho_r ≠ 1 → choc mobile
      v_s = (f(rho_r)-f(rho_l))/(rho_r-rho_l) < 0 → remonte vers la gauche

    rarefaction :
      Dense à gauche, fluide à droite.
      rho_l=0.80, rho_r=0.20 → f'(rho_l) < f'(rho_r)
      → caractéristiques divergent → fan de raréfaction qui s'ouvre

    stop_and_go :
      3 plateaux denses (rho=0.90) séparés par zones fluides (rho=0.10).
      Fronts quasi-discontinus (tanh w=0.012).
      Front montant (fluide→dense) : choc qui remonte ←
      Front descendant (dense→fluide) : raréfaction qui avance →
      → ondes dans les deux directions simultanément.
    """
    s = np.asarray(s, dtype=float)

    if profile == 'shock':
        return np.where(s <= 0.5, 0.20, 0.90).astype(float)

    elif profile == 'rarefaction':
        return np.where(s <= 0.75, 0.80, 0.20).astype(float)

    elif profile == 'stop_and_go':
        w     = 0.012
        rho   = 0.10 * np.ones_like(s)
        delta = 0.80   # 0.10 → 0.90
        for debut, fin in [(0.05, 0.22), (0.38, 0.50), (0.65, 0.82)]:
            rho += delta * (
                0.5 * (1 + np.tanh((s - debut) / w)) *
                0.5 * (1 - np.tanh((s - fin)   / w))
            )
        return np.clip(rho, 0.05, 0.95)

    else:
        raise ValueError(f"Profil inconnu : {profile!r}. "
                         "Choisir 'shock', 'rarefaction', 'stop_and_go'.")

# ══════════════════════════════════════════════════════════════════════
#  Transformation s ↔ x
# ══════════════════════════════════════════════════════════════════════

def build_x_from_s(profile: str, L_ref: float,
                   n_ref: int = 10001) -> tuple:
    """
    Calcule la transformation x(s) sur une grille fine.

    x(s) = L_ref * ∫_0^s 1/rho_0_norm(u) du / ∫_0^1 1/rho_0_norm(u) du

    L'inverse s(x) est obtenu par interpolation.

    Retourne :
      s_ref : (n_ref,)   grille en s
      x_ref : (n_ref,)   grille en x correspondante [km]
    """
    s_ref    = np.linspace(0.0, 1.0, n_ref)
    ds       = s_ref[1] - s_ref[0]
    inv_rho  = 1.0 / rho_0_norm(s_ref, profile)

    # Intégrale cumulée de 1/rho_0_norm(s)
    cum      = np.zeros(n_ref)
    cum[1:]  = np.cumsum(0.5 * (inv_rho[:-1] + inv_rho[1:]) * ds)
    total    = cum[-1]

    x_ref    = L_ref * cum / total   # x(s) ∈ [0, L_ref]
    return s_ref, x_ref

def s_from_x(x: np.ndarray, s_ref: np.ndarray,
             x_ref: np.ndarray) -> np.ndarray:
    """
    Transformation inverse s(x) par interpolation.
    x doit être dans [0, L_ref].
    """
    return np.interp(x, x_ref, s_ref)

# ══════════════════════════════════════════════════════════════════════
#  Positions initiales FtL
# ══════════════════════════════════════════════════════════════════════

def build_initial_positions(N: int, profile: str) -> tuple:
    """
    Place N+1 particules selon la formule du cours :
      x_0 = 0
      x_{i+1} = x_i + gap_i   où   gap_i = L_VEH / rho_0_norm(s_i)

    avec s_i = i/N et L = N*L_VEH (longueur bumper-to-bumper).

    Dérivation : gap_i = L/(N*rho_0_phys) = N*L_VEH/(N*rho_0_norm*RHO_MAX)
                       = L_VEH / rho_0_norm

    Cohérence : rho_i = 1/gap_i = rho_0_norm*RHO_MAX = rho_0_phys [veh/km] ✓
    Contrainte : gap_i ≥ L_VEH ⟺ rho_0_norm ≤ 1 ✓

    Retourne :
      X_0   : (N+1,) positions [km], X_0[0]=0
      rho_0 : (N,)   densités initiales [veh/km] = rho_0_phys(s_i)
      L_ref : float  longueur totale de la flotte [km] = X_0[N]
                (converge vers L_ref_∞ = L_VEH * ∫_0^1 1/rho_0_norm ds quand N→∞)
    """
    s          = np.arange(N) / N
    norm       = rho_0_norm(s, profile)
    gaps       = L_VEH / norm
    X_0        = np.zeros(N + 1)
    X_0[1:]    = np.cumsum(gaps)
    return X_0, norm * RHO_MAX, float(X_0[N])

# ══════════════════════════════════════════════════════════════════════
#  Simulation FtL - Euler explicite
# ══════════════════════════════════════════════════════════════════════

def simulate_ftl(N: int, T: float, profile: str) -> tuple:
    """
    Simule N+1 (N followers + 1 leader) véhicules par Euler explicite.

    Densité FtL = constante par morceaux sur N segments :
      rho_i(t) = L_VEH/(x_{i+1}(t)-x_i(t))  [veh/km]

    Pour N petit : N segments grossiers → densité très différente du profil continu.
    Pour N grand : N segments fins → densité converge vers le profil continu.
    C'est cette convergence qu'on mesure.

    dt_ftl = CFL*L_VEH/V_MAX — grille temporelle propre à FtL.
    """
    X_0, rho_0, L_ref = build_initial_positions(N, profile)
    dt_ftl = CFL * L_VEH / V_MAX
    NT     = int(T / dt_ftl) + 1
    t_eval = np.linspace(0.0, T, NT)

    X        = np.zeros((NT, N + 1))
    X[0, :]  = X_0

    for k in range(NT - 1):
        dt    = t_eval[k + 1] - t_eval[k]
        x_cur = X[k, :]
        gap   = np.maximum(x_cur[1:] - x_cur[:-1], L_VEH)
        X[k + 1, :N] = x_cur[:N] + dt * velocity(1.0 / gap)
        X[k + 1,  N] = x_cur[N]  + dt * V_MAX

    return t_eval, X, X_0, rho_0, L_ref

# ══════════════════════════════════════════════════════════════════════
#  Densité eulerienne constante par morceaux
# ══════════════════════════════════════════════════════════════════════

def ftl_density(X: np.ndarray, t_ftl: np.ndarray,
                t_query: np.ndarray) -> tuple:
    """
    Densité FtL : constante par morceaux sur N segments.

    Pour chaque temps t_query[k] :
      x_seg[k]  = positions des N+1 véhicules (interpolées)
      rho_seg[k] = 1/gap_i = densité du segment i  [veh/km]

    Retourne :
      x_segments : (NT_query, N+1)  positions des véhicules
      rho_segs   : (NT_query, N)    densité de chaque segment
    """
    N        = X.shape[1] - 1
    interp_X = interp1d(t_ftl, X, axis=0, kind='linear',
                        bounds_error=False, fill_value=(X[0], X[-1]))

    NT_q       = len(t_query)
    x_segments = np.zeros((NT_q, N + 1))
    rho_segs   = np.zeros((NT_q, N))

    for k, tk in enumerate(t_query):
        x_k            = interp_X(tk)
        gap            = np.maximum(np.diff(x_k), L_VEH)
        x_segments[k]  = x_k
        rho_segs[k]    = np.minimum(1.0 / gap, RHO_MAX)

    return x_segments, rho_segs

def project_ftl_on_grid(x_segments: np.ndarray,
                         rho_segs: np.ndarray,
                         x_cells: np.ndarray) -> np.ndarray:
    """
    Projette la densité FtL (constante par morceaux) sur la grille Godunov.

    Pour chaque cellule x_cells[j] : cherche le segment i qui la contient
    et affecte rho_segs[k,i].

    Séparé de ftl_density pour clarifier que c'est une opération de
    post-traitement, pas une partie du modèle FtL.
    """
    NT_q, N = rho_segs.shape
    rho_ftl = np.zeros((NT_q, len(x_cells)))

    for k in range(NT_q):
        x_k      = x_segments[k]
        idx      = np.searchsorted(x_k, x_cells, side='right') - 1
        in_fleet = (idx >= 0) & (idx < N)
        rho_ftl[k, in_fleet] = rho_segs[k, idx[in_fleet]]

    return rho_ftl

# ══════════════════════════════════════════════════════════════════════
#  Condition initiale LWR — cohérente avec FtL en espace physique
# ══════════════════════════════════════════════════════════════════════

def initial_condition_lwr(x_cells: np.ndarray,
                           s_ref: np.ndarray,
                           x_ref: np.ndarray,
                           L_ref: float,
                           profile: str) -> np.ndarray:
    """
    rho_0_lwr(x) = rho_0_norm(s(x)) * RHO_MAX

    où s(x) est la transformation inverse x→s calculée via build_x_from_s.

    Ceci garantit que rho_0_lwr(x_i) = rho_0_norm(i/N) * RHO_MAX = rho_0_ftl_i
    quand N→∞, i.e. les deux conditions initiales coïncident en espace physique.

    Cellules hors [0, L_ref] : rho = 0.
    """
    rho      = np.zeros(len(x_cells))
    in_fleet = (x_cells >= 0.0) & (x_cells <= L_ref)
    s_vals   = s_from_x(x_cells[in_fleet], s_ref, x_ref)
    rho[in_fleet] = rho_0_norm(s_vals, profile) * RHO_MAX
    return rho

# ══════════════════════════════════════════════════════════════════════
#  Simulation LWR par schéma de Godunov
# ══════════════════════════════════════════════════════════════════════

def _godunov_flux_vec(rho_l: np.ndarray, rho_r: np.ndarray,
                      rho_c: float) -> np.ndarray:
    """
    Flux de Godunov vectorisé pour f concave (Greenshields).

    Raréfaction (rho_l ≤ rho_r) : F = min(f(rho_l), f(rho_r))
    Choc (rho_l > rho_r) :
      F = f(rho_l)  si rho_l ≤ rho_c
        = f(rho_r)  si rho_r ≥ rho_c
        = f(rho_c)  sinon
    """
    fl = flux(rho_l)
    fr = flux(rho_r)
    fc = flux(np.full_like(rho_l, rho_c))
    raref   = rho_l <= rho_r
    c_left  = (~raref) & (rho_l <= rho_c)
    c_right = (~raref) & (rho_r >= rho_c)
    c_mid   = (~raref) & (~c_left) & (~c_right)
    F = np.empty_like(rho_l)
    F[raref]   = np.minimum(fl, fr)[raref]
    F[c_left]  = fl[c_left]
    F[c_right] = fr[c_right]
    F[c_mid]   = fc[c_mid]
    return F

def simulate_lwr(T: float, x_cells: np.ndarray,
                 s_ref: np.ndarray, x_ref: np.ndarray,
                 L_ref: float, profile: str) -> tuple:
    """
    Résout LWR par Godunov sur le domaine x_cells pendant T heures.

    Condition initiale : initial_condition_lwr — profil continu, indépendant de N.

    LWR est la solution vers laquelle FtL converge.
    Sa condition initiale est le profil continu, pas les N segments FtL.

    Conditions aux bords :
      Gauche : Dirichlet rho[0] = rho_0_left (valeur initiale)
               → préserve la condition entrante, évite les réflexions artificielles
      Droite : rho[-1] = 0 (zone vide, leader sorti)

    dt_god = CFL*dx/V_MAX — grille temporelle propre à Godunov.
    """
    rho_c  = critical_density()
    dx     = x_cells[1] - x_cells[0]
    dt     = CFL * dx / V_MAX
    NT     = int(T / dt) + 1
    t_god  = np.linspace(0.0, T, NT)

    # Condition initiale continue — indépendante de N
    rho        = initial_condition_lwr(x_cells, s_ref, x_ref,
                                       L_ref, profile)
    rho_0_left = float(rho[0])   # Dirichlet gauche = valeur initiale au bord

    rho_lwr       = np.zeros((NT, len(x_cells)))
    rho_lwr[0, :] = rho.copy()

    for step in range(NT - 1):
        F = _godunov_flux_vec(rho[:-1], rho[1:], rho_c)
        rho_new        = rho.copy()
        rho_new[1:-1] -= (dt / dx) * (F[1:] - F[:-1])
        rho_new[0]     = rho_0_left   # Dirichlet gauche
        rho_new[-1]    = 0.0          # Dirichlet droite (zone vide)
        rho            = np.clip(rho_new, 0.0, RHO_MAX)
        rho_lwr[step + 1, :] = rho.copy()

    return t_god, rho_lwr

# ══════════════════════════════════════════════════════════════════════
#  ERREURS L1 ET L2 NORMALISÉES
# ══════════════════════════════════════════════════════════════════════

def l1_error(rho_ftl: np.ndarray, rho_lwr: np.ndarray,
             dx: float, dt: float) -> float:
    """
    err_L1 = (1/(T*L)) * ∫∫ |rho_FtL - rho_LWR| dx dt   [veh/km]
    """
    T_total  = dt * (rho_ftl.shape[0] - 1)
    L_domain = dx * rho_ftl.shape[1]
    return float(np.sum(np.abs(rho_ftl - rho_lwr)) * dx * dt
                 / (T_total * L_domain))

def l2_error(rho_ftl: np.ndarray, rho_lwr: np.ndarray,
             dx: float, dt: float) -> float:
    """
    err_L2 = sqrt( (1/(T*L)) * ∫∫ (rho_FtL-rho_LWR)² dx dt )   [veh/km]
    """
    T_total  = dt * (rho_ftl.shape[0] - 1)
    L_domain = dx * rho_ftl.shape[1]
    return float(np.sqrt(np.sum((rho_ftl - rho_lwr)**2) * dx * dt
                         / (T_total * L_domain)))

# ══════════════════════════════════════════════════════════════════════
#  PROGRAMME PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def generate_exact_csv_dataset(N=1000, duration_minutes=10, output_dir="experimentations/bastien/datasets"):
    """
    Simulates N vehicles for a specific duration, interpolates the positions
    to exact 1-second intervals, and saves them to a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    profiles = ['shock', 'rarefaction', 'stop_and_go']
    
    # Time parameters
    T_hours = duration_minutes / 60.0
    num_seconds = duration_minutes * 60
    # Create an exact 1-second grid (in hours for the simulation)
    t_query_hours = np.arange(0, num_seconds) / 3600.0 
    
    for profile in profiles:
        print(f"Generating simulation for {profile} profile (N={N})...")
        
        # 1. Run the simulation
        t_ftl, X_ftl, X_0_N, rho_0, L_ref_N = simulate_ftl(N, T_hours, profile)
        
        # 2. Interpolate to exact 1-second intervals
        interp_X = interp1d(t_ftl, X_ftl, axis=0, kind='linear', fill_value='extrapolate')
        X_1sec = interp_X(t_query_hours)
        
        # 3. Calculate instantaneous velocities at these 1-second marks
        gaps = np.maximum(np.diff(X_1sec, axis=1), L_VEH)
        V_followers = velocity(1.0 / gaps)
        V_leader = np.full((len(t_query_hours), 1), V_MAX)
        V_1sec = np.hstack((V_followers, V_leader))
        
        # 4. Flatten arrays to create the tabular dataset
        NT_q = len(t_query_hours)
        num_vehicles = N + 1 # N followers + 1 leader
        
        times_flat = np.repeat(np.arange(0, num_seconds), num_vehicles) # Time in seconds
        veh_ids_flat = np.tile(np.arange(num_vehicles), NT_q)
        x_flat = X_1sec.flatten()
        v_flat = V_1sec.flatten()
        
        df = pd.DataFrame({
            'Time_s': times_flat.astype(int),
            'Vehicle_ID': veh_ids_flat,
            'Position_km': x_flat,
            'Velocity_kmh': v_flat
        })
        
        # 5. Save to CSV
        filename = os.path.join(output_dir, f"traffic_N{N}_{profile}.csv")
        df.to_csv(filename, index=False)
        print(f"  -> Saved {filename} ({len(df):,} rows)")

if __name__ == '__main__':
    generate_exact_csv_dataset(N=1000, duration_minutes=10)
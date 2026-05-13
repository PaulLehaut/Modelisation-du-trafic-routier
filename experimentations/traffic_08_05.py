import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
    fl = flux(rho_l); fr = flux(rho_r)
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

if __name__ == '__main__':

    rho_c = critical_density()
    Q_MAX = float(flux(np.array([rho_c]))[0])

    # ── Visualisation des profils initiaux ────────────────────────────
    s_plot = np.linspace(0, 1, 1000)
    fig0, axes0 = plt.subplots(1, 3, figsize=(13, 4))
    fig0.suptitle("Profils de densité initiale normalisée rho_0_norm(s)", fontsize=13)
    for ax_, prof in zip(axes0, ['shock', 'rarefaction', 'stop_and_go']):
        rho_p = rho_0_norm(s_plot, prof)
        ax_.plot(s_plot, rho_p, 'b-', lw=2)
        ax_.axhline(0.5, color='red', ls='--', lw=1, label='rho_c/RHO_MAX')
        ax_.set_title(prof, fontsize=11)
        ax_.set_xlabel('s (position normalisée)')
        ax_.set_ylabel('rho_0_norm')
        ax_.set_ylim(0, 1.05)
        ax_.legend(fontsize=9)
        ax_.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("ftl_lwr_profiles.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("=" * 62)
    print("  Convergence FtL → LWR")
    print(f"  Profil  : {PROFILE}")
    print(f"  V_MAX   = {V_MAX} km/h  |  L_VEH = {L_VEH*1000:.0f} m")
    print(f"  RHO_MAX = {RHO_MAX:.0f} veh/km")
    print(f"  rho_c   = {rho_c:.1f} veh/km  (f'(rho_c)=0)")
    print(f"  Q_MAX   = {Q_MAX:.1f} veh/h   (capacité)")
    print(f"  CFL={CFL}")
    print("=" * 62)

    # ── Boucle sur N ──────────────────────────────────────────────────
    # FtL : simulation avec N particules propres à chaque N.
    # LWR : profil continu sur domaine fixe.
    errors_l1 = []
    errors_l2 = []

    for N, T in zip(N_LIST, T_LIST):

        # FtL avec N particules
        t_ftl, X_ftl, X_0_N, _, L_ref_N = simulate_ftl(N, T, PROFILE)

        x_max_N   = float(X_0_N[N]) + V_MAX * T
        NX_N      = NX_PAR_VEH * N
        dx_N      = x_max_N / NX_N
        x_cells_N = np.linspace(dx_N/2, x_max_N - dx_N/2, NX_N)

        s_ref_N, x_ref_N = build_x_from_s(PROFILE, L_ref_N)

        # LWR — domaine et condition initiale fixes
        t_god, rho_lwr = simulate_lwr(
                        T, x_cells_N, s_ref_N, x_ref_N,L_ref_N, PROFILE)
        dt_god = float(t_god[1] - t_god[0])

        # Densité FtL : constante par morceaux sur N segments
        x_seg, rho_seg = ftl_density(X_ftl, t_ftl, t_god)

        # Projection sur grille Godunov (post-traitement pour l'erreur)
        rho_ftl_grid = project_ftl_on_grid(x_seg, rho_seg, x_cells_N)

        e1 = l1_error(rho_ftl_grid, rho_lwr, dx_N, dt_god)
        e2 = l2_error(rho_ftl_grid, rho_lwr, dx_N, dt_god)
        errors_l1.append(e1)
        errors_l2.append(e2)

        print(f"N={N:6d}  T={T:.2f}h  L_ref={L_ref_N:.1f}km  "
                f"L1={e1:.5f}  L2={e2:.5f}")

    # ── Visualisation densités ─────────────────────────────────────────
    N_plot = [50, 1000, 5000]
    idx_plot = [N_LIST.index(n) for n in N_plot]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Convergence FtL → LWR  —  {PROFILE}", fontsize=14)

    for col, (N, idx) in enumerate(zip(N_plot, idx_plot)):
        T = T_LIST[idx]

        t_ftl, X_ftl, X_0_N, _, L_ref_N = simulate_ftl(N, T, PROFILE)
        x_min_N = 0.0
        x_max_N = float(X_0_N[N]) + V_MAX * T
        NX_N = NX_PAR_VEH * N
        dx_N      = x_max_N / NX_N
        x_cells_N = np.linspace(x_min_N + dx_N/2, x_max_N - dx_N/2, NX_N)

        # s(x) pour ce domaine (L_ref de ce N)
        s_ref_N, x_ref_N = build_x_from_s(PROFILE, L_ref_N)

        # LWR sur le domaine de ce N
        t_god_N, rho_lwr_N = simulate_lwr(T, x_cells_N, s_ref_N, x_ref_N,
                                            L_ref_N, PROFILE)
        
        # FtL projeté sur le domaine de ce N
        x_seg_N, rho_seg_N = ftl_density(X_ftl, t_ftl, t_god_N)
        rho_plot_N = project_ftl_on_grid(x_seg_N, rho_seg_N, x_cells_N)

        extent_N = [x_min_N, x_max_N, 0.0, T]

        ax = axes[0, col]
        im = ax.imshow(rho_plot_N, aspect='auto', origin='lower',
                       extent=extent_N, cmap='viridis',
                       vmin=0, vmax=RHO_MAX, interpolation='bilinear')
        ax.set_title(f"FtL  N={N}  T={T:.2f}h", fontsize=11)
        ax.set_xlabel("x [km]"); ax.set_ylabel("t [h]")
        fig.colorbar(im, ax=ax, label="ρ [veh/km]", fraction=0.046)

        ax2 = axes[1, col]
        im2 = ax2.imshow(rho_lwr_N, aspect='auto', origin='lower',
                         extent=extent_N, cmap='viridis',
                         vmin=0, vmax=RHO_MAX, interpolation='bilinear')
        ax2.set_title(f"LWR  N={N}  T={T:.2f}h", fontsize=11)
        ax2.set_xlabel("x [km]"); ax2.set_ylabel("t [h]")
        fig.colorbar(im2, ax=ax2, label="ρ [veh/km]", fraction=0.046)

    plt.tight_layout()
    plt.savefig("ftl_lwr_density.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ── Courbe de convergence ──────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(N_LIST, errors_l1, 'o-', color='steelblue',
              lw=2, markersize=8, label=r"Erreur $L^1(N)$")
    ax.loglog(N_LIST, errors_l2, 's-', color='darkgreen',
              lw=2, markersize=8, label=r"Erreur $L^2(N)$")
    C1 = errors_l1[0] * N_LIST[0]
    ax.loglog(N_LIST, [C1/N for N in N_LIST], '--', color='tomato',
              lw=1.5, label=r"$\mathcal{O}(1/N)$")
    ax.set_xlabel("N (nombre de véhicules)", fontsize=13)
    ax.set_ylabel("Erreur normalisée [veh/km]", fontsize=13)
    ax.set_title(f"Convergence FtL → LWR  —  {PROFILE}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("ftl_lwr_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ── Résumé numérique ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  {'N':>6}  {'T [h]':>6}  {'L=N·L_v':>8}  "
          f"{'Err L1':>10}  {'Ratio L1':>9}  {'Err L2':>10}  {'Ratio L2':>9}")
    print("-" * 72)
    for k, (N, T, e1, e2) in enumerate(zip(N_LIST, T_LIST,
                                            errors_l1, errors_l2)):
        r1 = f"{errors_l1[k-1]/e1:.2f}" if k > 0 else "       —"
        r2 = f"{errors_l2[k-1]/e2:.2f}" if k > 0 else "       —"
        print(f"  {N:>6d}  {T:>6.2f}  {N*L_VEH:>8.3f}  "
              f"{e1:>10.5f}  {r1:>9}  {e2:>10.5f}  {r2:>9}")
    print("=" * 72)
    print("  Ratio attendu ≈ N_k/N_{k-1}  si convergence O(1/N)")
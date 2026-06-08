import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# PARAMÈTRES DE CONFIGURATION ET REPRISE DU STYLE GRAPHIQUE
# =====================================================================
# Répertoire de base identique à votre script principal
BASE_DIR = r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\experimentations\charles\ResNet_alpha"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "graphics", "ground_truth")

# Liste des datasets à traiter
DATASET_FILES = [
    "traffic_N1000_rarefaction.csv",
    "traffic_N1000_shock.csv",
    "traffic_N1000_stop_and_go.csv",
]

# Paramètres de trafic constants (ajustables si besoin)
RHO_MAX = (
    160.0  # Correspond à 1 / 0.00625 km (espacement initial fourni dans l'en-tête)
)


def apply_global_style():
    """Applique exactement le même style graphique que le script ResNet."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "legend.frameon": True,
            "legend.edgecolor": "black",
        }
    )


def process_dataset(filename):
    file_path = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(file_path):
        print(f"[-] Fichier introuvable, passé : {file_path}")
        return

    print(f"\n[+] Traitement du dataset réel : {filename}...")

    # Lecture des données réelles
    df = pd.read_csv(file_path)

    # Nom du sous-dossier basé sur le nom du scénario (ex: traffic_N1000_stop_and_go)
    scenario_name = os.path.splitext(filename)[0]
    graphics_dir = os.path.join(OUTPUT_BASE_DIR, scenario_name)
    os.makedirs(graphics_dir, exist_ok=True)

    # Pivoter les données pour obtenir une matrice [Temps x Véhicules] des positions
    print("    Pivotage et structuration des matrices de trajectoires...")
    df_pivot = df.pivot(index="Time_s", columns="Vehicle_ID", values="Position_km")

    # Tri des colonnes pour s'assurer de l'ordre microscopique des véhicules
    df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)

    # Extraire les axes de temps et de position
    times_s = df_pivot.index.to_numpy()
    times_h = times_s / 3600.0  # Conversion en heures pour correspondre au ResNet
    T_h = times_h[-1] - times_h[0]

    X_grid = df_pivot.to_numpy()  # Matrice de forme (N_timestamps, N_vehicles)

    # Identification des followers et du leader (le véhicule de tête, ID max)
    hist_followers = X_grid[:, :-1]
    hist_leader = X_grid[:, -1]

    # --- 3. DIAGRAMME ESPACE-TEMPS RÉEL (Pour comparaison avec le Graphe 3) ---
    print("    Génération du Graphe 3 : Diagramme Espace-Temps...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Tracé des followers (Code visuel : #1f77b4, lw=1, alpha=0.6)
    for i in range(hist_followers.shape[1]):
        ax3.plot(times_h, hist_followers[:, i], color="#1f77b4", lw=1, alpha=0.6)

    # Tracé du leader (Code visuel : #ff7f0e, lw=2)
    ax3.plot(times_h, hist_leader, color="#ff7f0e", lw=2, label="Leader Réel")

    ax3.set_title(
        f"3. Diagramme Espace-Temps Réel ({scenario_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_xlabel("Temps [h]", fontsize=12)
    ax3.set_ylabel("Position [km]", fontsize=12)
    ax3.legend(loc="upper left")

    fig3.savefig(
        os.path.join(graphics_dir, "3_trajectories_true.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig3)

    # --- 4. PROFIL SPATIAL DE LA DENSITÉ RÉELLE FINALE (Pour comparaison avec le Graphe 4) ---
    print("    Génération du Graphe 4 : Profil de densité finale...")
    X_final = X_grid[-1, :]  # Positions à t = T
    final_gaps = X_final[1:] - X_final[:-1]

    # Densité microscopique réelle brute : 1 véhicule par intervalle (alpha_true = 1)
    density_true = 1.0 / final_gaps

    fig4, ax_main = plt.subplots(figsize=(12, 6))

    # Tracé en escalier de la Vérité Terrain (Code visuel : slategray grisé avec bordure)
    ax_main.stairs(
        density_true,
        X_final,
        baseline=0,
        fill=True,
        color="slategray",
        alpha=0.3,
        label="Vérité Terrain (Dataset)",
    )
    ax_main.stairs(
        density_true, X_final, baseline=None, color="slategray", lw=1.5, linestyle="--"
    )

    ax_main.axhline(
        RHO_MAX,
        color="black",
        linestyle=":",
        lw=1.5,
        label=f"Densité Max ({RHO_MAX:.0f} veh/km)",
    )

    ax_main.set_title(
        f"4. Profil spatial de la densité macroscopique réelle à $t={times_h[-1]:.2f}h$",
        fontsize=14,
        fontweight="bold",
    )
    ax_main.set_xlabel("Position spatiale $x$ sur la route [km]", fontsize=12)
    ax_main.set_ylabel("Densité $\\rho$ [veh/km]", fontsize=12)
    ax_main.set_ylim(0, RHO_MAX * 1.15)
    ax_main.legend(loc="upper right", framealpha=1.0)

    fig4.savefig(
        os.path.join(graphics_dir, "4_final_density_true.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig4)

    # --- 5. HEATMAP SPATIO-TEMPORELLE RÉELLE (Pour comparaison avec le Graphe 5) ---
    print("    Génération du Graphe 5 : Carte de chaleur spatio-temporelle...")
    fig5, ax5 = plt.subplots(figsize=(8, 5))

    # Calcul des mailles temporelles identiques à la structure du ResNet
    T_grid = np.tile(times_h.reshape(-1, 1), (1, X_grid.shape[1]))

    # Calcul des inter-distances et densités pour chaque pas de temps
    gaps_all = X_grid[:, 1:] - X_grid[:, :-1]
    density_all = 1.0 / gaps_all
    density_plot = density_all[
        :-1, :
    ]  # On retire la dernière ligne pour l'alignement flat du pcolormesh

    # Tracé avec la colormap 'magma' pour correspondre exactement à votre modèle
    mesh = ax5.pcolormesh(
        X_grid, T_grid, density_plot, cmap="magma", vmin=0, vmax=RHO_MAX, shading="flat"
    )
    fig5.colorbar(mesh, ax=ax5, label="ρ [veh/km]", fraction=0.046)

    ax5.set_title(
        f"5. Carte de chaleur spatio-temporelle de la Densité Réelle ({scenario_name})",
        fontsize=12,
        fontweight="bold",
    )
    ax5.set_xlabel("Position $x$ [km]", fontsize=12)
    ax5.set_ylabel("Temps $t$ [h]", fontsize=12)

    plt.tight_layout()
    fig5.savefig(
        os.path.join(graphics_dir, "5_spatiotemporal_density_true.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig5)

    print(f"[->] Graphiques sauvegardés dans : {graphics_dir}")


def main():
    apply_global_style()
    for dataset in DATASET_FILES:
        process_dataset(dataset)
    print(
        "\n[+] Opération terminée ! Tous les graphiques des données réelles ont été générés."
    )


if __name__ == "__main__":
    main()

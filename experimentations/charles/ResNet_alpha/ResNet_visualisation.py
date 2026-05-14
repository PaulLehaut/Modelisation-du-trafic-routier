import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# =====================================================================
# CONFIGURATION (Doit correspondre aux paramètres de l'entraînement)
# =====================================================================
PORTION_PROBE = 0.20
EPOCHS = 1000
LEARNING_RATE = 0.5

BASE_DIR = r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\experimentations\charles\ResNet_alpha"

# Reconstruction dynamique du nom de fichier
FILENAME = f"ResNet_probe{PORTION_PROBE}_ep{EPOCHS}_lr{LEARNING_RATE}.pt"
RESULTS_FILE = os.path.join(BASE_DIR, "training_results", FILENAME)


def main():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"Veuillez d'abord lancer l'entraînement. Fichier introuvable:\n{RESULTS_FILE}"
        )

    print("Chargement des résultats...")
    results = torch.load(RESULTS_FILE, weights_only=False)

    # Extraction des variables
    loss_history = results["loss_history"]
    alpha_history_np = results["alpha_history"]
    hist_followers = results["hist_followers"]
    hist_leader = results["hist_leader"]
    times_h = results["times_h"]
    alpha_optimise = results["alpha_optimise"]
    y_target_followers = results["y_target_followers"]
    n_gaps = results["n_gaps"]
    rho_max = results["rho_max"]
    T_h = results["T_h"]

    # Hyperparamètres pour le nommage du dossier
    portion = results["PORTION_PROBE"]
    epochs = results["EPOCHS"]
    lr = results["LEARNING_RATE"]

    # Création du chemin de sauvegarde cible
    dir_name = f"ResNet_probe{portion}_ep{epochs}_lr{lr}"
    graphics_dir = os.path.join(BASE_DIR, "graphics", dir_name)
    os.makedirs(graphics_dir, exist_ok=True)

    print(f"Génération des graphiques dans le dossier :\n{graphics_dir}")
    # Style de base
    plt.style.use("seaborn-v0_8-darkgrid")

    # =================================================================
    # 1. Évolution de la Loss (MSE)
    # =================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(loss_history, color="crimson", lw=2)
    ax1.set_yscale("log")
    ax1.set_title("1. Évolution de la Loss (MSE)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Erreur normalisée [Log]", fontsize=12)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    path1 = os.path.join(graphics_dir, "1_loss_mse.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # =================================================================
    # 2. Convergence des paramètres Alphas
    # =================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(alpha_history_np, alpha=0.6, lw=1.5)
    ax2.set_title(
        "2. Convergence des paramètres $\\alpha_i$", fontsize=14, fontweight="bold"
    )
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Véhicules estimés par segment", fontsize=12)
    ax2.grid(True, ls="--", alpha=0.5)

    path2 = os.path.join(graphics_dir, "2_alphas_convergence.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # =================================================================
    # 3. Diagramme Espace-Temps (Trajectoires)
    # =================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for i in range(hist_followers.shape[1]):
        ax3.plot(times_h, hist_followers[:, i], color="steelblue", lw=1.5, alpha=0.7)
    ax3.plot(times_h, hist_leader, color="darkorange", lw=2, label="Leader")

    target_times = np.full_like(y_target_followers, T_h)
    ax3.scatter(
        target_times,
        y_target_followers,
        color="red",
        marker="x",
        s=60,
        zorder=5,
        label="Vérités terrain (t=T)",
    )

    ax3.set_title(
        "3. Diagramme Espace-Temps (Trajectoires des PVs)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_xlabel("Temps [h]", fontsize=12)
    ax3.set_ylabel("Position [km]", fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, ls="--", alpha=0.5)

    path3 = os.path.join(graphics_dir, "3_trajectories.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # =================================================================
    # 4. Densité finale reconstruite par segment
    # =================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    final_gaps = np.append(hist_followers[-1, 1:], hist_leader[-1]) - hist_followers[-1]
    reconstructed_density = alpha_optimise / final_gaps

    x_pos = np.arange(n_gaps)
    ax4.bar(
        x_pos,
        reconstructed_density,
        color="mediumseagreen",
        edgecolor="black",
        alpha=0.85,
    )
    ax4.axhline(
        rho_max,
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Densité Max ({rho_max:.0f} veh/km)",
    )

    ax4.set_title(
        "4. Densité macroscopique estimée à $t=T$", fontsize=14, fontweight="bold"
    )
    ax4.set_xlabel("Indice du segment (entre PV $i$ et PV $i+1$)", fontsize=12)
    ax4.set_ylabel("Densité $\\rho$ [veh/km]", fontsize=12)
    ax4.set_ylim(0, rho_max * 1.1)
    ax4.legend(fontsize=11)
    ax4.grid(True, axis="y", ls="--", alpha=0.5)

    path4 = os.path.join(graphics_dir, "4_final_density.png")
    fig4.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # =================================================================
    # 5. NOUVEAU : Évolution Spatio-Temporelle de la Densité (Heatmap)
    # =================================================================

    # --- DÉBUT DES MODIFICATIONS : Format de la figure ---
    # Remplacement de figsize=(12, 7) par (8, 5) pour correspondre au ratio des subplots de référence
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    # --- FIN DES MODIFICATIONS ---

    # Création des grilles pour pcolormesh (X = positions, Y = temps)
    # Shape X_grid et T_grid : (num_steps + 1, n_gaps + 1)
    X_grid = np.hstack([hist_followers, hist_leader.reshape(-1, 1)])
    T_grid = np.tile(times_h.reshape(-1, 1), (1, n_gaps + 1))

    # Calcul de la densité à chaque instant
    # Shape gaps_all et density_all : (num_steps + 1, n_gaps)
    gaps_all = X_grid[:, 1:] - X_grid[:, :-1]
    density_all = alpha_optimise / gaps_all

    # Pour 'pcolormesh' avec shading='flat', la matrice des couleurs doit avoir
    # une dimension de moins que les grilles de coordonnées (M, N) vs (M+1, N+1).
    # On retire donc le tout dernier pas de temps pour la couleur.
    density_plot = density_all[:-1, :]

    mesh = ax5.pcolormesh(
        X_grid,
        T_grid,
        density_plot,
        cmap="viridis",
        vmin=0,
        vmax=rho_max,
        shading="flat",
    )

    # --- DÉBUT DES MODIFICATIONS : Standards visuels (Colorbar, Labels, Titre) ---
    # Utilisation d'une seule ligne pour la colorbar avec fraction=0.046 (standard de la réf)
    fig5.colorbar(mesh, ax=ax5, label="ρ [veh/km]", fraction=0.046)

    # Titre simplifié (sans gras) et taille de police ajustée à 11
    ax5.set_title("Carte de chaleur spatio-temporelle de la Densité", fontsize=11)

    # Renommage exact des axes selon la référence
    ax5.set_xlabel("x [km]")
    ax5.set_ylabel("t [h]")

    # (Les lignes de trajectoires ont été retirées pour correspondre au rendu lisse de la référence)

    # Ajout du tight_layout présent dans tous tes blocs de référence
    plt.tight_layout()

    path5 = os.path.join(graphics_dir, "5_spatiotemporal_density.png")

    # Alignement du DPI sur 150 (au lieu de 200)
    fig5.savefig(path5, dpi=150, bbox_inches="tight")
    # --- FIN DES MODIFICATIONS ---

    plt.close(fig5)

    print("Tous les graphiques ont été sauvegardés avec succès !")


if __name__ == "__main__":
    main()

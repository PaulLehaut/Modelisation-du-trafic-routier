import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# =====================================================================
# CONFIGURATION (Doit correspondre aux paramètres de l'entraînement)
# =====================================================================
PORTION_PROBE = 0.10
EPOCHS = 200
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

    print("Génération des graphiques...")
    # Style de base (grille sombre épurée)
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Reconstruction du Trafic par ResNet PINN (PV: {portion * 100:.0f}%, Epochs: {epochs}, LR: {lr})",
        fontsize=16,
        fontweight="bold",
    )

    # --- 1. Évolution de la Loss ---
    ax1 = axes[0, 0]
    ax1.plot(loss_history, color="crimson", lw=2)
    ax1.set_yscale("log")
    ax1.set_title("1. Évolution de la Loss (MSE)", fontsize=13)
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Erreur normalisée [Log]", fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- 2. Évolution des Alphas ---
    ax2 = axes[0, 1]
    ax2.plot(alpha_history_np, alpha=0.6, lw=1.5)
    ax2.set_title("2. Convergence des paramètres $\\alpha_i$", fontsize=13)
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Véhicules estimés par segment", fontsize=11)
    ax2.grid(True, ls="--", alpha=0.5)

    # --- 3. Diagramme Espace-Temps ---
    ax3 = axes[1, 0]
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

    ax3.set_title("3. Diagramme Espace-Temps (Trajectoires)", fontsize=13)
    ax3.set_xlabel("Temps [h]", fontsize=11)
    ax3.set_ylabel("Position [km]", fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, ls="--", alpha=0.5)

    # --- 4. Densité finale reconstruite par segment ---
    ax4 = axes[1, 1]
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

    ax4.set_title("4. Densité macroscopique estimée à $t=T$", fontsize=13)
    ax4.set_xlabel("Indice du segment (entre PV $i$ et PV $i+1$)", fontsize=11)
    ax4.set_ylabel("Densité $\\rho$ [veh/km]", fontsize=11)
    ax4.set_ylim(0, rho_max * 1.1)  # Laisser un peu de marge au-dessus de rho_max
    ax4.legend(fontsize=10)
    ax4.grid(True, axis="y", ls="--", alpha=0.5)

    plt.tight_layout(
        rect=(0, 0.03, 1, 0.96)
    )  # Ajustement pour ne pas superposer le suptitle

    # --- SAUVEGARDE ET AFFICHAGE ---
    save_path = os.path.join(graphics_dir, "training_summary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Graphique sauvegardé avec succès dans :\n{save_path}")

    plt.show()


if __name__ == "__main__":
    main()

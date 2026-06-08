import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# =====================================================================
# PARAMÈTRES DE CHARGEMENT
# =====================================================================
BASE_DIR = r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\experimentations\charles\ResNet_alpha"
PORTION_PROBE = 0.05
EPOCHS = 1000
LOSS_POLICY = "continuous"

FILENAME = "ResNet_epochs1000_lr0.5_method-random_traffic_N1000_stop_and_go.pt"
# FILENAME = f"ResNet_probe{PORTION_PROBE}_ep{EPOCHS}_loss-{LOSS_POLICY}.pt"
RESULTS_FILE = os.path.join(BASE_DIR, "training_results", FILENAME)


def main():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Fichier introuvable:\n{RESULTS_FILE}")

    print("Chargement des résultats...")
    results = torch.load(RESULTS_FILE, weights_only=False)

    loss_history = results["loss_history"]
    alpha_history_np = results["alpha_history"]
    alpha_true = results["alpha_true"]
    density_true = results["density_true"]
    hist_followers = results["hist_followers"]
    hist_leader = results["hist_leader"]
    times_h = results["times_h"]
    alpha_optimise = results["alpha_optimise"]
    y_target_followers = results["y_target_followers"]

    config = results["CONFIG"]
    n_gaps = len(alpha_optimise)
    rho_max = config["RHO_MAX"]
    T_h = times_h[-1] - times_h[0]

    dir_name = f"ResNet_probe{PORTION_PROBE}_ep{EPOCHS}_{LOSS_POLICY}"
    graphics_dir = os.path.join(BASE_DIR, "graphics/graphics", dir_name)
    os.makedirs(graphics_dir, exist_ok=True)

    # Style global plus épuré
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "legend.frameon": True,
            "legend.edgecolor": "black",
        }
    )

    # --- 1. Loss ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(loss_history, color="#d62728", lw=2)  # Rouge brique
    ax1.set_yscale("log")
    ax1.set_title(
        f"1. Évolution de la Loss MSE ({LOSS_POLICY})", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Erreur [Log]", fontsize=12)
    fig1.savefig(
        os.path.join(graphics_dir, "1_loss_mse.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig1)

    # --- 2. Convergence Alphas ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    max_epochs = min(200, alpha_history_np.shape[0])  # sécurité si < 200
    sample_size = min(8, n_gaps)

    colors = plt.cm.Dark2(np.linspace(0, 1, sample_size))

    epochs = np.arange(max_epochs)

    for i in range(sample_size):
        (line,) = ax2.plot(
            epochs, alpha_history_np[:max_epochs, i], color=colors[i], lw=2
        )

        ax2.axhline(
            alpha_true[i],
            color=line.get_color(),
            linestyle="--",
            lw=1.5,
            alpha=0.8,
            label=f"$\\alpha_{i}$ réel/estimé",
        )

    ax2.set_title(
        "2. Convergence des paramètres $\\alpha_i$ (200 premières epochs)",
        fontsize=14,
        fontweight="bold",
    )

    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Nb de véhicules par segment", fontsize=12)

    # Nettoyage légende
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(
        by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc="upper left"
    )

    fig2.savefig(
        os.path.join(graphics_dir, "2_alphas_convergence_200epochs.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig2)

    # --- 3. Trajectoires ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for i in range(hist_followers.shape[1]):
        ax3.plot(times_h, hist_followers[:, i], color="#1f77b4", lw=1, alpha=0.6)
    ax3.plot(times_h, hist_leader, color="#ff7f0e", lw=2, label="Leader")
    target_times = np.full_like(y_target_followers, T_h)
    ax3.scatter(
        target_times,
        y_target_followers,
        color="red",
        marker="x",
        s=50,
        zorder=5,
        label="Vérités terrain (t=T)",
    )
    ax3.set_title("3. Diagramme Espace-Temps", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Temps [h]", fontsize=12)
    ax3.set_ylabel("Position [km]", fontsize=12)
    ax3.legend()
    fig3.savefig(
        os.path.join(graphics_dir, "3_trajectories.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig3)

    # --- 4. LA NOUVELLE COMPARAISON DE DENSITÉ (Tracé en escalier + Résidus) ---
    # Récupération des positions réelles finales pour faire un axe X spatial
    X_final = np.append(hist_followers[-1], hist_leader[-1])
    final_gaps = X_final[1:] - X_final[:-1]
    reconstructed_density = alpha_optimise / final_gaps

    # Création d'une figure à deux étages (Main plot + Erreur)
    fig4, (ax_main, ax_err) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Tracé de la Vérité Terrain (Grisé avec bordure)
    ax_main.stairs(
        density_true,
        X_final,
        baseline=0,
        fill=True,
        color="slategray",
        alpha=0.3,
        label="Vérité Terrain",
    )
    ax_main.stairs(
        density_true, X_final, baseline=None, color="slategray", lw=1.5, linestyle="--"
    )

    # Tracé de la Prédiction (Rouge vif)
    ax_main.stairs(
        reconstructed_density,
        X_final,
        baseline=None,
        color="#d62728",
        lw=2.5,
        label="Prédiction ResNet",
    )

    ax_main.axhline(
        rho_max,
        color="black",
        linestyle=":",
        lw=1.5,
        label=f"Densité Max ({rho_max:.0f} veh/km)",
    )

    ax_main.set_title(
        f"4. Profil spatial de la densité macroscopique à $t={T_h:.2f}h$",
        fontsize=14,
        fontweight="bold",
    )
    ax_main.set_ylabel("Densité $\\rho$ [veh/km]", fontsize=12)
    ax_main.set_ylim(0, rho_max * 1.15)
    ax_main.legend(loc="upper right", framealpha=1.0)

    # Subplot des résidus (Erreur de prédiction)
    error = reconstructed_density - density_true
    # Utilisation d'une colormap divergente basique (rouge si surestimé, bleu si sous-estimé)
    colors_err = ["#d62728" if e > 0 else "#1f77b4" for e in error]

    # On ruse un peu car plt.stairs ne prend pas de tableau de couleurs directement pour le fill
    for i in range(len(error)):
        ax_err.fill_between(
            [X_final[i], X_final[i + 1]],
            0,
            error[i],
            step="post",
            color=colors_err[i],
            alpha=0.6,
        )
        ax_err.plot(
            [X_final[i], X_final[i + 1]],
            [error[i], error[i]],
            color=colors_err[i],
            lw=2,
        )

    ax_err.axhline(0, color="black", lw=1.2)
    ax_err.set_ylabel("Erreur [veh/km]", fontsize=12)
    ax_err.set_xlabel("Position spatiale $x$ sur la route [km]", fontsize=12)

    # Ajout d'une métrique MAE (Mean Absolute Error) globale dans le coin du graphe
    mae = np.mean(np.abs(error))
    ax_err.text(
        0.01,
        0.85,
        f"MAE: {mae:.2f}",
        transform=ax_err.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    plt.tight_layout()
    fig4.savefig(
        os.path.join(graphics_dir, "4_final_density_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig4)

    # --- 5. Heatmap Spatio-Temporelle ---
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    X_grid = np.hstack([hist_followers, hist_leader.reshape(-1, 1)])
    T_grid = np.tile(times_h.reshape(-1, 1), (1, n_gaps + 1))
    gaps_all = X_grid[:, 1:] - X_grid[:, :-1]
    density_all = alpha_optimise / gaps_all
    density_plot = density_all[:-1, :]

    mesh = ax5.pcolormesh(
        X_grid, T_grid, density_plot, cmap="magma", vmin=0, vmax=rho_max, shading="flat"
    )
    fig5.colorbar(mesh, ax=ax5, label="ρ [veh/km]", fraction=0.046)
    ax5.set_title(
        "5. Carte de chaleur spatio-temporelle de la Densité",
        fontsize=12,
        fontweight="bold",
    )
    ax5.set_xlabel("Position $x$ [km]", fontsize=12)
    ax5.set_ylabel("Temps $t$ [h]", fontsize=12)
    plt.tight_layout()
    fig5.savefig(
        os.path.join(graphics_dir, "5_spatiotemporal_density.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig5)

    print("Tous les graphiques ont été sauvegardés avec succès !")


if __name__ == "__main__":
    main()

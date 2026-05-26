import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_DIR = r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\experimentations\charles\ResNet_alpha"
RESULTS_DIR = os.path.join(BASE_DIR, "training_results")


def get_next_comparison_dir(base_dir):
    """Crée et retourne le prochain dossier comparison_{n}"""
    comp_base = os.path.join(base_dir, "graphics", "comparisons")
    os.makedirs(comp_base, exist_ok=True)

    existing_dirs = [d for d in os.listdir(comp_base) if d.startswith("comparison_")]
    next_num = 1
    if existing_dirs:
        # Extrait les numéros et trouve le max
        nums = [
            int(d.split("_")[1]) for d in existing_dirs if d.split("_")[1].isdigit()
        ]
        if nums:
            next_num = max(nums) + 1

    new_dir = os.path.join(comp_base, f"comparison_{next_num}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def main():
    if not os.path.exists(RESULTS_DIR):
        raise FileNotFoundError(f"Dossier de résultats introuvable:\n{RESULTS_DIR}")

    GRAPHICS_DIR = get_next_comparison_dir(BASE_DIR)

    fichiers_modeles = [
        f
        for f in os.listdir(RESULTS_DIR)
        if f.startswith("ResNet") and f.endswith(".pt")
    ]

    if not fichiers_modeles:
        print("Aucun modèle trouvé.")
        return

    print(
        f"[{len(fichiers_modeles)}] fichiers trouvés. Génération des graphiques comparatifs..."
    )

    # Dictionnaire pour stocker les métriques de tous les modèles
    models_data = {}

    for fichier in fichiers_modeles:
        chemin_fichier = os.path.join(RESULTS_DIR, fichier)
        try:
            res = torch.load(chemin_fichier, weights_only=False)
            nom_modele = fichier.replace(".pt", "")

            # Extraction des données basiques
            loss_history = res.get("loss_history")
            if loss_history is None:
                continue

            # Extraction pour les métriques avancées (compatibles avec le nouveau code)
            alpha_hist = res.get("alpha_history")
            alpha_true = res.get("alpha_true")
            density_true = res.get("density_true")
            alpha_opt = res.get("alpha_optimise")
            hist_followers = res.get("hist_followers")
            hist_leader = res.get("hist_leader")

            data = {"loss_history": loss_history}

            # Calcul de l'erreur MAE sur l'historique des Alphas
            if alpha_hist is not None and alpha_true is not None:
                # MAE à chaque epoch : moyenne absolue de (alpha_pred - alpha_true)
                alpha_mae_history = np.mean(np.abs(alpha_hist - alpha_true), axis=1)
                data["alpha_mae_history"] = alpha_mae_history

            # Calcul de la Densité et de son Erreur (MAE Finale)
            if all(v is not None for v in [alpha_opt, hist_followers, hist_leader]):
                X_final = np.append(hist_followers[-1], hist_leader[-1])
                final_gaps = X_final[1:] - X_final[:-1]
                reconstructed_density = alpha_opt / final_gaps

                # --- GESTION DES NANS (DIVERGENCE) ---
                if np.isnan(X_final).any() or np.isnan(reconstructed_density).any():
                    print(
                        f"  [!] AVERTISSEMENT : Le modèle {nom_modele} a divergé (valeurs NaN détectées). Données de densité ignorées."
                    )
                else:
                    data["X_final"] = X_final
                    data["reconstructed_density"] = reconstructed_density

                    if density_true is not None:
                        density_mae = np.mean(
                            np.abs(reconstructed_density - density_true)
                        )
                        data["density_mae"] = density_mae
                        data["density_true"] = density_true  # Gardé pour référence

            models_data[nom_modele] = data

        except Exception as e:
            print(f"Erreur avec le fichier {fichier} : {e}")

    # =================================================================
    # GÉNÉRATION DES GRAPHIQUES
    # =================================================================
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

    # --- 1. Comparaison des Loss MSE ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i, (nom, data) in enumerate(models_data.items()):
        ax1.plot(data["loss_history"], lw=2, alpha=0.8, color=colors[i], label=nom)

    ax1.set_yscale("log")
    ax1.set_title(
        "Comparaison 1 : Évolution de la Loss d'entraînement (MSE)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("MSE Loss [Log]", fontsize=11)
    ax1.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig1.savefig(os.path.join(GRAPHICS_DIR, "comparison_1_loss_history.png"), dpi=150)
    plt.close(fig1)

    # --- 2. Comparaison de la convergence des variables physiques (Alphas) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    has_alpha_data = False
    for i, (nom, data) in enumerate(models_data.items()):
        if "alpha_mae_history" in data:
            has_alpha_data = True
            ax2.plot(
                data["alpha_mae_history"], lw=2, alpha=0.8, color=colors[i], label=nom
            )

    if has_alpha_data:
        ax2.set_yscale("log")
        ax2.set_title(
            "Comparaison 2 : Erreur Moyenne Absolue (MAE) sur les Alphas cachés",
            fontsize=13,
            fontweight="bold",
        )
        ax2.set_xlabel("Epochs", fontsize=11)
        ax2.set_ylabel("MAE vs Vérité Terrain [Log]", fontsize=11)
        ax2.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig2.savefig(
            os.path.join(GRAPHICS_DIR, "comparison_2_alpha_convergence.png"), dpi=150
        )
    plt.close(fig2)

    # --- 3. Comparaison finale des erreurs de Densité (Bar Chart) ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    noms_bar = []
    maes_bar = []
    colors_bar = []
    for i, (nom, data) in enumerate(models_data.items()):
        if "density_mae" in data:
            noms_bar.append(nom)
            maes_bar.append(data["density_mae"])
            colors_bar.append(colors[i])

    if noms_bar:
        bars = ax3.bar(
            noms_bar, maes_bar, color=colors_bar, edgecolor="black", alpha=0.8
        )
        ax3.set_title(
            "Comparaison 3 : Erreur finale sur la Densité (Inférence)",
            fontsize=13,
            fontweight="bold",
        )
        ax3.set_ylabel("Mean Absolute Error (veh/km)", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)

        # Ajout des valeurs au-dessus des barres
        for bar in bars:
            yval = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        fig3.savefig(
            os.path.join(GRAPHICS_DIR, "comparison_3_density_mae.png"), dpi=150
        )
    plt.close(fig3)

    # --- 4. Superposition des profils de Densité Spatiale ---
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    has_density_data = False

    # On trace la vérité terrain du premier modèle trouvé comme référence globale (en grisé)
    true_density_plotted = False

    for i, (nom, data) in enumerate(models_data.items()):
        if "reconstructed_density" in data and "X_final" in data:
            has_density_data = True

            # Tracé de la prédiction du modèle en escalier
            ax4.stairs(
                data["reconstructed_density"],
                data["X_final"],
                baseline=None,
                color=colors[i],
                lw=2,
                label=nom,
            )

            # Tracé de la vérité terrain (seulement une fois pour ne pas surcharger)
            if not true_density_plotted and "density_true" in data:
                ax4.stairs(
                    data["density_true"],
                    data["X_final"],
                    baseline=0,
                    fill=True,
                    color="gray",
                    alpha=0.2,
                    label="Vérité Terrain (Réf)",
                )
                ax4.stairs(
                    data["density_true"],
                    data["X_final"],
                    baseline=None,
                    color="gray",
                    lw=1.5,
                    linestyle="--",
                )
                true_density_plotted = True

    if has_density_data:
        ax4.set_title(
            "Comparaison 4 : Profils spatiaux de la densité reconstruite à $t=T$",
            fontsize=13,
            fontweight="bold",
        )
        ax4.set_xlabel("Position spatiale $x$ sur la route [km]", fontsize=11)
        ax4.set_ylabel("Densité $\\rho$ [veh/km]", fontsize=11)
        ax4.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig4.savefig(
            os.path.join(GRAPHICS_DIR, "comparison_4_density_profiles.png"), dpi=150
        )
    plt.close(fig4)

    print(f"Tous les graphiques comparatifs ont été générés dans :\n{GRAPHICS_DIR}")


if __name__ == "__main__":
    main()

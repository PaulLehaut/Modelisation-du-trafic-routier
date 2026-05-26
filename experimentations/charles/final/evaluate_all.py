# Fichier : experimentations/charles/final/evaluate_all.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from ResNet_training import train_resnet
from pinn import train_pinn_with_weights, generate_all_collocation_epochs

# ==========================================
# CONFIGURATION GLOBALE
# ==========================================
DATA_PATH = r"data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv"
RESULTS_DIR = r"experimentations\charles\final\training_results"
GRAPHICS_DIR = r"experimentations\charles\final\graphics"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHICS_DIR, exist_ok=True)

EPOCHS = 1000  # Réduit pour test, ajuste selon tes besoins
SEED = 42

resnet_config_base = {
    "DATA_PATH": DATA_PATH,
    "PORTION_PROBE": 0.2,
    "EPOCHS": EPOCHS,
    "LEARNING_RATE": 0.5,
    "N_TOTAL": 1000,
    "L_V": 0.005,
    "V_MAX": 50.0,
    "RHO_MAX": 200.0,
    "SEED": SEED,
}


# ==========================================
# FONCTIONS DE VISUALISATION
# ==========================================
def plot_losses(results_dict):
    """Génère un graphique avec la loss d'entrainement et de validation (CEE)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for name, res in results_dict.items():
        # Lissage de la loss d'entrainement pour la lisibilité
        epochs_train = np.arange(len(res["loss_train"]))
        axes[0].plot(epochs_train, res["loss_train"], label=name, alpha=0.8)

        # CEE est calculée tous les X epochs
        epochs_cee = np.linspace(
            0, len(res["loss_train"]) - 1, len(res["cee_val"])
        )  # [MODIFICATION] Correction de l'alignement des axes (N-1)
        axes[1].plot(epochs_cee, res["cee_val"], label=name, linestyle="--")

    axes[0].set_yscale("log")
    axes[0].set_title("Training Loss Evolution")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_yscale("log")
    axes[1].set_title("Validation CEE (Current Estimation Error)")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "loss_comparison.png"))
    plt.close()


def plot_final_density(results_dict):
    """Compare la densité finale réelle vs modèles"""
    plt.figure(figsize=(12, 6))

    # On prend la vraie densité du premier modèle (ils partagent tous la même)
    true_density = list(results_dict.values())[0]["rho_final_true"]
    x_axis = np.arange(len(true_density))

    plt.plot(
        x_axis,
        true_density,
        label="Ground Truth (Réelle)",
        color="black",
        linewidth=2,
        linestyle=":",
    )

    colors = ["blue", "cyan", "red", "orange"]
    for (name, res), color in zip(results_dict.items(), colors):
        plt.plot(
            x_axis, res["rho_final_pred"], label=f"Pred: {name}", alpha=0.7, color=color
        )

    plt.title("Comparaison de la Densité Spatiale Finale (t=T)")
    plt.xlabel("Index de Segment / Espace")
    plt.ylabel("Densité (veh/km)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPHICS_DIR, "final_density_comparison.png"))
    plt.close()


def plot_spatiotemporal_evolution(results_dict, pinn_models, df_path, device):
    """Affiche l'évolution de la densité via des coupes à des instants précis de la simulation"""
    print("\nGénération des coupes spatio-temporelles...")
    df = pd.read_csv(df_path)
    df["Density"] = 1.0 - (df["Velocity_kmh"] / 50.0)  # V_MAX = 50.0

    t_max = (df["Time_s"] / 3600.0).max()
    x_max = df["Position_km"].max()

    # Fractions du temps total N demandées
    fractions = [0.0, 1 / 3, 1 / 2, 2 / 3, 1.0]
    labels = ["t = 0", "t = N/3", "t = N/2", "t = 2N/3", "t = N"]

    fig, axes = plt.subplots(len(fractions), 1, figsize=(14, 18), sharex=True)

    # Récupération du nombre de pas et du nombre de segments via le ResNet
    res_base = results_dict.get("ResNet_Colleau") or results_dict.get("ResNet_Random")
    num_steps = len(res_base["intermediate_rhos"])
    n_gaps = len(res_base["rho_final_pred"])

    # Grille spatiale normalisée pour l'inférence des PINNs
    x_grid = np.linspace(0, x_max, n_gaps)
    x_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1).to(device)

    for i, (frac, label) in enumerate(zip(fractions, labels)):
        ax = axes[i]

        # 1. Calcul des index et temps physiques
        step_idx = int(frac * (num_steps - 1))
        t_phys = frac * t_max

        # 2. Trace de la Vérité Terrain (Ground Truth) approximée
        t_s_target = t_phys * 3600.0
        closest_time = df.iloc[(df["Time_s"] - t_s_target).abs().argsort()[:1]][
            "Time_s"
        ].values[0]
        df_t = df[df["Time_s"] == closest_time].sort_values("Position_km")

        if not df_t.empty:
            true_rho = df_t["Density"].values
            ax.plot(
                np.linspace(0, n_gaps, len(true_rho)),
                true_rho,
                "k:",
                label="Vérité Terrain",
                linewidth=2.5,
            )

        # 3. Trace des densités ResNet
        for name, res in results_dict.items():
            if "ResNet" in name:
                rho_resnet = res["intermediate_rhos"][step_idx]
                ax.plot(
                    np.arange(len(rho_resnet)),
                    rho_resnet,
                    label=f"Pred: {name}",
                    alpha=0.8,
                )

        # 4. Trace des densités PINNs
        t_tensor = (torch.ones_like(x_tensor) * t_phys).to(device)
        for name, model in pinn_models.items():
            model.eval()
            with torch.no_grad():
                rho_pinn = model(t_tensor, x_tensor).cpu().numpy().flatten()
            ax.plot(
                np.arange(len(rho_pinn)),
                rho_pinn,
                label=f"Pred: {name}",
                linestyle="--",
                alpha=0.9,
            )

        ax.set_title(
            f"Coupe Spatiale à {label} (Temps = {t_phys * 3600:.1f}s)", fontsize=12
        )
        ax.set_ylabel("Densité (veh/km)")
        ax.grid(True, alpha=0.4)
        if i == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.05))

    axes[-1].set_xlabel("Index de Segment / Espace", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "spatiotemporal_evolution_snapshots.png"))
    plt.close()


# ==========================================
# EXECUTION PRINCIPALE
# ==========================================
def main():
    results = {}
    pinn_models = {}

    # 1. ResNet - Random
    print("\n" + "=" * 40)
    print("--- Entrainement ResNet (Random) ---")
    conf_rr = resnet_config_base.copy()
    conf_rr["METHOD"] = "random"
    results["ResNet_Random"] = train_resnet(conf_rr)
    # [AJOUT] Print de vérification du succès de l'entraînement
    print(f" -> Terminé en {results['ResNet_Random']['train_time']:.2f}s")
    print(
        f" -> Loss finale: {results['ResNet_Random']['loss_train'][-1]:.6f} | CEE validation: {results['ResNet_Random']['cee_val'][-1]:.6f}"
    )

    # 2. ResNet - Colleau (Pondéré)
    print("\n" + "=" * 40)
    print("--- Entrainement ResNet (Colleau) ---")
    conf_rc = resnet_config_base.copy()
    conf_rc["METHOD"] = "colleau"
    results["ResNet_Colleau"] = train_resnet(conf_rc)
    # [AJOUT] Print de vérification du succès de l'entraînement
    print(f" -> Terminé en {results['ResNet_Colleau']['train_time']:.2f}s")
    print(
        f" -> Loss finale: {results['ResNet_Colleau']['loss_train'][-1]:.6f} | CEE validation: {results['ResNet_Colleau']['cee_val'][-1]:.6f}"
    )

    # Préparation données PINN
    print("\nPréparation des tenseurs pour les modèles PINNs...")
    df_temp = pd.read_csv(DATA_PATH)
    t_max = (df_temp["Time_s"] / 3600.0).max()
    x_max = df_temp["Position_km"].max()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_colloc, x_colloc = generate_all_collocation_epochs(
        2000, EPOCHS, t_max, x_max, device
    )

    # 3. PINN - Random
    print("\n" + "=" * 40)
    print("--- Entrainement PINN (Random) ---")
    pinn_model_r, e_log_r, l_tot_r, _, _, cee_r, t_train_r = train_pinn_with_weights(
        DATA_PATH, t_colloc, x_colloc, epochs=EPOCHS, use_adaptive=False, seed=SEED
    )
    pinn_models["PINN_Random"] = pinn_model_r

    # Mesure Inférence PINN
    start_inf = time.time()
    _ = pinn_model_r(
        # [CORRECTION] Ajout explicite de dtype=torch.float32
        torch.tensor([[t_max]], dtype=torch.float32).to(device),
        torch.tensor([[x_max]], dtype=torch.float32).to(device),
    )
    t_inf_r = time.time() - start_inf

    results["PINN_Random"] = {
        "loss_train": l_tot_r,
        "cee_val": cee_r,
        "train_time": t_train_r,
        "inf_time": t_inf_r,
        "cee_final": cee_r[-1],
        "params": sum(p.numel() for p in pinn_model_r.parameters()),
        "rho_final_pred": np.zeros(
            len(results["ResNet_Random"]["rho_final_true"])
        ),  # Placeholder formel
        "rho_final_true": results["ResNet_Random"]["rho_final_true"],
    }
    # [AJOUT] Print de vérification du succès de l'entraînement
    print(f" -> Terminé en {t_train_r:.2f}s")
    print(f" -> Loss finale: {l_tot_r[-1]:.6f} | CEE validation: {cee_r[-1]:.6f}")

    # 4. PINN - Colleau (Pondéré)
    print("\n" + "=" * 40)
    print("--- Entrainement PINN (Colleau) ---")
    pinn_model_c, e_log_c, l_tot_c, _, _, cee_c, t_train_c = train_pinn_with_weights(
        DATA_PATH, t_colloc, x_colloc, epochs=EPOCHS, use_adaptive=True, seed=SEED
    )
    pinn_models["PINN_Colleau"] = pinn_model_c

    results["PINN_Colleau"] = {
        "loss_train": l_tot_c,
        "cee_val": cee_c,
        "train_time": t_train_c,
        "inf_time": t_inf_r,
        "cee_final": cee_c[-1],
        "params": sum(p.numel() for p in pinn_model_c.parameters()),
        "rho_final_pred": np.zeros(len(results["ResNet_Random"]["rho_final_true"])),
        "rho_final_true": results["ResNet_Random"]["rho_final_true"],
    }
    # [AJOUT] Print de vérification du succès de l'entraînement
    print(f" -> Terminé en {t_train_c:.2f}s")
    print(f" -> Loss finale: {l_tot_c[-1]:.6f} | CEE validation: {cee_c[-1]:.6f}")

    # Sauvegarde PT
    print("\nSauvegarde des tenseurs globaux...")
    torch.save(results, os.path.join(RESULTS_DIR, "all_models_comparisons.pt"))

    # Graphiques
    print("Génération des graphiques...")
    plot_losses(results)
    plot_final_density({k: v for k, v in results.items() if "ResNet" in k})
    plot_spatiotemporal_evolution(results, pinn_models, DATA_PATH, device)

    # ==========================================
    # GENERATION DU CODE LATEX
    # ==========================================
    print("\n" + "=" * 50)
    print("CODE LATEX À COPIER :")
    print("=" * 50)
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\renewcommand{{\\arraystretch}}{{1.4}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{|l|c|c|c|c|c|}}
\\hline
\\textbf{{Méthode}} & \\textbf{{Initialisation}} & \\textbf{{CEE}} & \\textbf{{Temps d’inférence}} & \\textbf{{Temps d’entraînement}} & \\textbf{{Paramètres / poids}} \\\\
\\hline
ResNet & Seed={SEED} & {results["ResNet_Random"]["cee_final"]:.6f} & {results["ResNet_Random"]["inf_time"]:.4f}s & {results["ResNet_Random"]["train_time"]:.1f}s & {results["ResNet_Random"]["params"]} \\\\
\\hline
ResNet avec méthode Colleau & Seed={SEED} & {results["ResNet_Colleau"]["cee_final"]:.6f} & {results["ResNet_Colleau"]["inf_time"]:.4f}s & {results["ResNet_Colleau"]["train_time"]:.1f}s & {results["ResNet_Colleau"]["params"]} \\\\
\\hline
PINNs & Seed={SEED} & {results["PINN_Random"]["cee_final"]:.6f} & {results["PINN_Random"]["inf_time"]:.4f}s & {results["PINN_Random"]["train_time"]:.1f}s & {results["PINN_Random"]["params"]} \\\\
\\hline
PINNs avec méthode Colleau & Seed={SEED} & {results["PINN_Colleau"]["cee_final"]:.6f} & {results["PINN_Colleau"]["inf_time"]:.4f}s & {results["PINN_Colleau"]["train_time"]:.1f}s & {results["PINN_Colleau"]["params"]} \\\\
\\hline
\\end{{tabular}}
}}
\\caption{{Comparaison des performances entre ResNet et PINN avec échantillonnage aléatoire vs méthode Colleau}}
\\end{{table}}"""
    print(latex_table)


if __name__ == "__main__":
    main()

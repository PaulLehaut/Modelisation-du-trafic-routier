import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # [MODIFICATION] Correction de l'import seaborn (sns au lieu de pd)
import pandas as pd
import time
import hashlib
import json

from ResNet_training import train_resnet
from pinn import train_pinn_with_weights, generate_all_collocation_epochs

# ==========================================
# CONFIGURATION GLOBALE
# ==========================================
DATA_PATH = r"data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv"
RESULTS_DIR = r"experimentations\charles\final\training_results"
GRAPHICS_DIR = r"experimentations\charles\final\graphics"
CACHE_DIR = r"experimentations\charles\final\models_cache"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHICS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

EPOCHS = 2000
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
# FONCTIONS GESTION DU CACHE ET UTILITAIRES
# ==========================================
def get_cache_path(model_name, config_dict):
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{model_name}_{config_hash}.pt")


def compute_model_weight_kb(model_or_params):
    """Calcule le poids d'un modèle en kilo-octets (Ko)."""
    if isinstance(model_or_params, int):
        return (model_or_params * 4) / 1024
    else:
        param_size = sum(
            p.nelement() * p.element_size() for p in model_or_params.parameters()
        )
        buffer_size = sum(
            b.nelement() * b.element_size() for b in model_or_params.buffers()
        )
        return (param_size + buffer_size) / 1024


# ==========================================
# FONCTIONS DE VISUALISATION
# ==========================================
def plot_losses(results_dict):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for name, res in results_dict.items():
        epochs_train = np.arange(len(res["loss_train"]))
        axes[0].plot(epochs_train, res["loss_train"], label=name, alpha=0.8)

        epochs_cee = np.linspace(0, len(res["loss_train"]) - 1, len(res["cee_val"]))
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
    plt.savefig(os.path.join(GRAPHICS_DIR, "loss_comparison_final.png"))
    plt.close()


def plot_final_density(results_dict):
    plt.figure(figsize=(12, 6))
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

    colors = sns.color_palette("husl", len(results_dict))
    for (name, res), color in zip(results_dict.items(), colors):
        if "rho_final_pred" in res and res["rho_final_pred"] is not None:
            plt.plot(
                x_axis,
                res["rho_final_pred"],
                label=f"Pred: {name}",
                alpha=0.7,
                color=color,
            )

    plt.title("Comparaison de la Densité Spatiale Finale (t=T)")
    plt.xlabel("Index de Segment / Espace")
    plt.ylabel("Densité (veh/km)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPHICS_DIR, "final_density_comparison_final.png"))
    plt.close()


def plot_spatiotemporal_evolution(results_dict, pinn_models, df_path, device):
    print("\nGénération des coupes spatio-temporelles...")
    df = pd.read_csv(df_path)
    df["Density"] = (1.0 - (df["Velocity_kmh"] / 50.0)) * resnet_config_base["RHO_MAX"]

    t_max = (df["Time_s"] / 3600.0).max()
    x_max = df["Position_km"].max()
    fractions = [0.0, 1 / 3, 1 / 2, 2 / 3, 1.0]
    labels = ["t = 0", "t = N/3", "t = N/2", "t = 2N/3", "t = N"]

    fig, axes = plt.subplots(len(fractions), 1, figsize=(14, 18), sharex=True)

    res_base = results_dict.get("ResNet_Adaptatif") or results_dict.get("ResNet_Random")
    num_steps = len(res_base["intermediate_rhos"])
    n_gaps = len(res_base["rho_final_pred"])

    x_grid = np.linspace(0, x_max, n_gaps)
    x_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1).to(device)

    for i, (frac, label) in enumerate(zip(fractions, labels)):
        ax = axes[i]
        step_idx = int(frac * (num_steps - 1))
        t_phys = frac * t_max

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

        for name, res in results_dict.items():
            if "ResNet" in name:
                rho_resnet = res["intermediate_rhos"][step_idx]
                ax.plot(
                    np.arange(len(rho_resnet)),
                    rho_resnet,
                    label=f"Pred: {name}",
                    alpha=0.8,
                )

        t_tensor = (torch.ones_like(x_tensor) * t_phys).to(device)
        for name, model in pinn_models.items():
            model.eval()
            with torch.no_grad():
                rho_pinn = (
                    model(t_tensor, x_tensor).cpu().numpy().flatten()
                    * resnet_config_base["RHO_MAX"]
                )
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
    plt.savefig(os.path.join(GRAPHICS_DIR, "spatiotemporal_evolution_final.png"))
    plt.close()


# ==========================================
# EXECUTION PRINCIPALE
# ==========================================
def main():
    results = {}
    pinn_models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ RESNET RANDOM ------------------
    print("\n" + "=" * 40)
    print("--- Entrainement ResNet (Standard) ---")
    conf_rr = resnet_config_base.copy()
    conf_rr["METHOD"] = "random"

    cache_path_rr = get_cache_path("ResNet_Random", conf_rr)
    if os.path.exists(cache_path_rr):
        results["ResNet_Random"] = torch.load(
            cache_path_rr, map_location=device, weights_only=False
        )
        print(" -> Chargé depuis le cache.")
    else:
        results["ResNet_Random"] = train_resnet(conf_rr)
        torch.save(results["ResNet_Random"], cache_path_rr)

    results["ResNet_Random"]["weight_kb"] = compute_model_weight_kb(
        results["ResNet_Random"]["params"]
    )

    # ------------------ RESNET ADAPTATIF ------------------
    print("\n" + "=" * 40)
    # [MODIFICATION] Renommage de Colleau à Adaptatif
    print("--- Entrainement ResNet (Adaptatif) ---")
    conf_rc = resnet_config_base.copy()
    conf_rc["METHOD"] = "adaptative"  # [MODIFICATION] Renommage

    cache_path_rc = get_cache_path("ResNet_Adaptatif", conf_rc)
    if os.path.exists(cache_path_rc):
        results["ResNet_Adaptatif"] = torch.load(
            cache_path_rc, map_location=device, weights_only=False
        )
        print(" -> Chargé depuis le cache.")
    else:
        results["ResNet_Adaptatif"] = train_resnet(conf_rc)
        torch.save(results["ResNet_Adaptatif"], cache_path_rc)

    results["ResNet_Adaptatif"]["weight_kb"] = compute_model_weight_kb(
        results["ResNet_Adaptatif"]["params"]
    )

    # Préparation données PINNs
    df_temp = pd.read_csv(DATA_PATH)
    t_max = (df_temp["Time_s"] / 3600.0).max()
    x_max = df_temp["Position_km"].max()

    t_colloc, x_colloc = generate_all_collocation_epochs(
        2000, EPOCHS, t_max, x_max, device
    )

    # ------------------ PINNS (ABLATION STUDY) ------------------
    # [MODIFICATION] Remplacement des anciens PINNs par les 3 configurations d'Ablation
    pinn_configs_to_test = [
        {
            "name": "PINN_Hybride_Optimal",
            "mu": 0.99,
            "gamma": 0.05,
            "desc": "$\mu=0.99, \gamma=0.05$",
        },
        {
            "name": "PINN_Donnees_Pures",
            "mu": 1.0,
            "gamma": 0.05,
            "desc": "$\mu=1.0$ (Sans physique)",
        },
        {
            "name": "PINN_Physique_Pure",
            "mu": 0.0,
            "gamma": 0.05,
            "desc": "$\mu=0.0$ (Sans données)",
        },
    ]

    for p_conf in pinn_configs_to_test:
        name = p_conf["name"]
        print("\n" + "=" * 40)
        print(f"--- Entrainement {name} ---")

        cache_dict = {
            "epochs": EPOCHS,
            "rho_max": resnet_config_base["RHO_MAX"],
            "use_adaptive": True,  # On utilise la méthode adaptative par défaut pour tous les PINNs comparatifs
            "seed": SEED,
            "n_colloc": 2000,
            "data_path": DATA_PATH,
            "mu": p_conf["mu"],
            "gamma": p_conf["gamma"],
        }
        cache_path = get_cache_path(name, cache_dict)

        if os.path.exists(cache_path):
            cached_model = torch.load(
                cache_path, map_location=device, weights_only=False
            )
            pinn_models[name] = cached_model["model"]
            results[name] = cached_model["results"]
            print(f" -> Chargé depuis le cache.")
        else:
            pinn_model, e_log, l_tot, _, _, cee_hist, t_train = train_pinn_with_weights(
                DATA_PATH,
                t_colloc,
                x_colloc,
                epochs=EPOCHS,
                mu=p_conf["mu"],
                gamma=p_conf["gamma"],
                rho_max=resnet_config_base["RHO_MAX"],
                use_adaptive=True,
                seed=SEED,
                selection_method="adaptative",  # [MODIFICATION] Renommage
            )
            pinn_models[name] = pinn_model

            start_inf = time.time()
            _ = pinn_model(
                torch.tensor([[t_max]], dtype=torch.float32).to(device),
                torch.tensor([[x_max]], dtype=torch.float32).to(device),
            )
            t_inf = time.time() - start_inf

            # Inférence spatiale finale pour la visualisation
            n_gaps_val = len(results["ResNet_Random"]["rho_final_true"])
            x_grid = np.linspace(0, x_max, n_gaps_val)
            x_tensor_grid = (
                torch.tensor(x_grid, dtype=torch.float32).view(-1, 1).to(device)
            )
            t_tensor_grid = (torch.ones_like(x_tensor_grid) * t_max).to(device)
            with torch.no_grad():
                rho_final_pred = (
                    pinn_model(t_tensor_grid, x_tensor_grid).cpu().numpy().flatten()
                    * resnet_config_base["RHO_MAX"]
                )

            results[name] = {
                "loss_train": l_tot,
                "cee_val": cee_hist,
                "train_time": t_train,
                "inf_time": t_inf,
                "cee_final": cee_hist[-1],
                "params": sum(p.numel() for p in pinn_model.parameters()),
                "weight_kb": compute_model_weight_kb(pinn_model),
                "rho_final_pred": rho_final_pred,
                "rho_final_true": results["ResNet_Random"]["rho_final_true"],
            }
            torch.save({"model": pinn_model, "results": results[name]}, cache_path)
            print(
                f" -> Entraînement terminé en {t_train:.1f}s | CEE Finale: {cee_hist[-1]:.2f}"
            )

    print("\nGénération des graphiques...")
    plot_losses(results)
    plot_final_density(results)
    plot_spatiotemporal_evolution(results, pinn_models, DATA_PATH, device)

    # ==========================================
    # GENERATION DU CODE LATEX
    # ==========================================
    # [MODIFICATION] Intégration des 5 modèles dans le tableau final (ResNet vs Ablation PINNs)
    print("\n" + "=" * 50)
    print("CODE LATEX À COPIER :")
    print("=" * 50)
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\renewcommand{{\\arraystretch}}{{1.5}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{|l|c|c|c|c|c|p{{4.0cm}}|}}
\\hline
\\textbf{{Méthode}} & \\textbf{{Pondération sondes}} & \\textbf{{CEE}} & \\textbf{{Temps d'inférence}} & \\textbf{{Temps d'entraînement}} & \\textbf{{Poids modèle}} & \\textbf{{Paramètres spécifiques}} \\\\
\\hline
ResNet & Standard & {results["ResNet_Random"]["cee_final"]:.6f} & {results["ResNet_Random"]["inf_time"]:.4f} s & {results["ResNet_Random"]["train_time"]:.1f} s & {results["ResNet_Random"]["weight_kb"]:.2f} Ko & $N=1000$, LR $={resnet_config_base["LEARNING_RATE"]}$ \\\\
\\hline
ResNet & Adaptative & {results["ResNet_Adaptatif"]["cee_final"]:.6f} & {results["ResNet_Adaptatif"]["inf_time"]:.4f} s & {results["ResNet_Adaptatif"]["train_time"]:.1f} s & {results["ResNet_Adaptatif"]["weight_kb"]:.2f} Ko & $N=1000$, LR $={resnet_config_base["LEARNING_RATE"]}$ \\\\
\\hline
PINN Hybride Optimal & Adaptative & {results["PINN_Hybride_Optimal"]["cee_final"]:.6f} & {results["PINN_Hybride_Optimal"]["inf_time"]:.4f} s & {results["PINN_Hybride_Optimal"]["train_time"]:.1f} s & {results["PINN_Hybride_Optimal"]["weight_kb"]:.2f} Ko & $\mu=0.99, \gamma=0.05$ \\\\
\\hline
PINN Données Pures & Adaptative & {results["PINN_Donnees_Pures"]["cee_final"]:.6f} & {results["PINN_Donnees_Pures"]["inf_time"]:.4f} s & {results["PINN_Donnees_Pures"]["train_time"]:.1f} s & {results["PINN_Donnees_Pures"]["weight_kb"]:.2f} Ko & $\mu=1.0$ (Régression pure) \\\\
\\hline
PINN Physique Pure & N/A & {results["PINN_Physique_Pure"]["cee_final"]:.6f} & {results["PINN_Physique_Pure"]["inf_time"]:.4f} s & {results["PINN_Physique_Pure"]["train_time"]:.1f} s & {results["PINN_Physique_Pure"]["weight_kb"]:.2f} Ko & $\mu=0.0$ (LWR strict) \\\\
\\hline
\\end{{tabular}}
}}
\\caption{{Comparaison des performances : ResNet face aux différentes configurations de réseaux physiques (PINNs). Paramètres communs : Rarefaction $p={resnet_config_base["PORTION_PROBE"]}$, {EPOCHS} epochs, Seed={SEED}.}}
\\label{{tab:comparaison_resnet_pinn_ablation}}
\\end{{table}}"""
    print(latex_table)


if __name__ == "__main__":
    main()

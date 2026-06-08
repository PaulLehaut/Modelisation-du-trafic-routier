import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json

CACHE_DIR = r"experimentations\charles\final\models_cache"
GRAPHICS_DIR = r"experimentations\charles\final\graphics\pinn"
os.makedirs(GRAPHICS_DIR, exist_ok=True)

DATA_PATH = r"data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv"
EPOCHS = 1000
SEED = 42
RHO_MAX = 200.0


def get_cache_path(model_name, config_dict):
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{model_name}_{config_hash}.pt")


def plot_pinn_internal_dynamics(name, res_dict):
    epochs_logged = np.arange(len(res_dict["loss_train"]))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        epochs_logged,
        res_dict["loss_train"],
        label="Total Loss",
        color="black",
        linewidth=2,
    )

    if "loss_data_history" in res_dict:
        ax1.plot(
            epochs_logged,
            res_dict["loss_data_history"],
            label="Data Loss",
            linestyle="--",
            color="blue",
        )
    if "loss_phys_history" in res_dict:
        ax1.plot(
            epochs_logged,
            res_dict["loss_phys_history"],
            label="Physics Loss",
            linestyle=":",
            color="green",
        )

    ax1.set_yscale("log")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Losses")
    ax1.grid(True, alpha=0.4)

    ax2 = ax1.twinx()
    epochs_cee = np.linspace(
        epochs_logged[0], epochs_logged[-1], len(res_dict["cee_val"])
    )
    ax2.plot(
        epochs_cee,
        res_dict["cee_val"],
        label="Validation CEE",
        color="red",
        linewidth=2,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("CEE", color="red")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.title(f"Dynamique d'apprentissage interne - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, f"{name.lower()}_dynamics.png"))
    plt.close()


def plot_pinn_grid_search_ablation(device):
    configs = [
        {"name": "PINN_Physique_Pure", "mu": 0.0, "color": "black", "ls": ":"},
        {"name": "PINN_Hybride_Optimal", "mu": 0.99, "color": "purple", "ls": "-"},
        {"name": "PINN_Donnees_Pures", "mu": 1.0, "color": "red", "ls": "--"},
    ]
    plt.figure(figsize=(10, 6))

    for c in configs:
        conf = {
            "epochs": EPOCHS,
            "rho_max": RHO_MAX,
            "use_adaptive": True,
            "seed": SEED,
            "n_colloc": 2000,
            "data_path": DATA_PATH,
            "mu": c["mu"],
            "gamma": 0.05,
            "p": 0.05,
        }
        cache_path = get_cache_path(c["name"], conf)
        if os.path.exists(cache_path):
            res = torch.load(cache_path, map_location=device, weights_only=False)[
                "results"
            ]
            epochs_cee = np.linspace(0, len(res["loss_train"]), len(res["cee_val"]))
            plt.plot(
                epochs_cee,
                res["cee_val"],
                label=f"{c['name']} ($\\mu={c['mu']}$)",
                color=c["color"],
                linestyle=c["ls"],
                linewidth=2,
            )

    plt.yscale("log")
    plt.title("Ablation Study (Grid Search) : Évaluation de la régularisation $\\mu$")
    plt.xlabel("Epochs")
    plt.ylabel("Validation CEE")

    # Ajout du commentaire formel demandé
    comment = (
        "Régularisation : $\\mu=0.99$ offre le meilleur compromis.\n"
        "La physique ($\\mu=0$) seule diverge car mal posée.\n"
        "Les données seules ($\\mu=1$) surapprennent."
    )
    plt.text(
        0.5,
        0.1,
        comment,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "pinn_grid_search_ablation.png"))
    plt.close()


def plot_pinn_density_reconstruction_and_heatmap(
    name, model, t_max, x_max, res_dict, device
):
    # 1. HEATMAP (Couleurs)
    t_grid = np.linspace(0, t_max, 100)
    x_grid = np.linspace(0, x_max, 200)
    T, X = np.meshgrid(t_grid, x_grid)

    T_tensor = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    X_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    model.eval()
    with torch.no_grad():
        Rho_pred = model(T_tensor, X_tensor).cpu().numpy() * RHO_MAX
    Rho_pred = Rho_pred.reshape(T.shape)

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, T * 3600, Rho_pred, levels=50, cmap="viridis")
    cbar = plt.colorbar(contour)
    cbar.set_label("Densité $\\rho$ (veh/km)")
    plt.title(f"Reconstruction de la densité (Heatmap) - {name}")
    plt.xlabel("Position (km)")
    plt.ylabel("Temps (secondes)")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, f"{name.lower()}_heatmap.png"))
    plt.close()

    # 2. COURBE t=T vs REELLE
    if "rho_final_true" in res_dict:
        plt.figure(figsize=(10, 6))
        x_axis = np.linspace(0, x_max, len(res_dict["rho_final_true"]))

        # Inférence à t=T
        T_final = torch.ones((len(x_axis), 1), dtype=torch.float32).to(device) * t_max
        X_final = torch.tensor(x_axis, dtype=torch.float32).view(-1, 1).to(device)
        with torch.no_grad():
            Rho_final = model(T_final, X_final).cpu().numpy().flatten() * RHO_MAX

        plt.plot(
            np.arange(len(res_dict["rho_final_true"])),
            res_dict["rho_final_true"],
            label="Densité Réelle",
            color="black",
            linestyle=":",
            linewidth=2,
        )
        plt.plot(
            np.arange(len(res_dict["rho_final_true"])),
            Rho_final,
            label=f"Densité Prédite",
            color="purple",
        )
        plt.title(f"Reconstruction Spatiale à t=T - {name}")
        plt.xlabel("Index du segment spatial")
        plt.ylabel("Densité (veh/km)")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHICS_DIR, f"{name.lower()}_density_curve.png"))
        plt.close()


def plot_pinn_spatial_evolution(name, model, t_max, x_max, device):
    times = [0.0, t_max / 2, t_max]
    labels = ["t = 0", "t = T/2", "t = T"]
    x_grid = np.linspace(0, x_max, 200)
    X_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1).to(device)

    plt.figure(figsize=(12, 6))
    colors = ["#440154", "#21918c", "#fde725"]  # Viridis colors

    model.eval()
    for t_val, label, color in zip(times, labels, colors):
        T_tensor = (torch.ones_like(X_tensor) * t_val).to(device)
        with torch.no_grad():
            Rho_pred = model(T_tensor, X_tensor).cpu().numpy().flatten() * RHO_MAX
        plt.plot(x_grid, Rho_pred, label=label, color=color, linewidth=2)

    plt.title(f"Évolution Spatiale de la Densité $\\rho(x)$ au cours du temps - {name}")
    plt.xlabel("Position (km)")
    plt.ylabel("Densité (veh/km)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, f"{name.lower()}_spatial_evolution.png"))
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_conf = {
        "epochs": EPOCHS,
        "rho_max": RHO_MAX,
        "use_adaptive": True,
        "seed": SEED,
        "n_colloc": 2000,
        "data_path": DATA_PATH,
        "mu": 0.99,
        "gamma": 0.05,
        "p": 0.05,
    }
    name = "PINN_Hybride_Optimal"
    cache_path = get_cache_path(name, pinn_conf)

    if not os.path.exists(cache_path):
        print(
            "Erreur : Le modèle n'est pas dans le cache. Lancez evaluate_all.py en premier."
        )
        return

    cached_data = torch.load(cache_path, map_location=device, weights_only=False)

    import pandas as pd

    df_temp = pd.read_csv(DATA_PATH)
    t_max = (df_temp["Time_s"] / 3600.0).max()
    x_max = df_temp["Position_km"].max()

    print("Génération des graphiques spécifiques au PINN...")
    plot_pinn_internal_dynamics(name, cached_data["results"])
    plot_pinn_density_reconstruction_and_heatmap(
        name, cached_data["model"], t_max, x_max, cached_data["results"], device
    )
    plot_pinn_spatial_evolution(name, cached_data["model"], t_max, x_max, device)
    plot_pinn_grid_search_ablation(device)
    print(f"Terminé. Résultats dans {GRAPHICS_DIR}")


if __name__ == "__main__":
    main()

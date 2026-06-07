import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json

CACHE_DIR = r"experimentations\charles\final\models_cache"
GRAPHICS_DIR = r"experimentations\charles\final\graphics\resnet"
os.makedirs(GRAPHICS_DIR, exist_ok=True)

DATA_PATH = r"data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv"
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


def get_cache_path(model_name, config_dict):
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{model_name}_{config_hash}.pt")


def plot_resnet_losses(name, res_dict):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs_train = np.arange(len(res_dict["loss_train"]))
    ax1.plot(
        epochs_train,
        res_dict["loss_train"],
        label="Objectif (MSE)",
        color="black",
        linewidth=2,
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss Objectif")

    ax2 = ax1.twinx()
    epochs_val = np.linspace(
        0, len(res_dict["loss_train"]) - 1, len(res_dict["cee_val"])
    )
    ax2.plot(epochs_val, res_dict["cee_val"], label="CEE", color="red", linestyle="--")
    ax2.plot(
        epochs_val,
        res_dict["mae_val"],
        label="MAE (u0alpha_i)",
        color="blue",
        linestyle="-.",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Erreurs Validation (CEE / MAE)")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    plt.title(f"Évolution des pertes (Objectif, CEE, MAE) - {name}")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, f"{name.lower()}_loss_evolution.png"))
    plt.close()


def plot_resnet_density_reconstruction(name, res_dict):
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(res_dict["rho_final_true"]))
    plt.plot(
        x_axis,
        res_dict["rho_final_true"],
        label="Densité Réelle",
        color="black",
        linestyle=":",
        linewidth=2,
    )
    plt.plot(
        x_axis,
        res_dict["rho_final_pred"],
        label=f"Densité Prédite",
        color="blue",
        alpha=0.8,
    )
    plt.title(f"Comparaison de la Densité à $t=T$ - {name}")
    plt.xlabel("Index du segment spatial")
    plt.ylabel("Densité (veh/km)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(GRAPHICS_DIR, f"{name.lower()}_density_reconstruction.png")
    )
    plt.close()


def plot_resnet_alpha_convergence(name, res_dict):
    history = np.array(res_dict["alpha_history"])
    plt.figure(figsize=(10, 6))
    indices_to_plot = np.linspace(0, history.shape[1] - 1, 10, dtype=int)
    for idx in indices_to_plot:
        plt.plot(history[:, idx], label=f"$\\alpha_{{{idx}}}$")
    plt.title(
        f"Convergence des paramètres $\\alpha_i$ (Sélection de 10 segments) - {name}"
    )
    plt.xlabel("Époques évaluées")
    plt.ylabel("Valeur de $\\alpha_i$ (Nb véhicules)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, f"{name.lower()}_alpha_convergence.png"))
    plt.close()


def plot_resnet_alpha_comparison(name, res_dict):
    plt.figure(figsize=(14, 6))
    x = np.arange(len(res_dict["alpha_true"]))
    plt.bar(
        x - 0.2, res_dict["alpha_true"], width=0.4, label="$\\alpha$ Réel", color="gray"
    )
    plt.bar(
        x + 0.2,
        res_dict["alpha_pred"],
        width=0.4,
        label="$\\alpha$ Prédit",
        color="blue",
    )
    plt.title(f"Comparaison des $\\alpha_i$ finaux - {name}")
    plt.xlabel("Index du segment")
    plt.ylabel("Nombre de véhicules ($\\alpha_i$)")
    plt.legend()
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, f"{name.lower()}_alpha_comparison.png"))
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_rc = resnet_config_base.copy()
    conf_rc["METHOD"] = "adaptative"
    cache_path = get_cache_path("ResNet_Adaptatif", conf_rc)

    if not os.path.exists(cache_path):
        print("Erreur : Exécutez evaluate_all.py en premier.")
        return

    res = torch.load(cache_path, map_location=device, weights_only=False)
    name = "ResNet_Adaptatif"

    print("Génération des graphiques spécifiques au ResNet...")
    plot_resnet_losses(name, res)
    plot_resnet_density_reconstruction(name, res)
    plot_resnet_alpha_convergence(name, res)
    plot_resnet_alpha_comparison(name, res)
    print(f"Terminé. Résultats dans {GRAPHICS_DIR}")


if __name__ == "__main__":
    main()

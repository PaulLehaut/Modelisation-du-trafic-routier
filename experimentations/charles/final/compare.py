import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ResNet_training import train_resnet
from pinn import train_pinn_with_weights, generate_all_collocation_epochs

# Configuration
DATA_PATH = r"data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv"
GRAPHICS_DIR = r"experimentations\charles\final\graphics\comparison_final"
os.makedirs(GRAPHICS_DIR, exist_ok=True)
P_VALUE = 0.05
EPOCHS = 1000


def get_real_density(df, t_max, x_max, grid_res=100):
    """Calcule la densité réelle pour la comparaison."""
    # Simulation d'une grille pour la comparaison visuelle
    return np.random.rand(
        grid_res, grid_res
    )  # À remplacer par votre calcul de vérité terrain


def plot_density_heatmap(data, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(
        data, aspect="auto", origin="lower", extent=[0, 10, 0, 1], cmap="viridis"
    )  # Ajustez extent selon vos données
    plt.colorbar(label="Densité $\\rho$")
    plt.xlabel("Position (km)")
    plt.ylabel("Temps (h)")
    plt.title(title)
    plt.savefig(os.path.join(GRAPHICS_DIR, filename))
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA_PATH)
    t_max, x_max = df["Time_s"].max() / 3600.0, df["Position_km"].max()

    # 1. Entraînement / Récupération modèles (p=0.05)
    print("Entraînement des modèles pour p=0.05...")

    # ResNet
    conf_resnet = {
        "DATA_PATH": DATA_PATH,
        "PORTION_PROBE": P_VALUE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": 0.5,
        "N_TOTAL": 1000,
        "L_V": 0.005,
        "V_MAX": 50.0,
        "RHO_MAX": 200.0,
        "SEED": 42,
        "METHOD": "adaptative",
    }
    res_resnet = train_resnet(conf_resnet)

    # PINN
    t_colloc, x_colloc = generate_all_collocation_epochs(
        2000, EPOCHS, t_max, x_max, device
    )
    model_pinn, _, _, _, _, cee_hist, _ = train_pinn_with_weights(
        DATA_PATH,
        t_colloc,
        x_colloc,
        epochs=EPOCHS,
        mu=0.99,
        selection_method="adaptative",
    )

    # 2. Visualisation Densité (Heatmaps)
    # Note : Vous devrez utiliser les sorties de modèles (rho_final_pred ou inférence directe)
    print("Génération des graphiques...")
    # Exemple pour ResNet (adapté selon la forme de vos données de sortie)
    plot_density_heatmap(
        np.tile(res_resnet["rho_final_pred"], (100, 1)),
        f"Densité ResNet (p={P_VALUE})",
        "resnet_heatmap.png",
    )

    # 3. Comparaison métriques vs P (simulé ici avec vos données existantes)
    # Vous pouvez boucler sur les P comme dans evaluate_all.py pour obtenir les valeurs
    p_values = [0.01, 0.05, 0.10, 0.20]
    cees = [0.05, 0.03, 0.02, 0.01]  # Exemple de valeurs

    plt.figure(figsize=(10, 6))
    plt.plot(p_values, cees, marker="o", label="CEE")
    plt.xlabel("Taux de véhicules témoins (p)")
    plt.ylabel("Erreur (CEE)")
    plt.title("Performance vs Densité d'observation")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHICS_DIR, "performance_vs_p.png"))
    plt.close()


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hashlib
import json

from ResNet_training import train_resnet
from pinn import train_pinn_with_weights, generate_all_collocation_epochs

# ==========================================
# CONFIGURATION GLOBALE
# ==========================================
DATA_PATH = r"data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv"
CACHE_DIR = r"experimentations\charles\final\models_cache"
GRAPHICS_DIR = r"experimentations\charles\final\graphics\comparison"

os.makedirs(GRAPHICS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

EPOCHS = 2000
SEED = 42
PORTION_PROBES_TO_TEST = [0.01, 0.05, 0.10, 0.15, 0.20, 0.50]

resnet_config_base = {
    "DATA_PATH": DATA_PATH,
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


# ==========================================
# GRAPHIQUE DE COMPARAISON GLOBAL
# ==========================================
def plot_cee_vs_probe_percentage(results_across_p):
    plt.figure(figsize=(10, 6))
    methods = ["ResNet_Standard", "ResNet_Adaptatif", "PINN_Hybride_Optimal"]
    colors = sns.color_palette("husl", len(methods))

    for method, color in zip(methods, colors):
        cees = [
            results_across_p[p][method]["cee_final"] for p in PORTION_PROBES_TO_TEST
        ]
        plt.plot(
            PORTION_PROBES_TO_TEST,
            cees,
            marker="o",
            label=method.replace("_", " "),
            color=color,
            linewidth=2,
        )

    plt.yscale("log")
    plt.title("Évolution de la CEE selon le taux de véhicules témoins (p)")
    plt.xlabel("Taux de véhicules témoins (p)")
    plt.ylabel("CEE Finale (Log Scale)")
    plt.xticks(
        PORTION_PROBES_TO_TEST, [f"{int(p * 100)}%" for p in PORTION_PROBES_TO_TEST]
    )
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "comparaison_cee_vs_p.png"))
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_across_p = {}

    df_temp = pd.read_csv(DATA_PATH)
    t_max = (df_temp["Time_s"] / 3600.0).max()
    x_max = df_temp["Position_km"].max()
    t_colloc, x_colloc = generate_all_collocation_epochs(
        2000, EPOCHS, t_max, x_max, device
    )

    print("=" * 50)
    print("Entraînement des modèles pour comparaison et cache...")

    # 1. Boucle sur les P pour le graphe de comparaison
    for p in PORTION_PROBES_TO_TEST:
        print(f" -> Évaluation pour p = {p * 100}%")
        results_across_p[p] = {}

        for method, m_name in [
            ("random", "ResNet_Standard"),
            ("adaptative", "ResNet_Adaptatif"),
        ]:
            conf = resnet_config_base.copy()
            conf["PORTION_PROBE"], conf["METHOD"] = p, method
            cache = get_cache_path(m_name, conf)
            if os.path.exists(cache):
                results_across_p[p][m_name] = torch.load(
                    cache, map_location=device, weights_only=False
                )
            else:
                res = train_resnet(conf)
                torch.save(res, cache)
                results_across_p[p][m_name] = res

        # PINN Optimal
        pinn_conf = {
            "epochs": EPOCHS,
            "rho_max": 200.0,
            "use_adaptive": True,
            "seed": SEED,
            "n_colloc": 2000,
            "data_path": DATA_PATH,
            "mu": 0.99,
            "gamma": 0.05,
            "p": p,
        }
        cache_pinn = get_cache_path("PINN_Hybride_Optimal", pinn_conf)
        if os.path.exists(cache_pinn):
            results_across_p[p]["PINN_Hybride_Optimal"] = torch.load(
                cache_pinn, map_location=device, weights_only=False
            )["results"]
        else:
            model, _, l_tot, _, _, cee_hist, _ = train_pinn_with_weights(
                DATA_PATH,
                t_colloc,
                x_colloc,
                epochs=EPOCHS,
                mu=0.99,
                gamma=0.05,
                rho_max=200.0,
                use_adaptive=True,
                seed=SEED,
                selection_method="adaptative",
            )
            res = {"cee_final": cee_hist[-1], "loss_train": l_tot, "cee_val": cee_hist}
            torch.save({"model": model, "results": res}, cache_pinn)
            results_across_p[p]["PINN_Hybride_Optimal"] = res

    # 2. Remplissage du cache Ablation PINN pour p=0.20 (Requis pour evaluate_pinn.py)
    print("\n -> Vérification du cache Ablation PINN pour p=20%...")
    for p_conf in [
        {"name": "PINN_Donnees_Pures", "mu": 1.0},
        {"name": "PINN_Physique_Pure", "mu": 0.0},
    ]:
        c_dict = {
            "epochs": EPOCHS,
            "rho_max": 200.0,
            "use_adaptive": True,
            "seed": SEED,
            "n_colloc": 2000,
            "data_path": DATA_PATH,
            "mu": p_conf["mu"],
            "gamma": 0.05,
            "p": 0.20,
        }
        c_path = get_cache_path(p_conf["name"], c_dict)
        if not os.path.exists(c_path):
            model, _, l_tot, _, _, cee_hist, _ = train_pinn_with_weights(
                DATA_PATH,
                t_colloc,
                x_colloc,
                epochs=EPOCHS,
                mu=p_conf["mu"],
                gamma=0.05,
                rho_max=200.0,
                use_adaptive=True,
                seed=SEED,
                selection_method="adaptative",
            )
            torch.save(
                {"model": model, "results": {"cee_val": cee_hist, "loss_train": l_tot}},
                c_path,
            )

    plot_cee_vs_probe_percentage(results_across_p)
    print(f"Terminé. Comparaison sauvegardée dans {GRAPHICS_DIR}")


if __name__ == "__main__":
    main()


# le problème de cette approche c'est qu'on ne prend pas en compte le temps jusqu'au régime permanent.

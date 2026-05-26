# Fichier : experimentations/charles/final/debug_pinn_hyperparams.py

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Import des fonctions existantes depuis ton module PINN
from pinn import (
    TrafficPINN,
    compute_physics_loss,
    load_and_split_data,
    generate_all_collocation_epochs,
    calculate_pv_weights,
)

# ==========================================
# CONFIGURATION DU DEBOGAGE
# ==========================================
DATA_PATH = r"data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv"
GRAPHICS_DIR = r"experimentations\charles\final\graphics"
os.makedirs(GRAPHICS_DIR, exist_ok=True)

# [MODIFICATION] Nouvel espace de recherche (Ablation Study & Affinage)
SEARCH_SPACE = {
    "lr": [1e-3],  # Figé sur le meilleur candidat précédent pour gagner du temps
    "mu": [
        0.0,  # 100% Physique (Zéro donnée) : Devrait s'effondrer sur une solution triviale (0 partout)
        0.9,  # Meilleur candidat du test précédent
        0.95,  # Affinage : 95% donnée, 5% physique
        0.99,  # Affinage : 99% donnée, 1% physique (La physique agit comme micro-régularisateur)
        1.0,  # 100% Données (Zéro physique) : MLP classique, risque de sur-apprentissage
    ],
    "gamma": [0.05, 0.01],
}

DEBUG_EPOCHS = 1000
NUM_COLLOC = 2000
V_MAX = 50.0
RHO_MAX = 200.0
SEED = 42


def train_pinn_debug(
    t_data,
    x_data,
    rho_data,
    w_data,
    t_test,
    x_test,
    rho_test,
    t_colloc_all,
    x_colloc_all,
    t_max,
    x_max,
    config,
):
    """Boucle d'entraînement isolée pour tester une configuration spécifique"""
    device = t_data.device
    model = TrafficPINN(t_max, x_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    cee_history = []
    loss_total_history = []
    loss_data_history = []
    loss_phys_history = []
    epochs_logged = []

    for epoch in range(DEBUG_EPOCHS):
        model.train()
        optimizer.zero_grad()

        rho_pred = model(t_data, x_data)
        loss_data = torch.mean(w_data * (rho_pred - rho_data) ** 2)

        t_colloc = t_colloc_all[epoch]
        x_colloc = x_colloc_all[epoch]
        loss_phys = compute_physics_loss(
            model, t_colloc, x_colloc, V_MAX, config["gamma"]
        )

        loss_total = config["mu"] * loss_data + (1.0 - config["mu"]) * loss_phys

        loss_total.backward()
        optimizer.step()

        # Enregistrement synchronisé des pertes d'entraînement et de la CEE
        if epoch % 10 == 0 or epoch == DEBUG_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                rho_test_pred = model(t_test, x_test)
                cee = torch.mean(
                    ((rho_test_pred * RHO_MAX) - (rho_test * RHO_MAX)) ** 2
                ).item()

                cee_history.append(cee)
                loss_total_history.append(loss_total.item())
                loss_data_history.append(loss_data.item())
                loss_phys_history.append(loss_phys.item())
                epochs_logged.append(epoch)

    return {
        "epochs": epochs_logged,
        "cee": cee_history,
        "loss_total": loss_total_history,
        "loss_data": loss_data_history,
        "loss_phys": loss_phys_history,
    }


def main():
    print("Chargement des données pour le débogage...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, test_df, t_max, x_max, pv_ids = load_and_split_data(
        DATA_PATH, pv_ratio=0.10, v_max=V_MAX, seed=SEED, selection_method="colleau"
    )

    train_df = calculate_pv_weights(train_df)
    w_data = (
        torch.tensor(train_df["Weight"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    t_data = (
        torch.tensor(train_df["Time_h"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    x_data = (
        torch.tensor(train_df["Position_km"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    rho_data = (
        torch.tensor(train_df["Density"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    t_test = (
        torch.tensor(test_df["Time_h"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    x_test = (
        torch.tensor(test_df["Position_km"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    rho_test = (
        torch.tensor(test_df["Density"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )

    t_colloc, x_colloc = generate_all_collocation_epochs(
        NUM_COLLOC, DEBUG_EPOCHS, t_max, x_max, device
    )

    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # [AJOUT] Filtre pour éviter de calculer 2 fois mu=0.0 ou mu=1.0 si gamma n'a pas d'impact
    # Pour mu=1.0, gamma n'est pas utilisé dans la loss, on ne garde qu'une occurrence.
    filtered_combinations = []
    seen_mu_1 = False
    for c in combinations:
        if c["mu"] == 1.0:
            if not seen_mu_1:
                filtered_combinations.append(c)
                seen_mu_1 = True
        else:
            filtered_combinations.append(c)

    results = {}

    print(
        f"\nLancement du Grid Search (Ablation Study) sur {len(filtered_combinations)} configurations ({DEBUG_EPOCHS} epochs chacune)..."
    )
    print("-" * 60)

    for i, config in enumerate(filtered_combinations):
        config_name = f"lr={config['lr']}, mu={config['mu']}, g={config['gamma']}"
        print(f"Test {i + 1}/{len(filtered_combinations)} : {config_name}")

        start_time = time.time()
        res_dict = train_pinn_debug(
            t_data,
            x_data,
            rho_data,
            w_data,
            t_test,
            x_test,
            rho_test,
            t_colloc,
            x_colloc,
            t_max,
            x_max,
            config,
        )
        elapsed = time.time() - start_time

        final_cee = res_dict["cee"][-1]
        results[config_name] = res_dict
        print(f"  -> CEE Finale : {final_cee:.2f} (Temps : {elapsed:.1f}s)")

    # ==========================================
    # VISUALISATION DES RÉSULTATS
    # ==========================================
    print("\nGénération des graphiques de comparaison...")

    # Tri pour mettre en évidence les meilleures configurations
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]["cee"][-1]))

    # --- GRAPHIQUE 1 : Comparaison globale des CEE ---
    plt.figure(figsize=(14, 8))
    for i, (name, res) in enumerate(sorted_results.items()):
        # On met en évidence les cas extrêmes (mu=0 et mu=1) avec des styles spécifiques
        if "mu=0.0" in name:
            plt.plot(
                res["epochs"],
                res["cee"],
                label=f"{name} (Physique Pure)",
                color="black",
                linestyle=":",
                linewidth=2,
            )
        elif "mu=1.0" in name:
            plt.plot(
                res["epochs"],
                res["cee"],
                label=f"{name} (Données Pures)",
                color="red",
                linestyle="--",
                linewidth=2,
            )
        else:
            alpha = 1.0 if i < 3 else 0.3
            linewidth = 2.5 if i < 3 else 1.0
            plt.plot(
                res["epochs"],
                res["cee"],
                label=f"{name} (CEE: {res['cee'][-1]:.0f})",
                alpha=alpha,
                linewidth=linewidth,
            )

    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Validation CEE (Log Scale)")
    plt.title("Ablation Study : Hybridation vs Modèles Purs (Données ou Physique)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "pinn_ablation_study_cee.png"))
    plt.close()

    # --- GRAPHIQUE 2 : Suivi interne (Losses) des 3 meilleures configurations ---
    top_3_names = list(sorted_results.keys())[:3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, name in enumerate(top_3_names):
        res = sorted_results[name]
        ax1 = axes[i]

        ax1.plot(
            res["epochs"],
            res["loss_total"],
            label="Total Loss",
            color="black",
            linewidth=2,
        )
        ax1.plot(
            res["epochs"],
            res["loss_data"],
            label="Data Loss",
            linestyle="--",
            color="blue",
            alpha=0.8,
        )
        ax1.plot(
            res["epochs"],
            res["loss_phys"],
            label="Physics Loss",
            linestyle=":",
            color="green",
            alpha=0.8,
        )
        ax1.set_yscale("log")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Training Losses (Log)")
        ax1.set_title(f"Rang {i + 1}: {name}")
        ax1.grid(True, alpha=0.4)

        ax2 = ax1.twinx()
        ax2.plot(
            res["epochs"],
            res["cee"],
            label="Validation CEE",
            color="red",
            linewidth=2,
            alpha=0.7,
        )
        ax2.set_yscale("log")
        ax2.set_ylabel("CEE", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8
        )

    plt.suptitle("Dynamique interne d'apprentissage des 3 meilleurs PINNs", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "pinn_ablation_top3_dynamics.png"))
    plt.close()

    print(f"Graphiques sauvegardés dans le dossier : {GRAPHICS_DIR}")


if __name__ == "__main__":
    main()

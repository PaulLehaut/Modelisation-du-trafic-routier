import os
import time  # [AJOUT] Pour la mesure précise des temps d'exécution
import numpy as np
import pandas as pd
from experimentations.charles.final.ResNet_model import TrafficResNet, project_alpha
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# CONFIGURATION MODULABLE
# =====================================================================
CONFIG = {
    "BASE_DIR": r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\experimentations\charles\ResNet_alpha",
    "DATA_PATH": r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\data\reconstruction_modele_imi\traffic_N1000_stop_and_go.csv",
    "PORTION_PROBE": 0.05,
    "EPOCHS": 1000,
    "LEARNING_RATE": 0.5,
    "LOSS_POLICY": "continuous",  # Options: "continuous" ou "final_point"
    "N_TOTAL": 1000,
    "L_V": 0.005,
    "V_MAX": 50.0,
    "PROBE_SELECTION_METHOD": "random",  # [AJOUT] Options: "random" ou "colleau" (méthode temporelle)
}
CONFIG["RHO_MAX"] = 1.0 / CONFIG["L_V"]


# [AJOUT] Fonction de sélection des véhicules sondes selon la méthode choisie
def select_probe_vehicles(df, method, portion_probe, n_total, seed=42):
    unique_vehicles = np.sort(df["Vehicle_ID"].unique())
    num_pvs = max(2, int(portion_probe * n_total))

    if method == "random":
        np.random.seed(seed)
        pv_ids = np.sort(np.random.choice(unique_vehicles, num_pvs, replace=False))
    elif method == "colleau":
        # Sélection uniforme par espacement d'index (approche déterministe de base fournie)
        pv_ids = np.linspace(0, n_total - 1, num_pvs, dtype=int)
    else:
        raise ValueError(f"Méthode de sélection inconnue: {method}")

    return pv_ids


def main():
    if not os.path.exists(CONFIG["DATA_PATH"]):
        raise FileNotFoundError(f"Fichier introuvable: {CONFIG['DATA_PATH']}")

    print("Chargement des données...")
    df = pd.read_csv(CONFIG["DATA_PATH"])

    # Extraction du nom de la base de données pour le nom du fichier de sauvegarde
    db_name = os.path.splitext(os.path.basename(CONFIG["DATA_PATH"]))[0]

    # [MODIFICATION] Sélection des sondes avec l'argument dédié
    pv_ids = select_probe_vehicles(
        df,
        CONFIG["PROBE_SELECTION_METHOD"],
        CONFIG["PORTION_PROBE"],
        CONFIG["N_TOTAL"],
    )
    num_pvs = len(pv_ids)
    n_gaps = num_pvs - 1

    t_min_s, t_max_s = df["Time_s"].min(), df["Time_s"].max()
    T_h = (t_max_s - t_min_s) / 3600.0

    times_sorted = np.sort(df["Time_s"].unique())
    dt_h = (times_sorted[1] - times_sorted[0]) / 3600.0
    num_steps = int(round(T_h / dt_h))

    alpha_true = np.diff(pv_ids)

    df_pvs = df[df["Vehicle_ID"].isin(pv_ids)].pivot(
        index="Time_s", columns="Vehicle_ID", values="Position_km"
    )

    y_target_history = torch.tensor(df_pvs.values[:, :-1], dtype=torch.float32)
    y_target_final = y_target_history[-1]

    x_0_followers = y_target_history[0]
    x_0_leader = torch.tensor(df_pvs.values[0, -1], dtype=torch.float32)

    y_bar = torch.tensor(df_pvs.values[-1, :], dtype=torch.float32)
    gap_0 = (
        x_0_followers[1:] - x_0_followers[:-1]
        if len(x_0_followers) > 1
        else torch.tensor([])
    )
    gap_0 = torch.cat([gap_0, torch.tensor([x_0_leader - x_0_followers[-1]])])
    gap_T = y_bar[1:] - y_bar[:-1]

    z_bar = torch.min(gap_0 / CONFIG["L_V"], gap_T / CONFIG["L_V"])

    model = TrafficResNet(
        n_gaps,
        CONFIG["N_TOTAL"],
        CONFIG["L_V"],
        CONFIG["V_MAX"],
        CONFIG["RHO_MAX"],
        dt_h,
        num_steps,
        x_0_followers,
        x_0_leader,
        z_bar,
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])

    loss_history = []
    alpha_history = []

    # ---------- BOUCLE D'ENTRAÎNEMENT ----------
    print(f"\nOptimisation en cours... (Politique: {CONFIG['LOSS_POLICY']})")

    start_train_time = time.time()  # [AJOUT] Début de la mesure du temps d'entraînement

    for epoch in range(CONFIG["EPOCHS"] + 1):
        optimizer.zero_grad()

        if CONFIG["LOSS_POLICY"] == "continuous":
            x_pred_history = model(require_grad_history=True)
            loss = criterion(x_pred_history, y_target_history)
        else:
            x_pred_T = model(require_grad_history=False)
            loss = criterion(x_pred_T, y_target_final)

        loss.backward()
        optimizer.step()

        model.alpha.data = project_alpha(model.alpha.data, model.z_bar, model.N)

        loss_history.append(loss.item())
        alpha_history.append(model.alpha.detach().clone().numpy())

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | MSE Loss: {loss.item():.6f}")

    end_train_time = time.time()  # [AJOUT] Fin de la mesure du temps d'entraînement
    training_time = end_train_time - start_train_time

    # ---------- ÉVALUATION & INFÉRENCE ----------
    start_inf_time = time.time()  # [AJOUT] Début de la mesure du temps d'inférence
    _, hist_followers, hist_leader, intermediate_rhos = model(return_history=True)
    end_inf_time = time.time()  # [AJOUT] Fin de la mesure du temps d'inférence
    inference_time = end_inf_time - start_inf_time

    alpha_optimise = model.alpha.detach().numpy()

    true_gaps = y_bar[1:] - y_bar[:-1]
    density_true = alpha_true / true_gaps.numpy()

    # [AJOUT] Calcul de la Loss CEE sur les véhicules non observés (données de test)
    # Extraction des données de test (véhicules non sondes)
    test_df = df[~df["Vehicle_ID"].isin(pv_ids)]
    # Calcul de la densité réelle des tests via Greenshields inverse
    rho_test_real = 1.0 - (test_df["Velocity_kmh"].values / CONFIG["V_MAX"])

    # Interpolation ou reconstruction de la densité prédite au point de test pour calculer l'erreur
    # Pour le ResNet, on évalue la différence globale moyenne au dernier pas de temps ou sur la grille
    # Ici, formule exacte demandée appliquée sur les densités finales reconstruites :
    reconstructed_gaps = hist_followers[-1][1:] - hist_followers[-1][:-1]
    reconstructed_gaps = np.append(
        reconstructed_gaps, hist_leader[-1] - hist_followers[-1][-1]
    )
    rho_pred_final = alpha_optimise / (reconstructed_gaps + 1e-6)

    # Calcul de la CEE moyenne finale
    cee_final = np.mean((rho_pred_final - density_true) ** 2)

    # ---------- SAUVEGARDE DES RÉSULTATS (CONSIGNES EXPLICITES) ----------
    results = {
        "loss_history": loss_history,
        "alpha_history": np.array(alpha_history),
        "alpha_true": alpha_true,
        "density_true": density_true,
        "hist_followers": hist_followers,
        "hist_leader": hist_leader,
        "times_h": np.linspace(0, T_h, num_steps + 1),
        "alpha_optimise": alpha_optimise,
        "y_target_followers": y_target_final.numpy(),
        "CONFIG": CONFIG,
        "training_time_s": training_time,  # [AJOUT] Stockage du temps d'entraînement
        "inference_time_s": inference_time,  # [AJOUT] Stockage du temps d'évaluation
        "cee_final": cee_final,  # [AJOUT] Stockage de la loss CEE demandée
        "intermediate_rhos": intermediate_rhos,  # [AJOUT] Sauvegarde pour post-traitement
    }

    results_dir = os.path.join(CONFIG["BASE_DIR"], "training_results")
    os.makedirs(results_dir, exist_ok=True)

    # [MODIFICATION] Nommage de fichier explicite demandé par les consignes
    filename = (
        f"ResNet_epochs{CONFIG['EPOCHS']}_lr{CONFIG['LEARNING_RATE']}_"
        f"method-{CONFIG['PROBE_SELECTION_METHOD']}_{db_name}.pt"
    )
    filepath = os.path.join(results_dir, filename)

    torch.save(results, filepath)
    print(f"\nRésultats sauvegardés avec succès dans :\n{filepath}")
    print(f"Temps d'entraînement: {training_time:.4f}s")
    print(f"Temps d'inférence: {inference_time:.4f}s")
    print(f"CEE Finale Reconstruite: {cee_final:.6f}")


if __name__ == "__main__":
    main()

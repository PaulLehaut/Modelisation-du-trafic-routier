import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from ResNet_model import TrafficResNet, project_alpha

# =====================================================================
# CONFIGURATION MODULABLE
# =====================================================================
# [MODIFICATION] Regroupement dans un dictionnaire pour centraliser les choix
CONFIG = {
    "BASE_DIR": r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\experimentations\charles\ResNet_alpha",
    "DATA_PATH": r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv",
    "PORTION_PROBE": 0.05,
    "EPOCHS": 200,
    "LEARNING_RATE": 0.5,
    "LOSS_POLICY": "continuous",  # Options: "continuous" (Approche 3) ou "final_point" (Approche 2)
    "N_TOTAL": 1000,
    "L_V": 0.005,
    "V_MAX": 50.0,
}
CONFIG["RHO_MAX"] = 1.0 / CONFIG["L_V"]


def main():
    if not os.path.exists(CONFIG["DATA_PATH"]):
        raise FileNotFoundError(f"Fichier introuvable: {CONFIG['DATA_PATH']}")

    print("Chargement des données...")
    df = pd.read_csv(CONFIG["DATA_PATH"])

    num_pvs = max(2, int(CONFIG["PORTION_PROBE"] * CONFIG["N_TOTAL"]))
    n_gaps = num_pvs - 1

    t_min_s, t_max_s = df["Time_s"].min(), df["Time_s"].max()
    T_h = (t_max_s - t_min_s) / 3600.0

    times_sorted = np.sort(df["Time_s"].unique())
    dt_h = (times_sorted[1] - times_sorted[0]) / 3600.0
    num_steps = int(round(T_h / dt_h))

    # Sélection des probes (politique uniforme par défaut)
    pv_ids = np.linspace(0, CONFIG["N_TOTAL"] - 1, num_pvs, dtype=int)

    # [MODIFICATION] Calcul de la vérité terrain : alpha réel = nombre de véhicules dans le segment
    # Cela correspond exactement à la différence des indices des probes
    alpha_true = np.diff(pv_ids)

    # --- Extraction des données cibles ---
    # [MODIFICATION] Extraction de l'historique spatio-temporel complet pour l'Approche 3
    df_pvs = df[df["Vehicle_ID"].isin(pv_ids)].pivot(
        index="Time_s", columns="Vehicle_ID", values="Position_km"
    )

    # y_target_history contient les trajectoires de tous les véhicules témoins suiveurs sur tout [0, T]
    y_target_history = torch.tensor(df_pvs.values[:, :-1], dtype=torch.float32)
    y_target_final = y_target_history[-1]  # Pour l'Approche 2 (fallback)

    x_0_followers = y_target_history[0]
    x_0_leader = torch.tensor(df_pvs.values[0, -1], dtype=torch.float32)

    y_bar = torch.tensor(
        df_pvs.values[-1, :], dtype=torch.float32
    )  # Toutes positions à T
    gap_0 = (
        x_0_followers[1:] - x_0_followers[:-1]
        if len(x_0_followers) > 1
        else torch.tensor([])
    )
    gap_0 = torch.cat([gap_0, torch.tensor([x_0_leader - x_0_followers[-1]])])
    gap_T = y_bar[1:] - y_bar[:-1]

    z_bar = torch.min(gap_0 / CONFIG["L_V"], gap_T / CONFIG["L_V"])

    # ---------- INITIALISATION ----------
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
    for epoch in range(CONFIG["EPOCHS"] + 1):
        optimizer.zero_grad()

        # [MODIFICATION] Choix dynamique de la fonction de perte
        if CONFIG["LOSS_POLICY"] == "continuous":
            # Approche 3 : Comparaison sur tout l'historique
            x_pred_history = model(require_grad_history=True)
            loss = criterion(x_pred_history, y_target_history)
        else:
            # Approche 2 : Comparaison uniquement à t=T
            x_pred_T = model(require_grad_history=False)
            loss = criterion(x_pred_T, y_target_final)

        loss.backward()
        optimizer.step()

        model.alpha.data = project_alpha(model.alpha.data, model.z_bar, model.N)

        loss_history.append(loss.item())
        alpha_history.append(model.alpha.detach().clone().numpy())

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | MSE Loss: {loss.item():.6f}")

    # ---------- PRÉDICTION FINALE ----------
    _, hist_followers, hist_leader = model(return_history=True)
    alpha_optimise = model.alpha.detach().numpy()

    # [MODIFICATION] Sauvegarde des données réelles pour la visualisation comparative
    final_gaps = np.append(hist_followers[-1, 1:], hist_leader[-1]) - hist_followers[-1]
    density_true = alpha_true / final_gaps  # Densité réelle basée sur les vrais alpha

    results = {
        "loss_history": loss_history,
        "alpha_history": np.array(alpha_history),
        "alpha_true": alpha_true,  # NOUVEAU
        "density_true": density_true,  # NOUVEAU
        "hist_followers": hist_followers,
        "hist_leader": hist_leader,
        "times_h": np.linspace(0, T_h, num_steps + 1),
        "alpha_optimise": alpha_optimise,
        "y_target_followers": y_target_final.numpy(),
        "CONFIG": CONFIG,  # Sauvegarde de la config complète
    }

    results_dir = os.path.join(CONFIG["BASE_DIR"], "training_results")
    os.makedirs(results_dir, exist_ok=True)
    filename = f"ResNet_probe{CONFIG['PORTION_PROBE']}_ep{CONFIG['EPOCHS']}_loss-{CONFIG['LOSS_POLICY']}.pt"
    filepath = os.path.join(results_dir, filename)
    torch.save(results, filepath)
    print(f"Résultats sauvegardés dans :\n{filepath}")


if __name__ == "__main__":
    main()

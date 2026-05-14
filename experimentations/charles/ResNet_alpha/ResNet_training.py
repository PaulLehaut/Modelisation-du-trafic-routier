import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import de la classe et des fonctions depuis le fichier modèle
from ResNet_model import TrafficResNet, project_alpha

# =====================================================================
# CONFIGURATION ET HYPERPARAMÈTRES
# =====================================================================
BASE_DIR = r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\experimentations\charles\ResNet_alpha"
# Chemin vers le fichier de données reconstruites par Bastien à partir de notre modèle (à adapter selon votre organisation)
DATA_PATH = r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier\data\reconstruction_modele_imi\traffic_N1000_rarefaction.csv"

# Hyperparamètres importants (serviront pour le nommage)
PORTION_PROBE = 0.10  # 10% de pénétration
EPOCHS = 200
LEARNING_RATE = 0.5

# Paramètres physiques
N_TOTAL = 1000
L_V = 0.005
V_MAX = 50.0
RHO_MAX = 1.0 / L_V


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Fichier introuvable: {DATA_PATH}")

    print("Chargement des données...")
    df = pd.read_csv(DATA_PATH)

    num_pvs = max(2, int(PORTION_PROBE * N_TOTAL))
    n_gaps = num_pvs - 1

    t_min_s = df["Time_s"].min()
    t_max_s = df["Time_s"].max()
    T_h = (t_max_s - t_min_s) / 3600.0

    times_sorted = np.sort(df["Time_s"].unique())
    dt_s = times_sorted[1] - times_sorted[0]
    dt_h = dt_s / 3600.0
    num_steps = int(round(T_h / dt_h))

    print(f"Simulation sur T = {T_h:.4f}h avec dt = {dt_h:.6f}h")
    print(f"Proportion de PV: {PORTION_PROBE * 100}% -> {num_pvs} véhicules observés.")

    pv_ids = np.linspace(0, N_TOTAL - 1, num_pvs, dtype=int)

    df_t0 = df[df["Time_s"] == t_min_s].set_index("Vehicle_ID")
    df_tT = df[df["Time_s"] == t_max_s].set_index("Vehicle_ID")

    x_bar = torch.tensor(df_t0.loc[pv_ids, "Position_km"].values, dtype=torch.float32)
    y_bar = torch.tensor(df_tT.loc[pv_ids, "Position_km"].values, dtype=torch.float32)

    x_0_followers = x_bar[:-1]
    x_0_leader = x_bar[-1]
    y_target_followers = y_bar[:-1]

    gap_0 = x_bar[1:] - x_bar[:-1]
    gap_T = y_bar[1:] - y_bar[:-1]
    z_bar = torch.min(gap_0 / L_V, gap_T / L_V)

    # ---------- INITIALISATION ----------
    model = TrafficResNet(
        n_gaps,
        N_TOTAL,
        L_V,
        V_MAX,
        RHO_MAX,
        dt_h,
        num_steps,
        x_0_followers,
        x_0_leader,
        z_bar,
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    alpha_history = []

    # ---------- BOUCLE D'ENTRAÎNEMENT ----------
    print("\nDébut de l'optimisation...")
    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()

        x_pred_T = model(return_history=False)
        loss = criterion(x_pred_T, y_target_followers)

        loss.backward()
        optimizer.step()

        model.alpha.data = project_alpha(model.alpha.data, model.z_bar, model.N)

        loss_history.append(loss.item())
        alpha_history.append(model.alpha.detach().clone().numpy())

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | MSE Loss: {loss.item():.6f}")

    # ---------- PRÉDICTION FINALE ----------
    _, hist_followers, hist_leader = model(return_history=True)
    times_h = np.linspace(0, T_h, num_steps + 1)
    alpha_optimise = model.alpha.detach().numpy()

    # ---------- SAUVEGARDE DES RÉSULTATS ----------
    print("\nSauvegarde des données d'entraînement pour la visualisation...")
    results = {
        "loss_history": loss_history,
        "alpha_history": np.array(alpha_history),
        "hist_followers": hist_followers,
        "hist_leader": hist_leader,
        "times_h": times_h,
        "alpha_optimise": alpha_optimise,
        "y_target_followers": y_target_followers.numpy(),
        "n_gaps": n_gaps,
        "rho_max": RHO_MAX,
        "T_h": T_h,
        "PORTION_PROBE": PORTION_PROBE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
    }

    # Création d'un dossier pour stocker les résultats bruts
    results_dir = os.path.join(BASE_DIR, "training_results")
    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # --- DÉBUT DES MODIFICATIONS : NOM DE FICHIER DYNAMIQUE ---
    # ---------------------------------------------------------------------

    # Formatage du nom de fichier avec les hyperparamètres
    filename = f"ResNet_probe{PORTION_PROBE}_ep{EPOCHS}_lr{LEARNING_RATE}.pt"
    filepath = os.path.join(results_dir, filename)

    # Sauvegarde dans le nouveau chemin
    torch.save(results, filepath)
    print(f"Entraînement terminé et résultats sauvegardés dans :\n{filepath}")

    # ---------------------------------------------------------------------
    # --- FIN DES MODIFICATIONS ---
    # ---------------------------------------------------------------------


if __name__ == "__main__":
    main()

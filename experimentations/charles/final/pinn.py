import os
import time  # [AJOUT] Pour le stockage du temps d'entraînement et d'inférence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import qmc
import torch
import torch.nn as nn


# ==========================================
# 1. PINN Architecture
# ==========================================
class TrafficPINN(nn.Module):
    def __init__(self, t_max, x_max, layers=[2, 32, 32, 32, 32, 1]):
        super().__init__()
        self.t_max = t_max
        self.x_max = x_max

        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.hidden_layers.append(nn.Tanh())

        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, t, x):
        # Internal Normalization to [-1, 1]
        t_norm = 2.0 * (t / self.t_max) - 1.0
        x_norm = 2.0 * (x / self.x_max) - 1.0

        inputs = torch.cat([t_norm, x_norm], dim=1)
        for layer in self.hidden_layers:
            inputs = layer(inputs)

        return self.sigmoid(self.output_layer(inputs))


# ==========================================
# 2. Physics Loss Computation
# ==========================================
def compute_physics_loss(model, t_colloc, x_colloc, v_max, gamma=0.05):
    t_colloc.requires_grad_(True)
    x_colloc.requires_grad_(True)

    rho = model(t_colloc, x_colloc)

    rho_t = torch.autograd.grad(
        rho, t_colloc, grad_outputs=torch.ones_like(rho), create_graph=True
    )[0]
    rho_x = torch.autograd.grad(
        rho, x_colloc, grad_outputs=torch.ones_like(rho), create_graph=True
    )[0]
    rho_xx = torch.autograd.grad(
        rho_x, x_colloc, grad_outputs=torch.ones_like(rho_x), create_graph=True
    )[0]

    # Viscous Greenshields LWR Residual
    flux_derivative = v_max * (1.0 - 2.0 * rho)
    residual = rho_t + flux_derivative * rho_x - gamma * rho_xx

    return torch.mean(residual**2)


# ==========================================
# 3. Data Loading & LHS Generation
# ==========================================
# [MODIFICATION] Ajout de l'argument de méthode de sélection globale pour correspondre au rapport
def load_and_split_data(
    csv_path, pv_ratio=0.10, v_max=50.0, seed=42, selection_method="random"
):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    df["Time_h"] = df["Time_s"] / 3600.0
    df["Density"] = 1.0 - (df["Velocity_kmh"] / v_max)

    t_max = df["Time_h"].max()
    x_max = df["Position_km"].max()

    unique_vehicles = df["Vehicle_ID"].unique()
    num_pvs = int(len(unique_vehicles) * pv_ratio)

    # [MODIFICATION] Prise en compte explicite de la méthode de choix demandée
    if selection_method == "random":
        np.random.seed(seed)
        pv_ids = np.random.choice(unique_vehicles, num_pvs, replace=False)
    elif selection_method == "colleau":
        # Utilisation d'une méthode de sélection déterministe par pas régulier
        pv_ids = unique_vehicles[
            np.linspace(0, len(unique_vehicles) - 1, num_pvs, dtype=int)
        ]
    else:
        raise ValueError(f"Méthode de sélection inconnue : {selection_method}")

    train_df = df[df["Vehicle_ID"].isin(pv_ids)]
    test_df = df[~df["Vehicle_ID"].isin(pv_ids)]

    print(
        f"Data Split ({selection_method}): {len(train_df)} training points, {len(test_df)} test points."
    )
    return train_df, test_df, t_max, x_max, pv_ids


def generate_all_collocation_epochs(
    num_points, num_epochs, t_max, x_max, device, seed=42
):
    print(
        f"Pre-computing LHS collocation points for {num_epochs} epochs"
        f" ({num_points} points/epoch)..."
    )
    total_points = num_points * num_epochs
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    sample = sampler.random(n=total_points)

    t_colloc_np = sample[:, 0] * t_max
    x_colloc_np = sample[:, 1] * x_max

    t_colloc_tensor = (
        torch.tensor(t_colloc_np, dtype=torch.float32)
        .view(num_epochs, num_points, 1)
        .to(device)
    )
    x_colloc_tensor = (
        torch.tensor(x_colloc_np, dtype=torch.float32)
        .view(num_epochs, num_points, 1)
        .to(device)
    )

    return t_colloc_tensor, x_colloc_tensor


# ==========================================
# 4. Adaptive Weight Calculation (Méthode Colleau Temporelle)
# ==========================================
def calculate_pv_weights(train_df, max_weight_cap=5.0):
    x_min = train_df["Position_km"].min()
    x_max = train_df["Position_km"].max()
    milestones = np.arange(x_min, x_max + 0.01, 0.01)

    crossing_times = []
    grouped = train_df.groupby("Vehicle_ID")
    for pv_id, group in grouped:
        group = group.sort_values("Time_s")
        group = group.drop_duplicates(subset=["Position_km"], keep="first")

        x_pv = group["Position_km"].values
        t_pv = group["Time_s"].values

        if len(x_pv) < 2:
            continue

        valid_milestones = milestones[
            (milestones >= x_pv.min()) & (milestones <= x_pv.max())
        ]
        t_cross = np.interp(valid_milestones, x_pv, t_pv)

        for m, t in zip(valid_milestones, t_cross):
            crossing_times.append({"Milestone": m, "Vehicle_ID": pv_id, "Time_s": t})

    cross_df = pd.DataFrame(crossing_times)
    cross_df = cross_df.sort_values(["Milestone", "Time_s"])

    cross_df["gap_front"] = cross_df.groupby("Milestone")["Time_s"].diff()
    cross_df["gap_behind"] = cross_df.groupby("Milestone")["Time_s"].diff(-1).abs()

    cross_df["gap_front"] = cross_df["gap_front"].fillna(cross_df["gap_behind"])
    cross_df["gap_behind"] = cross_df["gap_behind"].fillna(cross_df["gap_front"])

    cross_df["mean_gap_at_m"] = (cross_df["gap_front"] + cross_df["gap_behind"]) / 2.0
    pv_mean_time_gaps = cross_df.groupby("Vehicle_ID")["mean_gap_at_m"].mean()

    raw_weights = 1.0 / (pv_mean_time_gaps**3 + 1e-6)
    normalized_weights = raw_weights / raw_weights.mean()
    clipped_weights = np.clip(normalized_weights, a_min=0.1, a_max=max_weight_cap)

    final_weights_dict = clipped_weights.to_dict()
    train_df_out = train_df.copy()
    train_df_out["Weight"] = train_df_out["Vehicle_ID"].map(final_weights_dict)
    train_df_out["Weight"] = train_df_out["Weight"].fillna(1.0)

    return train_df_out


# ==========================================
# 5. Training Function
# ==========================================
def train_pinn_with_weights(
    csv_path,
    t_colloc_all,
    x_colloc_all,
    epochs=3000,
    num_colloc=10000,
    mu=0.1,
    gamma=0.05,
    v_max=50.0,
    use_adaptive=False,
    seed=42,
    selection_method="random",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mesure du temps de chargement et split des données
    train_df, test_df, t_max, x_max, pv_ids = load_and_split_data(
        csv_path,
        pv_ratio=0.10,
        v_max=v_max,
        seed=seed,
        selection_method=selection_method,
    )

    if use_adaptive:
        train_df = calculate_pv_weights(train_df)
        w_data = (
            torch.tensor(train_df["Weight"].values, dtype=torch.float32)
            .view(-1, 1)
            .to(device)
        )
    else:
        w_data = torch.ones((len(train_df), 1), dtype=torch.float32).to(device)

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

    model = TrafficPINN(t_max, x_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs_logged = []
    loss_total_history = []
    loss_data_history = []
    loss_phys_history = []
    cee_history = []

    best_cee = float("inf")
    best_model_state = None

    # [AJOUT] Début de la mesure du temps d'entraînement global
    start_train_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        rho_pred = model(t_data, x_data)
        loss_data = torch.mean(w_data * (rho_pred - rho_data) ** 2)

        t_colloc = t_colloc_all[epoch]
        x_colloc = x_colloc_all[epoch]

        loss_phys = compute_physics_loss(model, t_colloc, x_colloc, v_max, gamma)
        loss_total = mu * loss_data + (1.0 - mu) * loss_phys

        loss_total.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # [AJOUT] Calcul précis du temps d'inférence spécifique
                start_inf = time.time()
                rho_test_pred = model(t_test, x_test)
                inference_time_step = time.time() - start_inf

                # Calcul formel de la CEE demandée sur l'échantillon test non observé
                cee = torch.mean((rho_test_pred - rho_test) ** 2).item()

                if cee < best_cee:
                    best_cee = cee
                    best_model_state = model.state_dict()

            epochs_logged.append(epoch)
            loss_total_history.append(loss_total.item())
            loss_data_history.append(loss_data.item())
            loss_phys_history.append(loss_phys.item())
            cee_history.append(cee)

    # [AJOUT] Calcul final de la durée totale d'entraînement
    total_training_time = time.time() - start_train_time

    model.load_state_dict(best_model_state)

    # [AJOUT] Sauvegarde systématique dans un fichier .pt explicite
    db_name = os.path.splitext(os.path.basename(csv_path))[0]
    results_to_save = {
        "model_state": best_model_state,
        "epochs_logged": epochs_logged,
        "loss_total_history": loss_total_history,
        "loss_data_history": loss_data_history,
        "loss_phys_history": loss_phys_history,
        "cee_history": cee_history,
        "best_cee": best_cee,
        "training_time": total_training_time,
        "selection_method": selection_method,
        "use_adaptive_weights": use_adaptive,
    }

    save_filename = f"Pinn_epochs{epochs}_mu{mu}_method-{selection_method}_weights-{use_adaptive}_{db_name}.pt"
    torch.save(results_to_save, save_filename)
    print(
        f"Modèle PINN enregistré sous: {save_filename} | Temps d'entrainement: {total_training_time:.2f}s"
    )

    return (
        model,
        epochs_logged,
        loss_total_history,
        loss_data_history,
        loss_phys_history,
        cee_history,
        total_training_time,
    )

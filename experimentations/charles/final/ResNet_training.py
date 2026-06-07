import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ResNet_model import TrafficResNet, project_alpha


def calculate_pv_weights(df, pv_ids, max_weight_cap=5.0):
    x_min, x_max = df["Position_km"].min(), df["Position_km"].max()
    milestones = np.arange(x_min, x_max + 0.01, 0.01)
    crossing_times = []
    grouped = df[df["Vehicle_ID"].isin(pv_ids)].groupby("Vehicle_ID")

    for pv_id, group in grouped:
        group = group.sort_values("Time_s").drop_duplicates(
            subset=["Position_km"], keep="first"
        )
        x_pv, t_pv = group["Position_km"].values, group["Time_s"].values
        if len(x_pv) < 2:
            continue
        valid_milestones = milestones[
            (milestones >= x_pv.min()) & (milestones <= x_pv.max())
        ]
        t_cross = np.interp(valid_milestones, x_pv, t_pv)
        for m, t in zip(valid_milestones, t_cross):
            crossing_times.append({"Milestone": m, "Vehicle_ID": pv_id, "Time_s": t})

    if not crossing_times:
        return torch.ones(len(pv_ids), dtype=torch.float32)

    cross_df = pd.DataFrame(crossing_times).sort_values(["Milestone", "Time_s"])
    cross_df["gap_front"] = cross_df.groupby("Milestone")["Time_s"].diff()
    cross_df["gap_behind"] = cross_df.groupby("Milestone")["Time_s"].diff(-1).abs()
    cross_df["gap_front"] = cross_df["gap_front"].fillna(cross_df["gap_behind"])
    cross_df["gap_behind"] = cross_df["gap_behind"].fillna(cross_df["gap_front"])
    cross_df["mean_gap"] = (cross_df["gap_front"] + cross_df["gap_behind"]) / 2.0

    mean_gaps = cross_df.groupby("Vehicle_ID")["mean_gap"].mean()
    raw_weights = 1.0 / (mean_gaps**3 + 1e-6)
    normalized = raw_weights / raw_weights.mean()
    clipped = np.clip(normalized, 0.1, max_weight_cap)

    weight_dict = dict(enumerate(clipped))
    weights_array = np.array([weight_dict.get(pid, 1.0) for pid in pv_ids])
    return torch.tensor(weights_array, dtype=torch.float32)


def train_resnet(config):
    df = pd.read_csv(config["DATA_PATH"])
    unique_vehicles = np.sort(df["Vehicle_ID"].unique())
    num_pvs = max(2, int(config["PORTION_PROBE"] * config["N_TOTAL"]))

    if config["METHOD"] == "random":
        np.random.seed(config["SEED"])
        pv_ids = np.sort(np.random.choice(unique_vehicles, num_pvs, replace=False))
        weights = torch.ones(num_pvs - 1, dtype=torch.float32)
    else:
        np.random.seed(config["SEED"])
        pv_ids = np.sort(np.random.choice(unique_vehicles, num_pvs, replace=False))
        weights = calculate_pv_weights(df, pv_ids)
        weights = weights[:-1]

    n_gaps = num_pvs - 1
    t_min_s, t_max_s = df["Time_s"].min(), df["Time_s"].max()
    T_h = (t_max_s - t_min_s) / 3600.0
    dt_h = (
        np.sort(df["Time_s"].unique())[1] - np.sort(df["Time_s"].unique())[0]
    ) / 3600.0
    num_steps = int(round(T_h / dt_h))

    # Vérité terrain
    alpha_true = np.diff(pv_ids)

    df_pvs = df[df["Vehicle_ID"].isin(pv_ids)].pivot(
        index="Time_s", columns="Vehicle_ID", values="Position_km"
    )
    y_target_history = torch.tensor(df_pvs.values[:, :-1], dtype=torch.float32)

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
    z_bar = torch.min(gap_0 / config["L_V"], gap_T / config["L_V"])

    model = TrafficResNet(
        n_gaps,
        config["N_TOTAL"],
        config["L_V"],
        config["V_MAX"],
        config["RHO_MAX"],
        dt_h,
        num_steps,
        x_0_followers,
        x_0_leader,
        z_bar,
    )
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    loss_history, cee_history, mae_history, alpha_history = [], [], [], []
    start_train = time.time()
    density_true = alpha_true / (y_bar[1:] - y_bar[:-1]).numpy()

    for epoch in range(config["EPOCHS"]):
        optimizer.zero_grad()
        x_pred_history = model(require_grad_history=True)

        if config["METHOD"] == "adaptative":
            loss = torch.mean(weights * (x_pred_history - y_target_history) ** 2)
        else:
            loss = nn.MSELoss()(x_pred_history, y_target_history)

        loss.backward()
        optimizer.step()
        model.alpha.data = project_alpha(model.alpha.data, model.z_bar, model.N)
        loss_history.append(loss.item())

        if epoch % 10 == 0 or epoch == config["EPOCHS"] - 1:
            with torch.no_grad():
                alpha_curr = model.alpha.detach().numpy().copy()
                alpha_history.append(alpha_curr)

                _, hist_f, hist_l, _ = model(return_history=True)
                gaps = np.append(
                    hist_f[-1][1:] - hist_f[-1][:-1], hist_l[-1] - hist_f[-1][-1]
                )
                rho_pred = alpha_curr / (gaps + 1e-6)

                cee_history.append(np.mean((rho_pred - density_true) ** 2))
                mae_history.append(np.mean(np.abs(alpha_curr - alpha_true)))

    train_time = time.time() - start_train
    start_inf = time.time()
    _, hist_followers, hist_leader, intermediate_rhos = model(return_history=True)
    inf_time = time.time() - start_inf

    return {
        "model": "ResNet",
        "method": config["METHOD"],
        "loss_train": loss_history,
        "cee_val": cee_history,
        "mae_val": mae_history,
        "alpha_history": alpha_history,
        "alpha_pred": model.alpha.detach().numpy(),
        "alpha_true": alpha_true,
        "rho_final_pred": rho_pred,
        "rho_final_true": density_true,
        "train_time": train_time,
        "inf_time": inf_time,
        "cee_final": cee_history[-1],
        "params": model.alpha.numel(),
    }

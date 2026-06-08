import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


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
        t_norm = 2.0 * (t / self.t_max) - 1.0
        x_norm = 2.0 * (x / self.x_max) - 1.0

        inputs = torch.cat([t_norm, x_norm], dim=1)
        for layer in self.hidden_layers:
            inputs = layer(inputs)

        return self.sigmoid(self.output_layer(inputs))


# ==========================================
# 2. Physics Loss Computation
# ==========================================
def compute_physics_loss(model, t_colloc, x_colloc, v_max, gamma=0.01):
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

    flux_derivative = v_max * (1.0 - 2.0 * rho)
    residual = rho_t + flux_derivative * rho_x - gamma * rho_xx

    return torch.mean(residual**2)


# ==========================================
# 3. Data Loading & LHS Generation
# ==========================================
def load_and_split_data(csv_path, pv_ratio=0.05, v_max=50.0, seed=42):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    df["Time_h"] = df["Time_s"] / 3600.0
    df["Density"] = 1.0 - (df["Velocity_kmh"] / v_max)

    t_max = df["Time_h"].max()
    x_max = df["Position_km"].max()

    unique_vehicles = df["Vehicle_ID"].unique()
    num_pvs = int(len(unique_vehicles) * pv_ratio)

    np.random.seed(seed)
    pv_ids = np.random.choice(unique_vehicles, num_pvs, replace=False)

    train_df = df[df["Vehicle_ID"].isin(pv_ids)]
    test_df = df[~df["Vehicle_ID"].isin(pv_ids)]

    print(
        f"Data Split: {len(train_df)} training points ({pv_ratio * 100}%), {len(test_df)} test points."
    )
    return train_df, test_df, t_max, x_max, pv_ids


def generate_all_collocation_epochs(
    num_points, num_epochs, t_max, x_max, device, seed=42
):
    print(
        f"Pre-computing LHS collocation points for {num_epochs} epochs ({num_points} points/epoch)..."
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
# 4. Training
# ==========================================
def train_pinn_with_history(
    csv_path,
    t_colloc_all,
    x_colloc_all,
    epochs=1500,
    mu=0.99,
    gamma=0.01,
    v_max=50.0,
    seed=42,
    save_dir=".",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_df, test_df, t_max, x_max, _ = load_and_split_data(
        csv_path, pv_ratio=0.05, v_max=v_max, seed=seed
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

    model = TrafficPINN(t_max, x_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs_logged = []
    loss_total_history = []
    loss_data_history = []
    loss_phys_history = []
    cee_history = []

    best_cee = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        rho_pred = model(t_data, x_data)
        loss_data = torch.mean((rho_pred - rho_data) ** 2)

        t_colloc = t_colloc_all[epoch]
        x_colloc = x_colloc_all[epoch]
        loss_phys = compute_physics_loss(model, t_colloc, x_colloc, v_max, gamma)

        loss_total = mu * loss_data + (1.0 - mu) * loss_phys

        loss_total.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                rho_test_pred = model(t_test, x_test)
                cee = torch.mean((rho_test_pred - rho_test) ** 2).item()

                if cee < best_cee:
                    best_cee = cee
                    best_model_state = model.state_dict()

            epochs_logged.append(epoch)
            loss_total_history.append(loss_total.item())
            loss_data_history.append(loss_data.item())
            loss_phys_history.append(loss_phys.item())
            cee_history.append(cee)

            print(
                f"Epoch {epoch:04d} | Total: {loss_total.item():.6f} | Data: {loss_data.item():.6f} | Phys: {loss_phys.item():.6f} | CEE: {cee:.6f}"
            )

    model.load_state_dict(best_model_state)
    print(f"Training finished. Best CEE: {best_cee:.6f}")

    # Save the model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "best_pinn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return (
        model,
        epochs_logged,
        loss_total_history,
        loss_data_history,
        loss_phys_history,
        cee_history,
        t_max,
        x_max,
    )


# ==========================================
# 5. Plotting and Saving Functions
# ==========================================
def save_training_data(save_dir, epochs_logged, loss_total, cee_history):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(
        {"Epoch": epochs_logged, "Loss_Total": loss_total, "CEE": cee_history}
    )
    data_path = os.path.join(save_dir, "training_metrics.csv")
    df.to_csv(data_path, index=False)
    print(f"Training metrics saved to {data_path}")


def plot_loss_and_cee(save_dir, csv_path="training_metrics.csv"):
    full_csv_path = os.path.join(save_dir, csv_path)
    if not os.path.exists(full_csv_path):
        print("Data file not found for plotting.")
        return

    df = pd.read_csv(full_csv_path)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:red"
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Total Loss (Log)", color=color, fontsize=12)
    ax1.plot(
        df["Epoch"],
        df["Loss_Total"],
        color=color,
        label="Total Loss (Train)",
        linewidth=2,
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("CEE / Validation Error (Log)", color=color, fontsize=12)
    ax2.plot(
        df["Epoch"],
        df["CEE"],
        color=color,
        label="CEE (Validation)",
        linestyle="--",
        linewidth=2,
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_yscale("log")

    fig.tight_layout()
    plt.title(
        "Evolution of Training Loss and Validation Error",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(os.path.join(save_dir, "loss_cee_evolution.png"), dpi=150)
    plt.close()
    print("Saved Loss/CEE plot.")


def save_and_plot_density_heatmap(
    model, t_max, x_max, save_dir, nx=200, nt=200, rho_max=200.0
):
    device = next(model.parameters()).device

    x_vals = np.linspace(0, x_max, nx)
    t_vals = np.linspace(0, t_max, nt)

    X, T = np.meshgrid(x_vals, t_vals)

    x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_tensor = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    model.eval()
    with torch.no_grad():
        rho_pred = model(t_tensor, x_tensor).cpu().numpy().reshape(nt, nx)

    rho_pred_veh_km = rho_pred * rho_max

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "reconstructed_density_grid.npy"), rho_pred_veh_km)
    np.save(os.path.join(save_dir, "x_grid.npy"), x_vals)
    np.save(os.path.join(save_dir, "t_grid.npy"), t_vals)
    print("Density grid data saved.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    # shading="nearest" corrige l'erreur de pcolormesh
    mesh = ax.pcolormesh(
        X, T, rho_pred_veh_km, cmap="magma", vmin=0, vmax=rho_max, shading="nearest"
    )

    fig.colorbar(mesh, ax=ax, label="ρ [veh/km]", fraction=0.046)

    ax.set_title(
        "Carte de chaleur spatio-temporelle de la Densité (PINN)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Position $x$ [km]", fontsize=12)
    ax.set_ylabel("Temps $t$ [h]", fontsize=12)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "reconstructed_density_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved reconstructed density heatmap.")


# ==========================================
# Run the Code
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION DES CHEMINS ABSOLUS ---
    BASE_DIR = r"C:\Users\charl\OneDrive\Documents\PontsEtChaussees\2A\PROJET\code-projet-IMI\Modelisation-du-trafic-routier"

    csv_file = os.path.join(
        BASE_DIR,
        "experimentations",
        "charles",
        "ResNet_alpha",
        "datasets",
        "traffic_N1000_rarefaction.csv",
    )
    save_directory = os.path.join(
        BASE_DIR, "experimentations", "charles", "graphs", "pinn_results"
    )
    model_path = os.path.join(save_directory, "best_pinn_model.pth")
    metrics_path = os.path.join(save_directory, "training_metrics.csv")

    if os.path.exists(csv_file):
        df_temp = pd.read_csv(csv_file)
        t_max_global = (df_temp["Time_s"] / 3600.0).max()
        x_max_global = df_temp["Position_km"].max()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vérifie l'existence du modèle et des métriques
        if os.path.exists(model_path) and os.path.exists(metrics_path):
            print(f"Loading existing model from {model_path}...")
            model = TrafficPINN(t_max_global, x_max_global).to(device)
            # weights_only=True évite les warnings PyTorch liés à la sécurité
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )

            print("Model loaded. Generating plots...")
            plot_loss_and_cee(save_directory)
            save_and_plot_density_heatmap(
                model, t_max_global, x_max_global, save_directory
            )

        else:
            print("Model or metrics not found. Training from scratch...")
            epochs = 1500
            num_colloc = 20000
            mu = 0.99
            gamma = 0.01

            t_colloc_all, x_colloc_all = generate_all_collocation_epochs(
                num_points=num_colloc,
                num_epochs=epochs,
                t_max=t_max_global,
                x_max=x_max_global,
                device=device,
            )

            (
                model,
                epochs_logged,
                loss_total_history,
                loss_data_history,
                loss_phys_history,
                cee_history,
                t_max,
                x_max,
            ) = train_pinn_with_history(
                csv_file,
                t_colloc_all,
                x_colloc_all,
                epochs=epochs,
                mu=mu,
                gamma=gamma,
                seed=42,
                save_dir=save_directory,
            )

            save_training_data(
                save_directory, epochs_logged, loss_total_history, cee_history
            )
            print("Training complete. Generating plots...")
            save_and_plot_density_heatmap(model, t_max, x_max, save_directory)
            plot_loss_and_cee(save_directory)

    else:
        print(f"Dataset not found at: {csv_file}")
